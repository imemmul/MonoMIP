import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MSFeatureFusionAttn(nn.Module):
    def __init__(self, in_channels_list, out_channels, target_level=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.target_level = target_level
        self.num_levels = len(in_channels_list)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels_list
        ])
        self.attn_logits = nn.ModuleList([
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=True)
            for _ in in_channels_list
        ])

    def forward(self, feats, target_size=None):
        if target_size is None:
            target_size = feats[self.target_level].shape[-2:]
        proj_feats = []
        logits = []
        for i in range(self.num_levels):
            x = self.proj[i](feats[i])
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            proj_feats.append(x)
            logits.append(self.attn_logits[i](x))
        attn = torch.cat(logits, dim=1)
        attn = F.softmax(attn, dim=1)
        fused = 0
        for i in range(self.num_levels):
            fused = fused + proj_feats[i] * attn[:, i:i+1]
        return fused


class MHIPM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = int(cfg["hidden_dim"])

        self.bev_shape = tuple(cfg.get("bev_shape", (96, 80)))
        self.bev_range = tuple(cfg.get("bev_range", (1e-3, 60.0, -30.0, 30.0)))
        self.num_feature_levels = int(cfg["bev_num_feature_levels"])

        self.splat_no_grad = bool(cfg.get("splat_no_grad", False))

        self.ipm_heights = list(cfg.get("ipm_heights", [-0.5, 0.5, 1.0, 1.65]))
        self.mh_representation = cfg.get("mh_representation", "ground_contrast")

        num_levels = int(cfg.get("num_feature_levels", 4))

        self.ms_fusion = MSFeatureFusionAttn(
            [d_model] * num_levels,
            d_model,
            target_level=1,
            norm_layer=lambda c: nn.GroupNorm(32, c),
        )

        num_heights = len(self.ipm_heights)
        self.bev_norms = nn.ModuleList([nn.GroupNorm(32, d_model) for _ in range(num_heights)])

        self.bev_map_classifier = nn.Conv2d(d_model, 1, 1)


    def _split_memory_to_maps(self, memory_2d: torch.Tensor, spatial_shapes: torch.Tensor) -> List[torch.Tensor]:
        B, _, C = memory_2d.shape
        feats = []
        start = 0
        for h, w in spatial_shapes.tolist():
            length = int(h * w)
            feat = memory_2d[:, start:start + length].transpose(1, 2).contiguous().view(B, C, int(h), int(w))
            feats.append(feat)
            start += length
        return feats
    
    def _ipm_bev(self, feat_l, calib, bev_edges_x, bev_edges_z, img_wh_full):
        B, C, Hs, Ws = feat_l.shape
        device = feat_l.device
        dtype = feat_l.dtype

        Uz = bev_edges_z.numel() - 1
        Ux = bev_edges_x.numel() - 1

        z_centers = 0.5 * (bev_edges_z[:-1] + bev_edges_z[1:])
        x_centers = 0.5 * (bev_edges_x[:-1] + bev_edges_x[1:])

        Z_bev, X_bev = torch.meshgrid(z_centers, x_centers, indexing="ij")
        Z_bev = Z_bev.unsqueeze(0).expand(B, -1, -1)
        X_bev = X_bev.unsqueeze(0).expand(B, -1, -1)

        fx = calib[:, 0, 0].view(B, 1, 1)
        fy = calib[:, 1, 1].view(B, 1, 1)
        cx = calib[:, 0, 2].view(B, 1, 1)
        cy = calib[:, 1, 2].view(B, 1, 1)
        tx = (calib[:, 0, 3].view(B, 1, 1) / (-fx)).to(torch.float32)
        ty = (calib[:, 1, 3].view(B, 1, 1) / (-fy)).to(torch.float32)
        W_full = img_wh_full[:, 0].to(device=device, dtype=torch.float32).view(B, 1, 1)
        H_full = img_wh_full[:, 1].to(device=device, dtype=torch.float32).view(B, 1, 1)

        inv_Z = 1.0 / Z_bev.clamp(min=1e-3)

        grid_x = 2.0 * (fx * (X_bev - tx) * inv_Z + cx) / W_full - 1.0

        height_feats = []
        height_masks = []

        for Y_cam in self.ipm_heights:
            v_px = fy * (Y_cam - ty) * inv_Z + cy
            grid_y = 2.0 * v_px / H_full - 1.0

            grid = torch.stack([grid_x, grid_y], dim=-1)

            feat_k = F.grid_sample(
                feat_l, grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )

            in_bounds = (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0)
            feat_k = feat_k * in_bounds.unsqueeze(1).to(dtype)

            height_feats.append(feat_k)
            height_masks.append(~in_bounds)

        return height_feats, height_masks

    def forward(
        self,
        image_features_list: List[torch.Tensor],
        spatial_shapes,
        depth_logits: torch.Tensor,
        calib: torch.Tensor,
        img_sizes: torch.Tensor,
        align_corners: bool = True,
    ):
        B = depth_logits.shape[0]
        device = depth_logits.device
        Uz, Ux = self.bev_shape
        Z_min, Z_max, X_min, X_max = self.bev_range

        image_features_list = self._split_memory_to_maps(image_features_list, spatial_shapes)
        target_size = image_features_list[1].shape[-2:]
        fused_img_feat = self.ms_fusion(image_features_list, target_size)

        bev_edges_x = torch.linspace(X_min, X_max, Ux + 1, device=device)
        bev_edges_z = torch.linspace(Z_min, Z_max, Uz + 1, device=device)

        if self.splat_no_grad:
            with torch.no_grad():
                bev_feats, bev_masks = self._ipm_bev(
                    fused_img_feat, calib, bev_edges_x, bev_edges_z, img_sizes,
                )
        else:
            bev_feats, bev_masks = self._ipm_bev(
                fused_img_feat, calib, bev_edges_x, bev_edges_z, img_sizes,
            )

        if len(bev_feats) == 4 and self.mh_representation == "ground_contrast":
            ground = bev_feats[3]
            bev_feats = [
                ground,
                bev_feats[2] - ground,
                bev_feats[1] - ground,
                bev_feats[0] - ground,
            ]
            bev_masks = [
                bev_masks[3],
                bev_masks[2] | bev_masks[3],
                bev_masks[1] | bev_masks[3],
                bev_masks[0] | bev_masks[3],
            ]
        elif len(bev_feats) == 4 and self.mh_representation == "diagonal_correlation":
            ground = bev_feats[3]
            ground_unit = F.normalize(ground, dim=1, eps=1e-6)
            upper_planes = [bev_feats[2], bev_feats[1], bev_feats[0]]
            bev_feats = [ground] + [
                F.normalize(upper, dim=1, eps=1e-6) * ground_unit
                for upper in upper_planes
            ]
            bev_masks = [
                bev_masks[3],
                bev_masks[2] | bev_masks[3],
                bev_masks[1] | bev_masks[3],
                bev_masks[0] | bev_masks[3],
            ]
        elif len(bev_feats) == 4:
            raise ValueError(
                f"Unknown mh_representation: {self.mh_representation!r}"
            )

        bev_feats = [self.bev_norms[i](f) for i, f in enumerate(bev_feats)]

        if len(bev_feats) > 1:
            fused_bev = torch.stack(bev_feats, dim=0).mean(dim=0)
        else:
            fused_bev = bev_feats[0]
        pred_logits = self.bev_map_classifier(fused_bev)
        return bev_feats, pred_logits, bev_masks
