import torch
import torch.nn as nn
import math
import numpy as np

from .balancer import Balancer
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F

# based on:
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py


class DDNLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 fg_weight=13,
                 bg_weight=1,
                 downsample_factor=1,
                 depth_min=1e-3,
                 depth_max=60.0,
                 depth_mode="LID",
                 sigma_bins=1.0):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma

        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.depth_mode = str(depth_mode)
        self.sigma_bins = float(sigma_bins)

    def build_target_depth_from_3dcenter(self, depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img):
        B, _, H, W = depth_logits.shape
        depth_maps = torch.full((B, H, W), 60.0, device=depth_logits.device, dtype=depth_logits.dtype)

        # Set box corners
        gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
        gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
        gt_boxes2d = gt_boxes2d.long()

        # Set all values within each box to True
        gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
        gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
        B = len(gt_boxes2d)
        for b in range(B):
            center_depth_per_batch = gt_center_depth[b]
            center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
            gt_boxes_per_batch = gt_boxes2d[b][sorted_idx]
            for n in range(gt_boxes_per_batch.shape[0]):
                u1, v1, u2, v2 = gt_boxes_per_batch[n]
                depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]

        return depth_maps
    def build_target_depth_from_3dcenter_gaussian(
        self,
        depth_logits: torch.Tensor,
        gt_boxes2d: torch.Tensor,
        gt_center_depth: torch.Tensor,
        num_gt_per_img: list,
        gt_centers3d: torch.Tensor,
        *,
        sigma_scale: float = 0.5,
        min_sigma: float = 2.0,
        center_anchor_size: int = 3,
        depth_max: float = 60.0
    ):
        B, _, H, W = depth_logits.shape
        device, dtype = depth_logits.device, depth_logits.dtype

        depth_maps = torch.full((B, H, W), depth_max, device=device, dtype=dtype)
        boxes_per_img = gt_boxes2d.split(num_gt_per_img, dim=0)
        depths_per_img = gt_center_depth.split(num_gt_per_img, dim=0)
        centers_per_img = gt_centers3d.split(num_gt_per_img, dim=0)

        for b in range(B):
            bboxes = boxes_per_img[b]
            cdepths = depths_per_img[b]
            centers = centers_per_img[b]

            if bboxes.numel() == 0:
                continue

            sort_idx = torch.argsort(cdepths, dim=0, descending=True)
            bboxes = bboxes[sort_idx]
            cdepths = cdepths[sort_idx]
            centers = centers[sort_idx]
            for n in range(bboxes.shape[0]):
                u1_f, v1_f, u2_f, v2_f = bboxes[n]
                cx, cy = centers[n]
                cdepth = cdepths[n]
                
                u1 = int(torch.floor(u1_f))
                v1 = int(torch.floor(v1_f))
                u2 = int(torch.ceil(u2_f))
                v2 = int(torch.ceil(v2_f))

                u1, u2 = max(0, u1), min(W, u2)
                v1, v2 = max(0, v1), min(H, v2)
                
                box_h, box_w = v2 - v1, u2 - u1
                if box_h <= 0 or box_w <= 0:
                    continue

                ys = torch.arange(v1, v2, device=device, dtype=dtype)
                xs = torch.arange(u1, u2, device=device, dtype=dtype)
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')

                sigma_x = max(box_w * sigma_scale, min_sigma)
                sigma_y = max(box_h * sigma_scale, min_sigma)

                gx = torch.exp(-torch.pow(xx - cx, 2) / (2 * sigma_x**2))
                gy = torch.exp(-torch.pow(yy - cy, 2) / (2 * sigma_y**2))
                G = gy * gx

                local_depth = cdepth * G + depth_max * (1 - G)

                if center_anchor_size >= 1:
                    icx, icy = int(round(cx.item())), int(round(cy.item()))
                    half = center_anchor_size // 2
                    
                    patch_y0 = max(v1, icy - half)
                    patch_y1 = min(v2, icy + half + 1)
                    patch_x0 = max(u1, icx - half)
                    patch_x1 = min(u2, icx + half + 1)
                    
                    local_py0 = int(patch_y0 - v1)
                    local_py1 = int(patch_y1 - v1)
                    local_px0 = int(patch_x0 - u1)
                    local_px1 = int(patch_x1 - u1)

                    if local_py1 > local_py0 and local_px1 > local_px0:
                        local_depth[local_py0:local_py1, local_px0:local_px1] = cdepth

                current_depth_in_box = depth_maps[b, v1:v2, u1:u2]
                depth_maps[b, v1:v2, u1:u2] = torch.minimum(current_depth_in_box, local_depth)
        
        return depth_maps
    
    def gaussian_kernel(self, size: int, sigma: float):
        """Function to create a 1D Gaussian kernel."""
        x = torch.arange(-size // 2 + 1., size // 2 + 1.)
        kernel = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
    
    def build_weighted_depth_from_logits(self, depth_logits):
        depth_num_bins = 80
        depth_min = 1e-3
        depth_max = 60.0
        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False).to(depth_logits.device)
        
        '''
        # Create Gaussian kernel
        kernel_size = int(2 * self.sigma + 1)  # Ensure the kernel size is odd
        gaussian_kernel = self.gaussian_kernel(kernel_size, self.sigma).to(depth_logits.device)
        
        # Expand the kernel to apply it across depth_num_bins dimension
        gaussian_kernel = gaussian_kernel.view(1, 1, -1, 1, 1)  # Shape: (1, 1, kernel_size, 1, 1)

        # Pad the depth_logits tensor to apply the convolution
        padding = (kernel_size // 2, kernel_size // 2)
        padded_logits = F.pad(depth_logits.unsqueeze(2), (0, 0, 0, 0, padding[0], padding[1]), mode='reflect')
        
        # Apply the Gaussian filter along the depth_num_bins dimension
        smoothed_logits = F.conv3d(padded_logits, gaussian_kernel, groups=depth_logits.size(0)).squeeze(2)
        '''
        
        # Calculate the median value along the depth_num_bins dimension
        # Apply the threshold: set values below the median to zero
        
        
        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        return weighted_depth
    
    
    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                      (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)
       
        # Apply Gaussian noise to the depth indices
        #noise = torch.normal(mean=0, std=2, size=indices.shape, device=indices.device)
        #indices = indices +  noise.round().long()

        # Clamp indices to be within the valid range
        #indices = indices.clamp(0, num_bins)
        
        return indices

    def metric_depth_to_soft_bins(self, depth_maps, num_bins, depth_min, depth_max, sigma_bins=1.5, eps=1e-3):
        B, H, W = depth_maps.shape
        device, dtype = depth_maps.device, depth_maps.dtype

        mask_ignore = (
            (depth_maps < depth_min) |
            (depth_maps > depth_max) |
            (depth_maps >= depth_max - eps) |
            (~torch.isfinite(depth_maps))
        )

        idx_float = self.bin_depths(
            depth_maps,
            mode=self.depth_mode,
            depth_min=depth_min,
            depth_max=depth_max,
            num_bins=num_bins,
            target=False,
        )
        idx_float = torch.where(mask_ignore, torch.zeros_like(idx_float), idx_float)

        bin_centers = torch.arange(num_bins, device=device, dtype=dtype).view(1, num_bins, 1, 1)
        idx = idx_float.unsqueeze(1)
        sigma = max(float(sigma_bins), 1e-6)
        dist2 = (bin_centers - idx).pow(2)
        q = torch.exp(-0.5 * dist2 / (sigma * sigma))
        q_sum = q.sum(dim=1, keepdim=True).clamp(min=1e-6)
        q = q / q_sum

        q_probs = torch.zeros((B, num_bins + 1, H, W), device=device, dtype=dtype)
        q_probs[:, :num_bins] = q
        if mask_ignore.any():
            q_probs[:, :num_bins][mask_ignore.unsqueeze(1).expand_as(q)] = 0.0
            q_probs[:, num_bins][mask_ignore] = 1.0

        return q_probs
    def forward(self, depth_logits, gt_boxes2d, num_gt_per_img, gt_center_depth, gt_3dcenter):
        """
        Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """

        # Bin depth map to create target
        depth_maps = self.build_target_depth_from_3dcenter_gaussian(
            depth_logits,
            gt_boxes2d,
            gt_center_depth,
            num_gt_per_img,
            gt_3dcenter,
            sigma_scale=0.25,
            min_sigma=3.0,
            center_anchor_size=3,
            depth_max=60.0
        )
        num_bins = depth_logits.shape[1] - 1
        q_probs = self.metric_depth_to_soft_bins(
            depth_maps,
            num_bins=num_bins,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            sigma_bins=self.sigma_bins,
        )
        log_p = F.log_softmax(depth_logits, dim=1)
        loss_map = -(q_probs * log_p).sum(dim=1)
        loss = self.balancer(loss=loss_map, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)
        return loss
