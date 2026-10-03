import torch
import torch.nn.functional as F
from torch import inverse, nn
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .position_encoding import build_position_encoding
from .det2d_transformer import build_det2d_transformer
from .bevdet_transformer import build_bevdet_transformer

from .depth_predictor import DepthPredictor
from .depth_predictor import MHIPM
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from .position_encoding import PositionEmbeddingCamRay


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MonoMIP(nn.Module):
    """ This is the MonoMIP module that performs monocular 3D object detection """
    def __init__(self, backbone, depth_predictor, ortho_predictor, det2d_transformer, bevdet_transformer,
                  num_classes, num_queries, num_feature_levels,
                  aux_loss=True, with_box_refine=False, init_box=False, group_num=11, cfg=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            det2d_transformer: transformer architecture. See det2d_transformer.py
            det3d_transformer: transformer architecture. See det3d_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()

        self.bev_feat_source = str(cfg.get('bev_feat_source', 'memory')) if cfg else 'memory'

        self.num_queries = num_queries
        self.det2d_transformer = det2d_transformer
        self.bevdet_transformer = bevdet_transformer
        self.ortho_predictor = ortho_predictor
        self.depth_predictor = depth_predictor
        hidden_dim = det2d_transformer.d_model
        self.hidden_dim = hidden_dim
        
        self.num_feature_levels = num_feature_levels
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        self.bev_embed = MLP(hidden_dim, hidden_dim, 5, 3)
        

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.query_embed_2d = nn.Embedding(num_queries * group_num, hidden_dim*2)


        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.num_classes = num_classes

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = det2d_transformer.decoder.num_layers + 1
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.bev_embed = _get_clones(self.bev_embed, num_pred)
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)

            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            for m in self.bev_embed:
                nn.init.constant_(m.layers[-1].weight.data[4], 0.0)
                nn.init.constant_(m.layers[-1].bias.data[4], 0.0)
            # hack implementation for iterative bounding box refinement
            self.det2d_transformer.decoder.bbox_embed = self.bbox_embed
            self.bevdet_transformer.decoder.bev_embed = self.bev_embed
            self.bevdet_transformer.decoder.ref_point_proj = self.bevdet_transformer.ref_point_proj

    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
       
        features, pos = self.backbone(images)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.training:
            query_embeds_2d = self.query_embed_2d.weight
        else:
            # only use one group in inference
            query_embeds_2d = self.query_embed_2d.weight[:self.num_queries]

        pred_depth_map_logits, depth_pos_embed, weighted_depth = self.depth_predictor(srcs, masks[1], pos[1])

        
        intermediate_output = self.det2d_transformer(srcs, masks, pos, query_embeds_2d, depth_pos_embed)
        hs_2d = intermediate_output['hs']
        init_reference_2d = intermediate_output['init_reference_out']
        inter_references_2d = intermediate_output['inter_references_out']
        inter_coords = []
        inter_classes = []
        inter_3d_dims = []
        inter_depths = []
        bev_classes = []
        bev_angles = []
        bev_3d_dims = []
        bev_boxes = []
        bev_depths = []
        
        for lvl in range(hs_2d.shape[0]):
            if lvl == 0:
                reference = init_reference_2d
            else:
                reference = inter_references_2d[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs_2d[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            output_coord = tmp.sigmoid()
            inter_coords.append(output_coord)

            # classes
            output_class = self.class_embed[lvl](hs_2d[lvl])
            inter_classes.append(output_class)
            size3d = self.dim_embed_3d[lvl](hs_2d[lvl])
            inter_3d_dims.append(size3d)
            box2d_height_norm = output_coord[:, :, 4] + output_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0]/ box2d_height * calibs[:, 0, 0].unsqueeze(1)
            depth_err = self.depth_embed[lvl](hs_2d[lvl])
            depth_ave = torch.cat([depth_geo.unsqueeze(-1) + depth_err[..., 0:1], depth_err[..., 1:2]], -1)
            inter_depths.append(depth_ave)
            
        inter_coord = torch.stack(inter_coords)
        inter_class = torch.stack(inter_classes)
        inter_3d_dim = torch.stack(inter_3d_dims)
        inter_depth = torch.stack(inter_depths)
        
        
        if self.bev_feat_source == 'image':
            bev_input_features = torch.cat([s.flatten(2) for s in srcs], dim=2).transpose(1, 2)
            bev_input_shapes = torch.tensor(
                [[s.shape[2], s.shape[3]] for s in srcs], device=srcs[0].device, dtype=torch.long
            )
        else:
            bev_input_features = intermediate_output['memory']
            bev_input_shapes = intermediate_output['spatial_shapes']

        bev_srcs, bev_logits, bev_masks = self.ortho_predictor(
            bev_input_features,
            bev_input_shapes,
            pred_depth_map_logits,
            calibs,
            img_sizes,
            align_corners=True
        )

        query_embeds = hs_2d[-1]
        init_depth = inter_depth[-1][:, :, 0:1].detach()
        init_logvar = inter_depth[-1][:, :, 1:2].detach()
        inter_ref_to_bev = inter_coord[-1].detach()
        inter_dim_to_bev = inter_3d_dim[-1].detach()
        bev_output = self.bevdet_transformer(
            bev_srcs,
            bev_masks,
            query_embeds,
            inter_ref_to_bev,
            inter_dim_to_bev,
            init_depth,
            img_sizes,
            calibs,
            init_logvar=init_logvar,
        )
        hs_bev = bev_output['hs']
        init_reference_bev = bev_output['init_reference_out']
        inter_referneces_bev = bev_output['inter_references_out']
        init_logvar_bev = bev_output['init_logvar_out']
        inter_logvars_bev = bev_output['inter_logvars_out']
        bev_range_z = (1e-3, 60.0)
        bev_range_x = (-30.0, 30.0)
        for lvl in range(hs_bev.shape[0]):
            if lvl == 0:
                reference_bev = init_reference_bev
                prev_logvar = init_logvar_bev
            else:
                reference_bev = inter_referneces_bev[lvl - 1]
                prev_logvar = inter_logvars_bev[lvl - 1]

            reference_bev = inverse_sigmoid(reference_bev)

            tmp_bev = self.bev_embed[lvl](hs_bev[lvl])
            if reference_bev.shape[-1] == 4:
                tmp_bev[..., :4] += reference_bev

            bev_coord = tmp_bev[..., :4].sigmoid()
            bev_class = self.class_embed[lvl](hs_bev[lvl])
            bev_classes.append(bev_class)
            bev_boxes.append(bev_coord)
            bev_z = bev_coord[..., 1] * (bev_range_z[1] - bev_range_z[0]) + bev_range_z[0]
            delta_logvar = torch.tanh(tmp_bev[..., 4:5])
            bev_logvar = prev_logvar + delta_logvar
            bev_depth_ave = torch.cat([bev_z.unsqueeze(-1), bev_logvar], -1)
            bev_depths.append(bev_depth_ave)
            size3d_w_norm = bev_coord[..., 2]
            size3d_l_norm = bev_coord[..., 3]
            size3d_w = size3d_w_norm * (bev_range_x[1] - bev_range_x[0])
            size3d_l = size3d_l_norm * (bev_range_z[1] - bev_range_z[0])
            bev_3d_dims.append(torch.stack([inter_dim_to_bev[..., 0], size3d_w, size3d_l], -1))
            angle = self.angle_embed[lvl](hs_bev[lvl])
            bev_angles.append(angle)
        
        outputs_class = torch.stack(bev_classes)
        outputs_coord = inter_coord
        outputs_bev_box = torch.stack(bev_boxes)
        outputs_3d_dim = torch.stack(bev_3d_dims)
        outputs_angle = torch.stack(bev_angles)
        outputs_depth = torch.stack(bev_depths)
        out = dict()
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes'] = outputs_coord[-1]
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_bev_boxes'] = outputs_bev_box[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_bev_confidence'] = bev_logits
        out['pred_depth_map_logits'] = pred_depth_map_logits
        out['inter_outputs'] = self._set_inter_loss(inter_class, inter_coord, inter_3d_dim, inter_depth)
        

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth, outputs_bev_box) 
        
        return out
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth, outputs_bev_box):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b,
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e, 'pred_bev_boxes': f}
                for a, b, c, d, e, f in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1], outputs_bev_box[:-1])]

    @torch.jit.unused
    def _set_inter_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_depth):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_3d_dim': c, 'pred_depth': d}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_3d_dim, outputs_depth)]
    @torch.jit.unused
    def _set_bev_loss(self, outputs_class, outputs_coord, outputs_angle, outputs_3d_dim, outputs_depth, outputs_bev_center):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_angle': c, 'pred_3d_dim': d, 'pred_depth': e, 'pred_bev_center': f}
                for a, b, c, d, e, f in zip(outputs_class, outputs_coord, outputs_angle, outputs_3d_dim, outputs_depth, outputs_bev_center)]


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoMIP.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses,
                 inter_losses, bev_losses, group_num=11,
                 use_in_radius_depth_nll=False,
                 in_radius_thresh=0.10,
                 in_radius_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.inter_losses = inter_losses
        self.bev_losses = bev_losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.bce = nn.BCELoss()
        self.bce_noReduce = nn.BCELoss(reduction='none')

        self.group_num = group_num
        self.use_in_radius_depth_nll = bool(use_in_radius_depth_nll)
        self.in_radius_thresh = float(in_radius_thresh)
        self.in_radius_weight = float(in_radius_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_bevcenter(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        pred_uv = outputs['pred_bev_boxes'][idx][..., :2]
        gt_uv = torch.cat([torch.tensor(t['bev_boxes'][..., 0:2], device=pred_uv.device)[i]
                        for t, (_, i) in zip(targets, indices)], 0)
        loss_bev_center = F.l1_loss(pred_uv, gt_uv, reduction='none')
        
        return {'loss_bev_center': loss_bev_center.sum() / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_bev_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_bev_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bevboxes = outputs['pred_bev_boxes'][:, :, 0: 4][idx]
        target_bevboxes = torch.cat([t['bev_boxes'][:, 0: 4][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_bevboxes, target_bevboxes, reduction='none')
        losses = {}
        losses['loss_bev_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        bev_range_x = (-30.0, 30.0)
        bev_range_z = (1e-3, 60.0)
        if src_bevboxes.numel() > 0:
            pred_angle_logits = outputs['pred_angle'][idx]
            pred_boxes3d = outputs['pred_boxes'][:, :, 0:2][idx]
            target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0).view(-1)
            target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0).view(-1)
            target_heading_cls = target_heading_cls.to(device=src_bevboxes.device, dtype=torch.long)
            target_heading_res = target_heading_res.to(device=src_bevboxes.device, dtype=src_bevboxes.dtype)
            target_boxes3d = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_bevboxes.device, dtype=src_bevboxes.dtype)
            img_sizes_list = []
            calib_list = []
            for t, (_, matched_ids) in zip(targets, indices):
                if matched_ids.numel() == 0:
                    continue
                size_tensor = torch.as_tensor(t['img_size'], device=src_bevboxes.device, dtype=src_bevboxes.dtype)
                img_sizes_list.append(size_tensor.unsqueeze(0).repeat(matched_ids.shape[0], 1))
                calib_tensor = torch.as_tensor(t['calibs'][matched_ids], device=src_bevboxes.device, dtype=src_bevboxes.dtype)
                calib_list.append(calib_tensor)
            if img_sizes_list and calib_list:
                target_img_sizes = torch.cat(img_sizes_list, dim=0)
                target_calibs = torch.cat(calib_list, dim=0)
                rotated_loss = compute_bev_rotated_iou_loss(
                    src_bevboxes,
                    pred_angle_logits,
                    pred_boxes3d,
                    target_bevboxes,
                    target_heading_cls, 
                    target_heading_res,
                    target_boxes3d,
                    target_img_sizes,
                    target_calibs,
                    bev_range_x,
                    bev_range_z)
                losses['loss_bev_rgiou'] = rotated_loss / num_boxes

        return losses
    
    def loss_depths(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()
        z_pred, lv = src_depths[:, 0], src_depths[:, 1]
        matched_loss = 1.4142 * torch.exp(-lv) * (z_pred - target_depths).abs() + lv

        loss_depth_val = matched_loss.sum() / num_boxes

        if (self.use_in_radius_depth_nll
                and 'pred_boxes' in outputs
                and outputs['pred_depth'].dim() == 3):
            all_z  = outputs['pred_depth'][..., 0]
            all_lv = outputs['pred_depth'][..., 1]
            all_uv = outputs['pred_boxes'][..., 0:2]
            extra_terms = []
            for b, t in enumerate(targets):
                if 'boxes_3d' not in t or 'depth' not in t:
                    continue
                G = int(t['boxes_3d'].shape[0])
                if G == 0:
                    continue
                gt_uv = t['boxes_3d'][:, 0:2]
                gt_depth = t['depth'].squeeze(-1) if t['depth'].dim() > 1 else t['depth']
                d = (all_uv[b].unsqueeze(1) - gt_uv.unsqueeze(0)).abs().sum(-1)
                nearest = d.argmin(dim=1)
                min_d = d.gather(1, nearest[:, None]).squeeze(-1)
                in_r = min_d < self.in_radius_thresh
                if not bool(in_r.any()):
                    continue
                z_tgt = gt_depth[nearest]
                err_no_grad = (all_z[b].detach() - z_tgt).abs()
                nll = 1.4142 * torch.exp(-all_lv[b]) * err_no_grad + all_lv[b]
                extra_terms.append(nll[in_r])
            if extra_terms:
                extra = torch.cat(extra_terms)
                loss_depth_val = (loss_depth_val
                                  + self.in_radius_weight * extra.sum()
                                    / max(extra.numel(), 1))

        return {'loss_depth': loss_depth_val}
    
    def loss_dims(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)
        gt_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2] for t in targets], dim=0) * torch.tensor([80, 24], device=depth_map_logits.device)
        
        losses = dict()
        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth, gt_3dcenter)
        return losses

    def loss_region(self, outputs, targets, indices, num_boxes):
        region_probs = outputs['pred_region_prob']
        gt_region = torch.cat([t['obj_region'].unsqueeze(0) for t in targets], dim=0)

        loss = 0
        losses = dict()
        for region_prob in region_probs:
            gt_region_resized = F.interpolate(gt_region.unsqueeze(1).float(), size=region_prob.shape[2:], mode='bilinear', align_corners=True)
            # Compute intersection and union
            intersection = (region_prob * gt_region_resized).sum()
            total = region_prob.sum() + gt_region_resized.sum()
            # Compute Dice Coefficient
            dice_coef = (2. * intersection + 1) / (total + 1)
            # Compute Dice Loss
            dice_loss = 1 - dice_coef
            loss += dice_loss

        losses['loss_region'] = loss

        return losses
    
    def unproject_queries(self, boxes_2d_norm, depths, calib, img_sizes):
        B, Nq, _ = boxes_2d_norm.shape
        fx = calib[:, 0, 0].view(B, 1)
        fy = calib[:, 1, 1].view(B, 1)
        cx = calib[:, 0, 2].view(B, 1)
        cy = calib[:, 1, 2].view(B, 1)

        img_w = img_sizes[:, 0].view(B, 1)
        img_h = img_sizes[:, 1].view(B, 1)
        u = boxes_2d_norm[..., 0] * img_w
        v = boxes_2d_norm[..., 1] * img_h

        Z = depths.squeeze(-1)
        X = (u - cx) * Z / (fx + 1e-6)
        Y = (v - cy) * Z / (fy + 1e-6)
        return torch.stack([X, Y, Z], dim=-1)
    
    def loss_bev_map_ms(self, outputs, targets, indices, num_boxes):
        pred = outputs['pred_bev_confidence']

        device = pred.device
        gt_full = torch.stack([t['bev_gt_map'] for t in targets], dim=0).unsqueeze(1).to(device)
        total = 0.0
        bce = F.binary_cross_entropy_with_logits(pred, gt_full)

        p = torch.sigmoid(pred)
        inter = (p * gt_full).sum(dim=(1,2,3))
        union = p.sum(dim=(1,2,3)) + gt_full.sum(dim=(1,2,3))
        dice = 1 - (2 * inter + 1) / (union + 1)
        dice = dice.mean()
        total += (bce + dice) / 2.0
        return {'loss_bev_map': total}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'bev_map': self.loss_bev_map_ms,
            'bev_center': self.loss_bevcenter,
            'bev_boxes': self.loss_bev_boxes,
            'region': self.loss_region,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        group_num = self.group_num if self.training else 1
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        # Compute Det 2D loss
        for i, inter_outputs in enumerate(outputs['inter_outputs']):
            indices = self.matcher(inter_outputs, targets, group_num=group_num)
            for loss in self.inter_losses:
                l_dict = self.get_loss(loss, inter_outputs, targets, indices, num_boxes)
                l_dict = {k + f'_inter_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        # Compute Det 2D and 3D loss
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'inter_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in self.losses:
                    if loss == 'depth_map' or loss == 'region' or loss == 'depth_map_reg' or loss == 'bev_map':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    det2d_transformer = build_det2d_transformer(cfg)
    bevdet_transformer = build_bevdet_transformer(cfg)
    
    depth_predictor = DepthPredictor(cfg)
    ortho_predictor = MHIPM(cfg)

    model = MonoMIP(
        backbone = backbone,
        depth_predictor = depth_predictor,
        ortho_predictor = ortho_predictor,
        det2d_transformer = det2d_transformer,
        bevdet_transformer = bevdet_transformer,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        init_box=cfg['init_box'],
        group_num=cfg['group_num'],
        cfg=cfg)

    # matcher
    matcher = build_matcher(cfg)

    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    weight_dict['loss_bev_map'] = cfg['depth_map_loss_coef']
    weight_dict['loss_bev_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_bev_bbox'] = cfg['bbox_loss_coef']
    weight_dict['loss_bev_rgiou'] = cfg['giou_loss_coef']
    
    
    losses = ['labels', 'boxes', 'cardinality', 'dims', 'angles', 'depths', 'center', 'bev_center', 'bev_map', 'depth_map']
    
    inter_losses = ['labels', 'boxes', 'center', 'dims', 'depths']

    bev_losses = ['labels', 'dims', 'angles', 'depths', 'bev_center']

    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    aux_weight_dict = {}
    inter_keys = ['loss_ce', 'loss_bbox', 'loss_giou', 'loss_center', 'loss_dim', 'loss_depth']
    layers = cfg['dec_layers']
    inter_weight_dict = {}
    for i in range(layers):
        inter_weight_dict.update({k + f'_inter_{i}': v for k, v in weight_dict.items() if k in inter_keys})
    weight_dict.update(inter_weight_dict)


    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        inter_losses=inter_losses,
        bev_losses=bev_losses,
        group_num=cfg['group_num'],
        use_in_radius_depth_nll=cfg.get('use_in_radius_depth_nll', False),
        in_radius_thresh=cfg.get('in_radius_thresh', 0.10),
        in_radius_weight=cfg.get('in_radius_weight', 1.0),
    )

    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion