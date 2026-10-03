import math
from typing import Tuple

import torch

from mmcv.ops import box_iou_rotated


def decode_predicted_yaw(angle_logits: torch.Tensor,
                         pred_boxes3d: torch.Tensor,
                         img_sizes: torch.Tensor,
                         calibs: torch.Tensor) -> torch.Tensor:
    if angle_logits.numel() == 0:
        return angle_logits.new_zeros((0,))

    num_bins = angle_logits.shape[-1] // 2
    cls_logits = angle_logits[:, :num_bins]
    res_logits = angle_logits[:, num_bins:]

    probs = torch.softmax(cls_logits, dim=-1)
    angle_per_class = 2 * math.pi / num_bins
    bin_centers = torch.arange(
        num_bins, device=angle_logits.device, dtype=angle_logits.dtype) * angle_per_class

    per_bin_angles = bin_centers.unsqueeze(0) + res_logits
    sin_comp = torch.sum(probs * torch.sin(per_bin_angles), dim=-1)
    cos_comp = torch.sum(probs * torch.cos(per_bin_angles), dim=-1)
    alpha = torch.atan2(sin_comp, cos_comp)

    img_w = img_sizes[:, 0]
    u_norm = pred_boxes3d[:, 0]
    u_px = u_norm * img_w
    cx = calibs[:, 0, 2]
    fx = calibs[:, 0, 0]

    yaw = alpha + torch.atan2(u_px - cx, fx)
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return yaw


def decode_target_yaw(target_cls: torch.Tensor,
                      target_res: torch.Tensor,
                      target_boxes3d: torch.Tensor,
                      target_img_sizes: torch.Tensor,
                      target_calibs: torch.Tensor,
                      num_bins: int = 12) -> torch.Tensor:
    if target_cls.numel() == 0:
        return target_cls.new_zeros((0,), dtype=target_res.dtype)

    angle_per_class = 2 * math.pi / num_bins
    cls_float = target_cls.view(-1).float()
    alpha = cls_float * angle_per_class + target_res.view(-1)
    img_w = target_img_sizes[:, 0]
    u_norm = target_boxes3d[:, 0]
    u_px = u_norm * img_w
    cx = target_calibs[:, 0, 2]
    fx = target_calibs[:, 0, 0]

    yaw = alpha + torch.atan2(u_px - cx, fx)
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return yaw


def _rotated_box_to_polygon(cx, cz, w, l, yaw):
    half_w = w / 2
    half_l = l / 2
    corners = torch.stack([
        torch.stack([-half_w, -half_l]),
        torch.stack([-half_w, half_l]),
        torch.stack([half_w, half_l]),
        torch.stack([half_w, -half_l]),
    ], dim=0)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rot_x = corners[:, 0] * cos_yaw - corners[:, 1] * sin_yaw + cx
    rot_z = corners[:, 0] * sin_yaw + corners[:, 1] * cos_yaw + cz
    return torch.stack([rot_x, rot_z], dim=-1)


def _polygon_area(poly: torch.Tensor) -> torch.Tensor:
    if poly.shape[0] < 3:
        return poly.new_tensor(0.0)
    x = poly[:, 0]
    y = poly[:, 1]
    area = torch.sum(x * torch.roll(y, shifts=-1, dims=0) -
                     y * torch.roll(x, shifts=-1, dims=0))
    return 0.5 * torch.abs(area)


def _clip_polygon(subject: torch.Tensor, clipper: torch.Tensor) -> torch.Tensor:
    def inside(point, a, b):
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        return (cross <= 0).item()

    def intersection(p1, p2, a, b):
        r = p2 - p1
        s = b - a
        denom = r[0] * s[1] - r[1] * s[0]
        eps = denom.new_tensor(1e-6)
        denom = torch.where(torch.abs(denom) < 1e-6, eps, denom)
        t = ((a[0] - p1[0]) * s[1] - (a[1] - p1[1]) * s[0]) / denom
        return p1 + t * r

    output = list(subject.unbind(0))
    for edge_idx in range(clipper.shape[0]):
        if not output:
            break
        input_list = output
        output = []
        a = clipper[edge_idx]
        b = clipper[(edge_idx + 1) % clipper.shape[0]]
        prev_point = input_list[-1]
        for curr_point in input_list:
            curr_inside = inside(curr_point, a, b)
            prev_inside = inside(prev_point, a, b)
            if curr_inside:
                if not prev_inside:
                    output.append(intersection(prev_point, curr_point, a, b))
                output.append(curr_point)
            elif prev_inside:
                output.append(intersection(prev_point, curr_point, a, b))
            prev_point = curr_point

    if not output:
        return subject.new_zeros((0, 2))
    return torch.stack(output, dim=0)


def _single_rotated_iou(box_a: Tuple[torch.Tensor, ...], box_b: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    polygon_a = _rotated_box_to_polygon(*box_a)
    polygon_b = _rotated_box_to_polygon(*box_b)
    inter_poly = _clip_polygon(polygon_a, polygon_b)
    inter_area = _polygon_area(inter_poly)
    area_a = _polygon_area(polygon_a)
    area_b = _polygon_area(polygon_b)
    union = area_a + area_b - inter_area
    return inter_area / (union + 1e-6)


def _denormalize_bev_boxes(boxes: torch.Tensor,
                           bev_range_x: Tuple[float, float],
                           bev_range_z: Tuple[float, float]) -> torch.Tensor:
    x_extent = bev_range_x[1] - bev_range_x[0]
    z_extent = bev_range_z[1] - bev_range_z[0]
    cx = boxes[:, 0] * x_extent + bev_range_x[0]
    cz = boxes[:, 1] * z_extent + bev_range_z[0]
    w = boxes[:, 2] * x_extent
    l = boxes[:, 3] * z_extent
    return torch.stack([cx, cz, w, l], dim=-1)


def compute_bev_rotated_iou_loss(pred_boxes: torch.Tensor,
                                 pred_angle_logits: torch.Tensor,
                                 pred_boxes3d: torch.Tensor,
                                 target_boxes: torch.Tensor,
                                 target_heading_cls: torch.Tensor,
                                 target_heading_res: torch.Tensor,
                                 target_boxes3d: torch.Tensor,
                                 target_img_sizes: torch.Tensor,
                                 target_calibs: torch.Tensor,
                                 bev_range_x: Tuple[float, float] = (-30.0, 30.0),
                                 bev_range_z: Tuple[float, float] = (1e-3, 60.0)) -> torch.Tensor:
    if pred_boxes.numel() == 0:
        return pred_boxes.new_tensor(0.0)
    pred_metric = _denormalize_bev_boxes(pred_boxes, bev_range_x, bev_range_z)
    target_metric = _denormalize_bev_boxes(target_boxes, bev_range_x, bev_range_z)
    pred_yaw = decode_predicted_yaw(
        pred_angle_logits,
        pred_boxes3d,
        target_img_sizes,
        target_calibs,
    )
    target_yaw = decode_target_yaw(target_heading_cls, target_heading_res,
                                   target_boxes3d, target_img_sizes, target_calibs)
    widths = torch.clamp(pred_metric[:, 2], min=1e-3)
    lengths = torch.clamp(pred_metric[:, 3], min=1e-3)
    tgt_widths = torch.clamp(target_metric[:, 2], min=1e-3)
    tgt_lengths = torch.clamp(target_metric[:, 3], min=1e-3)

    pred_boxes_rot = torch.stack(
        [pred_metric[:, 0], pred_metric[:, 1], widths, lengths, torch.rad2deg(pred_yaw)], dim=-1)
    pred_boxes_rot_flip = torch.stack(
        [pred_metric[:, 0], pred_metric[:, 1], widths, lengths, torch.rad2deg(pred_yaw + math.pi)], dim=-1)
    target_boxes_rot = torch.stack(
        [target_metric[:, 0], target_metric[:, 1], tgt_widths, tgt_lengths, torch.rad2deg(target_yaw)], dim=-1)
    ious = box_iou_rotated(pred_boxes_rot.contiguous(), target_boxes_rot.contiguous())
    ious_flip = box_iou_rotated(pred_boxes_rot_flip.contiguous(), target_boxes_rot.contiguous())
    diag = torch.diag(ious)
    diag_flip = torch.diag(ious_flip)
    best = torch.max(diag, diag_flip)
    return (1.0 - best).sum()
