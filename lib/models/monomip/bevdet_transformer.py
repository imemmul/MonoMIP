from mimetypes import init
from typing import Optional, List
import math
import copy
from weakref import ref
import torch
import torch.nn.functional as F
from torch import inverse, nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from utils.misc import inverse_sigmoid, NestedTensor
from .ops.modules import MSDeformAttn, MSDeformAttn_cross, MultiheadAttention, BevMSDeformAttn
from .position_encoding import PositionEmbeddingSine


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

def _get_sine_pos_embed(pos_tensor, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos = pos_tensor[..., None] * scale / dim_t
    pos = torch.stack([pos[..., 0::2].sin(), pos[..., 1::2].cos()], dim=-1).flatten(-2)
    pos = pos.flatten(-2)
    return pos


class BEVDetTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
            group_num=1,
            num_queries=50,
            bev_encoder_blocks=0):

        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.group_num = group_num
        self.num_queries = num_queries

        decoder_layer = TransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points, group_num=group_num)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.bev_pos_embed = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        self.ref_point_proj = MLP(d_model, d_model, d_model, 2)

        self.query_2d_to_bev = MLP(d_model * 2, d_model, d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        B = mask.shape[0]
        device = mask.device
        return torch.ones(B, 2, device=device, dtype=torch.float32)

    def unproject_queries(self, boxes_2d_norm, depths, calib, img_sizes):
        B, Nq, _ = boxes_2d_norm.shape
        fx = calib[:, 0, 0].view(B, 1)
        fy = calib[:, 1, 1].view(B, 1)
        cx = calib[:, 0, 2].view(B, 1)
        cy = calib[:, 1, 2].view(B, 1)
        tx = calib[:, 0, 3].view(B, 1) / (-fx)
        ty = calib[:, 1, 3].view(B, 1) / (-fy)
        img_w = img_sizes[:, 0].view(B, 1)
        img_h = img_sizes[:, 1].view(B, 1)
        u = boxes_2d_norm[..., 0] * img_w
        v = boxes_2d_norm[..., 1] * img_h

        Z = depths.squeeze(-1)
        X = (u - cx) * Z / fx + tx
        Y = (v - cy) * Z / fy + ty
        return torch.stack([X, Y, Z], dim=-1)


    def forward(self, srcs, masks, query_embed=None, reference_points_2d=None, inter_dim_to_bev=None, init_depth=None, img_sizes=None, calibs=None, init_logvar=None):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            pos_embed = self.bev_pos_embed(NestedTensor(src, mask))
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            mask = mask.flatten(1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        src_flatten = src_flatten + lvl_pos_embed_flatten
        inter_dim_to_bev = inter_dim_to_bev
        dim_h, dim_w, dim_l = inter_dim_to_bev[..., 0], inter_dim_to_bev[..., 1], inter_dim_to_bev[..., 2]
        init_coord = reference_points_2d
        points_3d_cam = self.unproject_queries(
            boxes_2d_norm=init_coord[..., :2],
            depths=init_depth,
            calib=calibs,
            img_sizes=img_sizes
        )

        coords_x_metric = points_3d_cam[..., 0]
        coords_z_metric = points_3d_cam[..., 2]
        bev_range_x = (-30.0, 30.0)
        bev_range_z = (1e-3, 60.0)
        ref_u_bev = (coords_x_metric - bev_range_x[0]) / (bev_range_x[1] - bev_range_x[0])
        ref_v_bev = (coords_z_metric - bev_range_z[0]) / (bev_range_z[1] - bev_range_z[0])
        ref_w_bev = dim_w / (bev_range_x[1] - bev_range_x[0])
        ref_l_bev = dim_l / (bev_range_z[1] - bev_range_z[0])
        reference_points = torch.stack([ref_u_bev, ref_v_bev, ref_w_bev, ref_l_bev], dim=-1).clamp(min=0.0, max=1.0)

        ref_sine = _get_sine_pos_embed(reference_points, num_pos_feats=self.d_model // 4)
        query_pos = self.ref_point_proj(ref_sine)

        tgt = self.query_2d_to_bev(
            torch.cat([query_embed.detach(), query_pos], dim=-1)
        )

        init_reference_out = reference_points
        init_logvar_out = init_logvar
        hs, inter_references, inter_logvars = self.decoder(
            tgt,
            reference_points,
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_pos,
            mask_flatten,
            bs=bs,
            init_logvar=init_logvar)

        inter_references_out = inter_references
        reference_points = inter_references[-1]
        intermediate_output = {
            'hs': hs,
            'init_reference_out': init_reference_out,
            'inter_references_out': inter_references_out,
            'init_logvar_out': init_logvar_out,
            'inter_logvars_out': inter_logvars,
            'reference_points': reference_points,
        }

        return intermediate_output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, group_num=1):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.group_num = group_num

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
                bs):

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)
        num_queries = q.shape[0]

        if self.training:
            num_noise = num_queries-self.group_num * 50
            num_queries = self.group_num * 50
            q_noise = q[:num_noise].repeat(1,self.group_num, 1)
            k_noise = k[:num_noise].repeat(1,self.group_num, 1)
            v_noise = v[:num_noise].repeat(1,self.group_num, 1)
            q = q[num_noise:]
            k = k[num_noise:]
            v = v[num_noise:]
            q = torch.cat(q.split(num_queries // self.group_num, dim=0), dim=1)
            k = torch.cat(k.split(num_queries // self.group_num, dim=0), dim=1)
            v = torch.cat(v.split(num_queries // self.group_num, dim=0), dim=1)
            q = torch.cat([q_noise,q], dim=0)
            k = torch.cat([k_noise,k], dim=0)
            v = torch.cat([v_noise,v], dim=0)

        tgt2 = self.self_attn(q, k, v)[0]
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=1), dim=0).transpose(0, 1)

        else:
            tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)


        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.d_model = decoder_layer.linear1.in_features
        self.bev_embed = None
        self.ref_point_proj = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, bs=None, init_logvar=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        intermediate_logvars = []
        bs = src.shape[0]
        logvar = init_logvar
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:, None]
            elif reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output,
                           query_pos,
                           reference_points_input,
                           src,
                           src_spatial_shapes,
                           src_level_start_index,
                           src_padding_mask,
                           bs)
            if self.bev_embed is not None:
                tmp = self.bev_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    reference_points = inverse_sigmoid(reference_points)
                    reference_points += tmp[..., :4]
                    new_reference_points = reference_points.sigmoid()
                reference_points = new_reference_points.detach()

                if logvar is not None:
                    delta_logvar = torch.tanh(tmp[..., 4:5])
                    logvar = (logvar + delta_logvar).detach()

                if self.ref_point_proj is not None:
                    ref_sine = _get_sine_pos_embed(reference_points, num_pos_feats=self.d_model // 4)
                    query_pos = self.ref_point_proj(ref_sine)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                if logvar is not None:
                    intermediate_logvars.append(logvar)

        inter_logvars = torch.stack(intermediate_logvars) if intermediate_logvars else None
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), inter_logvars
        return output, reference_points, inter_logvars


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_bevdet_transformer(cfg):
    return BEVDetTransformer(
        d_model=cfg['hidden_dim'],
        dropout=cfg['dropout'],
        activation="relu",
        nhead=cfg['nheads'],
        dim_feedforward=cfg['dim_feedforward'],
        num_encoder_layers=cfg['enc_layers'],
        num_decoder_layers=cfg['dec_layers'],
        return_intermediate_dec=cfg['return_intermediate_dec'],
        num_feature_levels=cfg['bev_num_feature_levels'],
        dec_n_points=cfg['dec_n_points'],
        enc_n_points=cfg['enc_n_points'],
        group_num=cfg['group_num'],
        num_queries=cfg['num_queries'],
        bev_encoder_blocks=cfg.get('bev_encoder_blocks', 0))
