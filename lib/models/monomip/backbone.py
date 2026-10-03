# ------------------------------------------------------------------------
# ResNet-vd (PResNet) with SSLD ImageNet weights from RT-DETR
# (https://github.com/lyuwenyu/RT-DETR), as ported in MonoCoP
# (https://github.com/alanzhangcs/MonoCoP). Joiner from Deformable DETR.
# ------------------------------------------------------------------------
from collections import OrderedDict
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import NestedTensor
from .position_encoding import build_position_encoding


def get_activation(act: str, inpace: bool = True):
    act = act.lower()
    if act == "silu":
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError("")
    if hasattr(m, "inplace"):
        m.inplace = inpace
    return m


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride,
                              padding=(kernel_size - 1) // 2 if padding is None else padding,
                              bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


ResNet_cfg = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}
download_url = {
    18: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth",
    34: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth",
    50: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth",
    101: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth",
}


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="b"):
        super().__init__()
        if variant == "a":
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride
        width = ch_out
        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ("conv", ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1)),
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act="relu", variant="b"):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(block(
                ch_in, ch_out,
                stride=2 if i == 0 and stage_num != 2 else 1,
                shortcut=False if i == 0 else True,
                variant=variant, act=act))
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class PResNet(nn.Module):
    def __init__(self, depth=50, variant="d", num_stages=4, return_idx=(1, 2, 3),
                 act="relu", freeze_at=0, freeze_norm=True, pretrained=True,
                 weights_path=None):
        super().__init__()
        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ("c", "d"):
            conv_def = [[3, ch_in // 2, 3, 2, "conv1_1"],
                        [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                        [ch_in // 2, ch_in, 3, 1, "conv1_3"]]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]
        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act))
            for c_in, c_out, k, s, _name in conv_def]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck
        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]
        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            self.res_layers.append(Blocks(block, ch_in, ch_out_list[i],
                                          block_nums[i], i + 2, act=act, variant=variant))
            ch_in = _out_channels[i]

        self.return_idx = list(return_idx)
        self.out_channels = [_out_channels[i] for i in self.return_idx]
        self.out_strides = [_out_strides[i] for i in self.return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])
        if freeze_norm:
            self._freeze_norm(self)

        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state)
        elif pretrained:
            state = torch.hub.load_state_dict_from_url(download_url[depth])
            self.load_state_dict(state)

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


_DEPTH = {"resnet18": 18, "resnet34": 34, "resnet50": 50, "resnet101": 101}


class BackboneVD(nn.Module):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, cfg):
        super().__init__()
        assert name in _DEPTH, f"backbone_vd supports {list(_DEPTH)}, got {name}"
        assert return_interm_layers, "backbone_vd assumes num_feature_levels > 1"
        self.body = PResNet(
            depth=_DEPTH[name],
            variant="d",
            return_idx=(1, 2, 3),
            freeze_at=0,
            freeze_norm=True,
            pretrained=cfg.get("backbone_pretrained", True),
            weights_path=cfg.get("backbone_weights_path", None),
        )
        self.strides = list(self.body.out_strides)
        self.num_channels = list(self.body.out_channels)
        if not train_backbone:
            for p in self.body.parameters():
                p.requires_grad_(False)

    def forward(self, images):
        feats = self.body(images)
        out = {}
        for i, x in enumerate(feats):
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3],
                            dtype=torch.bool, device=x.device)
            out[str(i)] = NestedTensor(x, m)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, images):
        xs = self[0](images)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    return_interm_layers = cfg["masks"] or cfg["num_feature_levels"] > 1
    backbone = BackboneVD(cfg["backbone"], cfg["train_backbone"], return_interm_layers, cfg)
    return Joiner(backbone, position_embedding)
