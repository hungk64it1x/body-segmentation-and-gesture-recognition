import torch
import math
from torch import Tensor
from torch.nn import functional as F
import torch
from typing import Tuple
from torch import nn, Tensor
import warnings
warnings.filterwarnings('ignore')

class PPM(nn.Module):
    """Pyramid Pooling Module in PSPNet
    """
    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(c1, c2, 1)
            )
        for scale in scales])

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))

        outs = [x] + outs[::-1]
        out = self.bottleneck(torch.cat(outs, dim=1))
        return out

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dwconv = DWConv(hidden_dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, mlp_ratio=4, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=64, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, patch_size//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


pvtv2_settings = {
    'B1': [2, 2, 2, 2],    # depths
    'B2': [3, 4, 6, 3],
    'B3': [3, 4, 18, 3],
    'B4': [3, 8, 27, 3],
    'B5': [3, 6, 40, 3]
}


class PVTv2(nn.Module):     
    def __init__(self, model_name: str = 'B1') -> None:
        super().__init__()
        assert model_name in pvtv2_settings.keys(), f"PVTv2 model name should be in {list(pvtv2_settings.keys())}"
        depths = pvtv2_settings[model_name]
        embed_dims = [64, 128, 320, 512]
        drop_path_rate = 0.1
        self.channels = embed_dims
        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # transformer encoder
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, 8, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, 4, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, 4, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x1, x2, x3, x4

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "\
                      "The distribution of values may be incorrect.",\
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d


class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """
    def __init__(self, inplanes, planes, use_scale=False, groups=8):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNL block uses 'SCALE'", 'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNL block uses '{}' groups".format(self.groups), \
        'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = c // self.groups

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []

            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class GALDBlock(nn.Module):
    def __init__(self, inplane, plane):
        super(GALDBlock, self).__init__()
        """
            Note down the spatial into 1/16
        """
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane,kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.long_relation = SpatialCGNL(inplane, plane)
        self.local_attention = LocalAttenModule(inplane)

    def forward(self, x):
        size = x.size()[2:]
        x = self.down(x)
        x = self.long_relation(x)
        # local attention
        x = F.interpolate(x,size=size, mode="bilinear", align_corners=True)
        res = x
        x = self.local_attention(x)
        return x + res


class LocalAttenModule(nn.Module):
    def __init__(self, inplane):
        super(LocalAttenModule, self).__init__()
        self.dconv1 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        res1 = x
        res2 = x
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        x_mask = self.sigmoid_spatial(x)

        res1 = res1 * x_mask

        return res2 + res1

class BaseModel(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__()
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
            
class UPerHead(nn.Module):

    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output
    
class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class CustomModel(BaseModel):
    def __init__(self, backbone: str='PVTv2-B3', decode: str='UPerHead', num_classes: int=1, pretrained=None) -> None:
        super().__init__(backbone, num_classes)
        self.gald_block = GALDBlock(64, 8)
        self.decode_head = eval(str(decode))([128, 320, 512], 128, num_classes)
        self.conv1 = ConvModule(64, 128, 1)
        self.conv2 = ConvModule(64, 320, 1)
        self.conv3 = ConvModule(64, 512, 1)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        x1 = y[0]  #  64x88x88
        x2 = y[1]  # 128x44x44
        x3 = y[2]  # 320x22x22
        x4 = y[3]  # 512x11x11
        # LD
        gce = self.gald_block(x1)
        gce1 = F.interpolate(gce, scale_factor=0.5, mode='bilinear')
        gce2 = F.interpolate(gce, scale_factor=0.25, mode='bilinear')
        gce3 = F.interpolate(gce, scale_factor=0.125, mode='bilinear')
        gce1 = self.conv1(gce1)
        gce2 = self.conv2(gce2)
        gce3 = self.conv3(gce3)
        
        x2 = x2 + gce1
        x3 = x3 + gce2
        x4 = x4 + gce3
        y = self.decode_head([x2, x3, x4])   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


if __name__ == '__main__':
    model = CustomModel()
    x = torch.zeros(2, 3, 352, 352)
    y = model(x)
    print(y.shape)
        

