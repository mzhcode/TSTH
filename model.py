import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepSimLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model
# import ptwt
from DWT_IDWT_layer_level2 import DWT_3D, IDWT_3D
from DWT_IDWT_layer_level2 import DWT_2D, IDWT_2D

__all__ = ['Spikingformer']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x


class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)

        self.q_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)

        self.k_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x


class TM_mixing_time(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.act1 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act2 = MultiStepLIFNode(tau=1.75, v_threshold=0.55, detach_reset=True, backend='cupy')
        self.act3 = MultiStepLIFNode(tau=1.75, v_threshold=0.55, detach_reset=True, backend='cupy')
        self.act4 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act5 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act6 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act7 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act8 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act9 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act10 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act11 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act12 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.act13 = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.hidden_size = dim
        self.time_step = 8
        self.num_blocks = 8
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.complex_weight_ll = nn.Parameter(
            torch.randn(dim, self.time_step // 4, h // 4, w // 4, dtype=torch.float32) * 0.02)

        # channel_mix
        self.complex_weight_lh_1 = nn.Parameter(
            torch.randn(self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_12 = nn.Parameter(
            torch.randn(self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)

        self.complex_weight_lh_2 = nn.Parameter(
            torch.randn(self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_22 = nn.Parameter(
            torch.randn(self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)

        self.complex_weight_lh_b1 = nn.Parameter(
            torch.randn(self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b12 = nn.Parameter(
            torch.randn(self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

        self.complex_weight_lh_b2 = nn.Parameter(
            torch.randn(self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b22 = nn.Parameter(
            torch.randn(self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

        # time_sptial mix
        self.weight_lh_1 = nn.Parameter(
            torch.randn(7, h//2, w//2, self.time_step//2, dtype=torch.float32) * 0.02)
        self.weight_lh_12 = nn.Parameter(
            torch.randn(7, h//2, w//2, self.time_step//2, dtype=torch.float32) * 0.02)

        self.weight_lh_2 = nn.Parameter(
            torch.randn(7, h//4, w//4, self.time_step//4, dtype=torch.float32) * 0.02)
        self.weight_lh_22 = nn.Parameter(
            torch.randn(7, h//4, w//4, self.time_step//4, dtype=torch.float32) * 0.02)

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn12 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.bn22 = nn.BatchNorm2d(dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.bn31 = nn.BatchNorm2d(dim)
        self.bn4 = nn.BatchNorm2d(dim)
        self.bn41 = nn.BatchNorm2d(dim)

        # self.wavelet = Wavelets()
        self.transform = DWT_3D(wavename='haar')
        self.inverse = IDWT_3D(wavename='haar')
        self.softshrink = 0.0

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def multiply_time(self, input, weights):
        return torch.einsum('...tbd,tbdk->...tbk', input, weights)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.to(torch.float32)
        x = self.act1(x).permute(1, 2, 0, 3, 4)

        x_lll, high1, high2 = self.transform(x)
        # 3D Wavelet transform

        # B C T H W --> T B C H W --> B C T H W
        x_lll = self.act2(x_lll.permute(2, 0, 1, 3, 4)).permute(1, 2, 0, 3, 4).contiguous()
        x_lll = x_lll * self.complex_weight_ll
        x_lll = self.act3(x_lll.permute(2, 0, 1, 3, 4)).permute(1, 2, 0, 3, 4).contiguous()

        high1 = self.act4(high1.permute(3, 0, 1, 2, 4, 5)).permute(1, 3, 2, 4, 5, 0).contiguous()
        high2 = self.act5(high2.permute(3, 0, 1, 2, 4, 5)).permute(1, 3, 2, 4, 5, 0).contiguous()
        # B 7 C T H W --> T B 7 C H W --> B 7 T H W C --> B 7 T H W head C//head

        # token_mix
        # B C 7 H W T
        high1 = high1 * self.weight_lh_1
        high1 = high1.permute(5, 0, 2, 1, 3, 4).contiguous()
        # T B 7 C H W
        high1 = (high1.flatten(0, 2))
        high1 = self.bn3(high1)
        high1 = high1.view(T // 2, B, 7, C, H // 2, W // 2)
        high1 = self.act10(high1)
        high1 = high1.permute(1, 3, 2, 4, 5, 0).contiguous()
        # B C 7 H W T
        high1 = high1 * self.weight_lh_12
        high1 = high1.permute(5, 0, 2, 1, 3, 4).contiguous()
        high1 = (high1.flatten(0, 2))
        high1 = self.bn31(high1)
        high1 = high1.view(T // 2, B, 7, C, H // 2, W // 2)
        high1 = self.act11(high1)
        high1 = high1.permute(1, 2, 0, 4, 5, 3).contiguous()
        high1 = high1.reshape(high1.shape[0], high1.shape[1], high1.shape[2], high1.shape[3], high1.shape[4],
                              self.num_blocks, self.block_size)
        # channel_mix
        high1 = self.multiply(high1, self.complex_weight_lh_1) + self.complex_weight_lh_b1
        high1 = (high1.flatten(0, 2)).flatten(-2, -1)
        high1 = high1.permute(0, 3, 1, 2).contiguous()
        high1 = self.bn1(high1)
        high1 = high1.permute(0, 2, 3, 1).contiguous()
        high1 = high1.view(B, 7, T // 2, high1.shape[1], high1.shape[2], self.num_blocks, self.block_size)
        high1 = self.act6(high1.permute(2, 0, 1, 3, 4, 5, 6)).permute(1, 2, 0, 3, 4, 5, 6).contiguous()
        high1 = self.multiply(high1, self.complex_weight_lh_12) + self.complex_weight_lh_b12
        high1 = (high1.flatten(0, 2)).flatten(-2, -1)
        high1 = high1.permute(0, 3, 1, 2).contiguous()
        high1 = self.bn12(high1)
        high1 = high1.permute(0, 2, 3, 1).contiguous()
        high1 = high1.view(B, 7, T // 2, high1.shape[1], high1.shape[2], self.num_blocks, self.block_size)
        high1 = self.act7((high1.flatten(-2, -1)).permute(2, 0, 1, 3, 4, 5)).permute(2, 1, 5, 0, 3, 4).contiguous()
        # T B 7 H W C --> B 7 C H W T --> B C 7 H W T

        # token_mix
        high2 = high2 * self.weight_lh_2
        high2 = high2.permute(5, 0, 2, 1, 3, 4).contiguous()
        # T B 7 C H W
        high2 = (high2.flatten(0, 2))
        high2 = self.bn4(high2)
        high2 = high2.view(T // 4, B, 7, C, H // 4, W // 4)
        high2 = self.act12(high2)
        high2 = high2.permute(1, 3, 2, 4, 5, 0).contiguous()
        high2 = high2 * self.weight_lh_22
        high2 = high2.permute(5, 0, 2, 1, 3, 4).contiguous()
        high2 = (high2.flatten(0, 2))
        high2 = self.bn41(high2)
        high2 = high2.view(T // 4, B, 7, C, H // 4, W // 4)
        high2 = self.act13(high2)
        high2 = high2.permute(1, 2, 0, 4, 5, 3).contiguous()

        # channel_mix
        high2 = high2.reshape(high2.shape[0], high2.shape[1], high2.shape[2], high2.shape[3], high2.shape[4],
                              self.num_blocks, self.block_size)
        high2 = self.multiply(high2, self.complex_weight_lh_2) + self.complex_weight_lh_b2
        high2 = (high2.flatten(0, 2)).flatten(-2, -1)
        high2 = high2.permute(0, 3, 1, 2).contiguous()
        high2 = self.bn2(high2)
        high2 = high2.permute(0, 2, 3, 1).contiguous()
        high2 = high2.view(B, 7, T // 4, high2.shape[1], high2.shape[2], self.num_blocks, self.block_size)
        high2 = self.act8(high2.permute(2, 0, 1, 3, 4, 5, 6)).permute(1, 2, 0, 3, 4, 5, 6).contiguous()
        high2 = self.multiply(high2, self.complex_weight_lh_22) + self.complex_weight_lh_b22
        high2 = (high2.flatten(0, 2)).flatten(-2, -1)
        high2 = high2.permute(0, 3, 1, 2).contiguous()
        high2 = self.bn22(high2)
        high2 = high2.permute(0, 2, 3, 1).contiguous()
        high2 = high2.view(B, 7, T // 4, high2.shape[1], high2.shape[2], self.num_blocks, self.block_size)
        high2 = self.act9((high2.flatten(-2, -1)).permute(2, 0, 1, 3, 4, 5)).permute(2, 1, 5, 0, 3, 4).contiguous()

        x = self.inverse(x_lll, high1, high2)
        x = x.reshape(T, B, C, H, W)
        return x


class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=512):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)

        self.proj1_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')
        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.proj1_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj2_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')
        self.proj2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj2_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj3_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')
        self.proj3_conv = nn.Conv2d(embed_dims // 2, 384, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(384)
        self.proj3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj4_lif = MultiStepLIFNode(tau=1.75, detach_reset=True, backend='cupy')
        self.proj4_conv = nn.Conv2d(384, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.dt1 = TM_mixing_time(128, 88, 88)
        self.mlp1 = MLP(in_features=128, hidden_features=4*128, drop=0)
        self.dt2 = TM_mixing_time(256, 44, 44)
        self.mlp2 = MLP(in_features=256, hidden_features=4 * 256, drop=0)
        self.dt3 = SpikingSelfAttention(384, 16)
        self.mlp3 = MLP(in_features=384, hidden_features=4 * 384, drop=0)
        self.dt4 = SpikingSelfAttention(512, 16)
        self.mlp4 = MLP(in_features=512, hidden_features=4 * 512, drop=0)

    def forward(self, x):
        T, B, C, H, W = x.shape

        # to spike
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj1_lif(x)

        # stage 1
        x = self.proj1_conv(x.flatten(0, 1))
        x = self.proj1_bn(x)
        x = self.proj1_mp(x).reshape(T, B, -1, int(H / 2), int(W / 2))
        x = x + self.dt1(x)
        x = x + self.mlp1(x)
        x = self.proj2_lif(x)

        # stage 2
        x = self.proj2_conv(x.flatten(0, 1))
        x = self.proj2_bn(x)
        x = self.proj2_mp(x).reshape(T, B, -1, int(H / 4), int(W / 4))
        x = x + self.dt2(x)
        x = x + self.mlp2(x)
        x = self.proj3_lif(x)

        # stage 3
        x = self.proj3_conv(x.flatten(0, 1))
        x = self.proj3_bn(x)
        x = self.proj3_mp(x).reshape(T, B, -1, int(H / 8), int(W / 8))
        x = x + self.dt3(x)
        x = x + self.mlp3(x)
        x = self.proj4_lif(x)

        # stage 4
        x = self.proj4_conv(x.flatten(0, 1))
        x = self.proj4_bn(x)
        x = self.proj4_mp(x).reshape(T, B, -1, int(H / 16), int(W / 16))
        x = x + self.dt4(x)
        x = x + self.mlp4(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)



class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=101,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 ):
        super().__init__()
        self.num_classes = 101
        self.depths = depths
        self.T = T
        hash_length = 64
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims)
        num_patches = patch_embed.num_patches

        setattr(self, f"patch_embed", patch_embed)

        # classification head
        self.head = nn.Linear(hash_length, self.num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        self.hash_layer = nn.Linear(embed_dims, hash_length)
        self.hash_lif = MultiStepSimLIFNode(tau=2.0, v_threshold=0.2, detach_reset=True, backend='cupy')
        # self.hash_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.2, detach_reset=True, backend='cupy')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def multiply(self, input, weights):
        return torch.einsum('...ijk,...ijkd->...jd', input, weights)

    def forward_features(self, x):
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        hash_out1 = self.hash_layer(x)
        hash_out, sim_x = self.hash_lif(hash_out1)
        cls_out = self.head(hash_out.mean(0))
        hash_out = hash_out.prod(dim=0)
        mean_x = torch.mean(sim_x, dim=0)
        return mean_x, hash_out, cls_out


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32).cuda()
    model = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).cuda()


    model.eval()
    y = model(input)
    print(y.shape)
    print('Test Good!')











