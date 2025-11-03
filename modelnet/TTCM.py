import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
from modelnet.DBM import DMB


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def conv2d(x, device):
    b, c, h, w = x.shape
    conv = nn.Conv2d(c, c // 2, kernel_size=1).to(device)
    return conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z * res + x


class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat*2, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            ChannelAttention(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class FeatureWiseAffine(nn.Module):

    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms, pan):
        b, c, h, w = ms.shape

        kv = self.kv_dwconv(self.kv(pan))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(ms))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1 = LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.cro = CrossAttention(dim, num_heads, bias)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, ms, pan=None):
        if pan == None:
            ms = ms + self.cro(self.norm_cro1(ms), self.norm_cro2(ms))
        else:
            ms = ms + self.cro(self.norm_cro1(ms), self.norm_cro2(pan))
        ms = ms + self.ffn(self.norm2(ms))
        return ms


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type == 'BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, basefilter) -> None:
        super().__init__()
        self.nc = basefilter

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, stride=4, in_chans=36, embed_dim=32 * 32 * 32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim, 'BiasFree')

    def forward(self, x):
        # （b,c,h,w)->(b,c*s*p,h//s,w//s)
        # (b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x


class ChannelInteract_MambaBlock(nn.Module):
    def __init__(self, dim):
        super(ChannelInteract_MambaBlock, self).__init__()
        self.encoder = Mamba(dim, bimamba_type=None)
        self.norm = LayerNorm(dim, 'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)

    def forward(self, ipt):
        x, residual = ipt
        residual = x + residual

        x = self.norm(residual)
        return (self.encoder(x), residual)


class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = Mamba(dim, bimamba_type=None)
        self.panencoder = Mamba(dim, bimamba_type=None)
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')

    def forward(self, ms, pan, ms_residual, pan_residual):
        # ms (B,N,C)
        # pan (B,N,C)
        ms_residual = ms + ms_residual
        pan_residual = pan + pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B, N, C = ms.shape
        ms_first_half = ms[:, :, :C // 2]
        pan_first_half = pan[:, :, :C // 2]
        ms_swap = torch.cat([pan_first_half, ms[:, :, C // 2:]], dim=2)
        pan_swap = torch.cat([ms_first_half, pan[:, :, C // 2:]], dim=2)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap, pan_swap, ms_residual, pan_residual


class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim, bimamba_type="v3")
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, ms, ms_resi, pan):
        ms_resi = ms + ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms), extra_emb1=self.norm2(pan))
        B, HW, C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, int(HW ** 0.5), int(HW ** 0.5))
        ms = (self.dwconv(ms) + ms).flatten(2).transpose(1, 2)
        return ms, ms_resi


class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x + resi

class Gate(nn.Module):
    def __init__(self, ch=128):
        super(Gate, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(ch,ch, kernel_size=3, padding=1, stride=1, groups=ch)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(ch,ch, kernel_size=3, padding=1, stride=1, groups=ch)
        )
    
    def forward(self, x, y):
        return F.gelu(self.conv1(x))*self.conv2(y)



class Net(nn.Module):
    def __init__(self, num_channels=None, device=None, base_filter=None, args=None):
        super(Net, self).__init__()
        self.kernel = DMB(device=device).to(device)
        for param in self.kernel.parameters():
            param.requires_grad = False
        base_filter = 128
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            Swish(),
            nn.Linear(num_channels * 4, num_channels)
        )
        self.noise_emb = ResnetBlock(num_channels, num_channels, num_channels, norm_groups=num_channels)
        self.base_filter = base_filter
        self.stride = 1
        self.patch_size = 1
        self.pan_encoder = nn.Sequential(nn.Conv2d(12, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),
                                         HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(num_channels, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),
                                        HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.noisy_encoder = nn.Sequential(nn.Conv2d(num_channels, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),
                                           HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))

        self.cross_ms_pan1 = TransformerBlock(128, 4, ffn_expansion_factor=2, bias=False, LayerNorm_type='with_bias')
        self.cross_ms_pan2 = TransformerBlock(128, 4, ffn_expansion_factor=2, bias=False, LayerNorm_type='with_bias')
        self.cross_noise_pan1 = TransformerBlock(128, 4, ffn_expansion_factor=2, bias=False, LayerNorm_type='with_bias')
        self.cross_noise_pan2 = TransformerBlock(128, 4, ffn_expansion_factor=2, bias=False, LayerNorm_type='with_bias')
        self.gate1 = Gate(ch=128)
        self.gate2 = Gate(ch=128)
        self.gate3 = Gate(ch=128)
        self.gate4 = Gate(ch=128)

        self.embed_dim = base_filter * self.stride * self.patch_size
        self.shallow_fusion1 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        self.shallow_fusion2 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        self.ms_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size,
                                      stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size,
                                       stride=self.stride)
        self.noise_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size,
                                         stride=self.stride)
        self.deep_fusion1 = CrossMamba(self.embed_dim)
        self.deep_fusion2 = CrossMamba(self.embed_dim)
        self.deep_fusion3 = CrossMamba(self.embed_dim)
        self.deep_fusion4 = CrossMamba(self.embed_dim)
        # self.deep_fusion5 = CrossMamba(self.embed_dim)
        self.noisy_feature_extraction1 = ChannelInteract_MambaBlock(self.embed_dim)
        self.noisy_feature_extraction2 = ChannelInteract_MambaBlock(self.embed_dim)
        self.noisy_feature_extraction3 = ChannelInteract_MambaBlock(self.embed_dim)
        self.noisy_feature_extraction4 = ChannelInteract_MambaBlock(self.embed_dim)

        self.pan_feature_extraction1 = ChannelInteract_MambaBlock(self.embed_dim)
        self.pan_feature_extraction2 = ChannelInteract_MambaBlock(self.embed_dim)
        self.pan_feature_extraction3 = ChannelInteract_MambaBlock(self.embed_dim)
        self.pan_feature_extraction4 = ChannelInteract_MambaBlock(self.embed_dim)

        self.ms_feature_extraction1 = ChannelInteract_MambaBlock(self.embed_dim)
        self.ms_feature_extraction2 = ChannelInteract_MambaBlock(self.embed_dim)
        self.ms_feature_extraction3 = ChannelInteract_MambaBlock(self.embed_dim)
        self.ms_feature_extraction4 = ChannelInteract_MambaBlock(self.embed_dim)

        self.patchunembe = PatchUnEmbed(base_filter)
        self.output = Refine(base_filter, num_channels)

    def matching(self, hs, ms):
        return self.kernel(hs, ms)

    def forward(self, ms, pan, x_noisy, noise_level):
        pan = self.matching(ms, pan)

        t = self.noise_level_mlp(noise_level)
        if x_noisy.shape[1] == ms.shape[1] * 2:
            x_noisy = conv2d(x_noisy, x_noisy.device)

        noise_f = self.noisy_encoder(self.noise_emb(x_noisy, t))
        ms_f = self.ms_encoder(ms)

        b, c, h, w = ms_f.shape
        pan_f = self.pan_encoder(pan)

        del x_noisy, pan

        residual_ms_f = 0
        residual_pan_f = 0
        residual_noise_f = 0

        noise_f = self.noise_to_token(noise_f)
        noise_f, residual_noise_f = self.noisy_feature_extraction1([noise_f, residual_noise_f])
        noise_f, residual_noise_f = self.noisy_feature_extraction2([noise_f, residual_noise_f])
        noise_f, residual_noise_f = self.noisy_feature_extraction3([noise_f, residual_noise_f])
        noise_f, residual_noise_f = self.noisy_feature_extraction4([noise_f, residual_noise_f])
        noise_f = self.patchunembe(noise_f, (h, w))

        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)

        ms_f, residual_ms_f = self.ms_feature_extraction1([ms_f, residual_ms_f])
        pan_f, residual_pan_f = self.pan_feature_extraction1([pan_f, residual_pan_f])
        ms_f, residual_ms_f = self.ms_feature_extraction2([ms_f, residual_ms_f])
        pan_f, residual_pan_f = self.pan_feature_extraction2([pan_f, residual_pan_f])
        ms_f, residual_ms_f = self.ms_feature_extraction3([ms_f, residual_ms_f])
        pan_f, residual_pan_f = self.pan_feature_extraction3([pan_f, residual_pan_f])
        ms_f, residual_ms_f = self.ms_feature_extraction4([ms_f, residual_ms_f])
        pan_f, residual_pan_f = self.pan_feature_extraction4([pan_f, residual_pan_f])
        ms_f = self.patchunembe(ms_f, (h, w))
        pan_f = self.patchunembe(pan_f, (h, w))
        
        ms_ori = ms_f
        ms_f = self.cross_ms_pan1(ms_f, pan_f)

        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        ms_f, residual_ms_f = self.deep_fusion1(ms_f, residual_ms_f, pan_f)
        ms_f = self.patchunembe(ms_f, (h, w))
        pan_f = self.patchunembe(pan_f, (h, w))
        ms_f = self.gate1(ms_ori, ms_f)

        noise_ori = noise_f
        noise_f = self.cross_noise_pan1(noise_f, pan_f)
        noise_f = self.ms_to_token(noise_f)
        pan_f = self.pan_to_token(pan_f)
        residual_noise_f = 0
        noise_f, residual_noise_f = self.deep_fusion2(noise_f, residual_noise_f, pan_f)
        noise_f = self.patchunembe(noise_f, (h, w))
        pan_f = self.patchunembe(pan_f, (h, w))
        noise_f = self.gate2(noise_ori, noise_f)


        ms_ori = ms_f
        ms_f = self.cross_ms_pan2(ms_f, pan_f)

        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        ms_f, residual_ms_f = self.deep_fusion3(ms_f, residual_ms_f, pan_f)
        ms_f = self.patchunembe(ms_f, (h, w))
        pan_f = self.patchunembe(pan_f, (h, w))
        ms_f = self.gate3(ms_ori, ms_f)

        noise_ori = noise_f
        noise_f = self.cross_noise_pan2(noise_f, pan_f)
        noise_f = self.ms_to_token(noise_f)
        pan_f = self.pan_to_token(pan_f)
        noise_f, residual_noise_f = self.deep_fusion4(noise_f, residual_noise_f, pan_f)
        noise_f = self.patchunembe(noise_f, (h, w))
        pan_f = self.patchunembe(pan_f, (h, w))
        noise_f = self.gate4(noise_ori, noise_f)
        
        hrms = self.output(torch.cat([ms_f, noise_f], dim=1)) + ms
        return hrms


if __name__ == "__main__":
    device = 'cuda:3'
    shape = torch.randn(4, 1)
    noise_level = torch.randint_like(shape, high=10).to(device)
    noisy = torch.randn(4, 1, 1, 1).to(device)
    x = torch.randn(4, 1, 1, 1).to(device)
    y = torch.randn(4, 1, 1, 1).to(device)
    model = Net(num_channels=3, device=device).to(device)
    res = model(x, y, noisy, noise_level=noise_level)
    print(res.shape)
