"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under MIT License.
https://github.com/Stability-AI/stable-audio-tools

-----------------------------------------------------
Simple implementation of (bidirectional) Transformer.
"""

import typing as tp
from functools import reduce
from packaging import version

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast

try:
    assert torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')
except AssertionError as e:
    raise e

from utils.torch_common import exists


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


class RotaryEmbedding(nn.Module):
    """ Rotary positional embedding """

    def __init__(
        self,
        dim,
        interpolation_factor=1.,
        base=10000,
        base_rescale_factor=1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device=device)
        return self.forward(t)

    @autocast('cuda', enabled=False)
    def forward(self, t):
        t = t.to(torch.float32)
        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        return freqs, 1.


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@autocast('cuda', enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta)


class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: tp.Callable,
        use_conv=False,
        conv_kernel_size=3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in,
                                                                                  dim_out * 2, conv_kernel_size, padding=(conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        no_bias=False,
        glu=True,
        use_conv=False,
        conv_kernel_size=3,
        zero_init_output=True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out

        # Default to SwiGLU

        activation = nn.SiLU()

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias=not no_bias) if not use_conv else nn.Conv1d(
                    dim, inner_dim, conv_kernel_size, padding=(conv_kernel_size // 2), bias=not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias=not no_bias) if not use_conv \
            else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding=(conv_kernel_size // 2), bias=not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)

        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        dim_context: tp.Optional[int] = None,
        causal: bool = False,
        zero_init_output: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context else dim
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads
        assert self.num_heads % self.kv_heads == 0

        if dim_context:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        # scaled-dot-product setting
        self.sdp_kwargs = dict(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        )

    def flash_attn(self, q, k, v, causal=None):
        _, heads, q_len, q_dim = q.shape
        kv_heads = k.shape[1]

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if heads != kv_heads:
            # Repeat interleave kv_heads to match q_heads
            heads_per_kv_head = heads // kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        causal = self.causal if (causal is None) else causal

        if q_len == 1:  # k_len > q_len
            causal = False

        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=causal
            )

        return out

    def forward(
        self,
        x: torch.Tensor,
        context: tp.Optional[torch.Tensor] = None,
        rotary_pos_emb=None,
        causal: tp.Optional[bool] = None
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, exists(context)

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h=h)

            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=kv_h), (k, v))
        else:
            # Use fused linear projection
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if exists(rotary_pos_emb) and not has_context:
            freqs, _ = rotary_pos_emb

            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)

            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)

        # Flash attention
        out = self.flash_attn(q, k, v, causal=causal)

        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        out = self.to_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_heads: int = 64,
            cross_attend: bool = False,
            dim_context: tp.Optional[int] = None,
            zero_init_branch_outputs: bool = True
    ):

        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context

        assert dim % dim_heads == 0 and ((dim_context % dim_heads == 0) if dim_context else True)

        self.pre_norm = LayerNorm(dim)
        self.self_attn = Attention(
            dim,
            dim_heads=dim_heads,
            zero_init_output=zero_init_branch_outputs
        )

        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim)
            self.cross_attn = Attention(
                dim,
                dim_heads=dim_heads,
                dim_context=dim_context,
                zero_init_output=zero_init_branch_outputs
            )

        self.ff_norm = LayerNorm(dim)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs)

    def forward(
        self,
        x: torch.Tensor,
        context: tp.Optional[torch.Tensor] = None,
        rotary_pos_emb=None
    ):
        # self attention
        x = x + self.self_attn(self.pre_norm(x), rotary_pos_emb=rotary_pos_emb)

        # cross attention
        if exists(context):
            x = x + self.cross_attn(self.cross_attend_norm(x), context=context)

        # MLP
        x = x + self.ff(self.ff_norm(x))

        return x


class ContinuousTransformer(nn.Module):
    """ Bidirectional Transformer """

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        dim_in: tp.Optional[int] = None,
        dim_out: tp.Optional[int] = None,
        dim_heads: int = 64,
        cross_attend: bool = False,
        cond_token_dim: tp.Optional[int] = None,
        rotary_pos_emb: bool = True,
        zero_init_branch_outputs: bool = True,
        use_checkpoint: bool = True,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out else nn.Identity()
        self.rotary_pos_emb = RotaryEmbedding(dim_heads // 2) if rotary_pos_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads=dim_heads,
                    cross_attend=cross_attend,
                    dim_context=cond_token_dim,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    **kwargs
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        prepend_cond: torch.Tensor,
        ** kwargs
    ):
        # bs, seq, dim = x.shape
        # bs, _, dim = prepend_cond.shape

        x = self.project_in(x)

        # prepend global conditions
        len_pre = prepend_cond.shape[1]
        x = torch.cat([prepend_cond, x], dim=1)

        # Rotary positional embedding
        rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1]) if self.rotary_pos_emb else None

        # Iterate over the transformer layers
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = checkpoint(layer, x, rotary_pos_emb=rotary_pos_emb, **kwargs)
            else:
                x = layer(x, rotary_pos_emb=rotary_pos_emb, **kwargs)

        x = x[:, len_pre:, :]
        x = self.project_out(x)

        return x
