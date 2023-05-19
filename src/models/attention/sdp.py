"""Original scaled dot produce attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pe.relative_position_bias import compute_t5_relative_position_bias


class PytorchScaledDotProductAttention(nn.Module):
    """Benchmark Pytorch 2.0 Attention implementations.
    - FLASH attention
    - memory-efficient xformer attention -> which one(s)?
    - Pytorch c++ implementation -> of what? standard sdp attention?
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.num_heads == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # causal mask
        self.register_buffer(
            "bias", torch.tril(torch.ones(config.block_size, config.block_size))
                    .view(1, 1, config.block_size, config.block_size)
        )
        self.n_embed = config.n_embed
        self.num_heads = config.num_heads
        self.attn_method = config.attn_method

    def forward(self, x):
        pass




class ScaledDotProductAttentionForT5(nn.Module):
    """For T5 model with relative position bias.
    """
    def __init__(self, config, has_relative_attention_bias=False, is_decoder=False):
        super().__init__()
        inner_dim = config.d_kv * config.num_heads
        # TODO: bias might be useful for long context?
        self.q = nn.Linear(config.d_model, inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, inner_dim, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.o = nn.Linear(inner_dim, config.d_model, bias=False)

        self.d_kv = config.d_kv
        self.inner_dim = inner_dim
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.is_decoder = is_decoder

        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                config.relative_attention_num_buckets, config.num_heads
            )
        self.has_relative_position_bias = has_relative_attention_bias

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, ScaledDotProductAttentionForT5):
                d_model = self.d_model
                d_kv = self.d_kv
                inner_dim = self.inner_dim
                module.q.weight.data.normal_(
                    mean=0.0,
                    std=(d_model*d_kv)**-0.5,
                )
                module.k.weight.data.normal_(mean=0.0, std=d_model**-0.5)
                module.v.weight.data.normal_(mean=0.0, std=d_model**-0.5)
                module.o.weight.data.normal_(mean=0.0, std=inner_dim**-0.5)
                if module.has_relative_position_bias:
                    module.relative_attention_bias.weight.data.normal_(mean=0.0, std=d_model**-0.5)

    def forward(
        self,
        x,
        attention_mask=None,
        position_bias=None,
        key_value_states=None,
    ):
        b, t, d = x.shape
        key_length = t if key_value_states is None else key_value_states.shape[1]
        # b, nh, t, hs
        query = self.q(x).view(b, t, self.num_heads, -1).transpose(1, 2)
        # self-attention
        if key_value_states is None:
            key = self.k(x).view(b, t, self.num_heads, -1).transpose(1, 2)
            value = self.v(x).view(b, t, self.num_heads, -1).transpose(1, 2)
        # cross-attention
        else:
            # `key_value_states` i.e. `encoder_hidden_states` might have
            # different `seq_len` dimension
            key = self.k(key_value_states).view(
                b, -1, self.num_heads, self.d_kv
            ).transpose(1, 2)
            value = self.v(key_value_states).view(
                b, -1, self.num_heads, self.d_kv
            ).transpose(1, 2)

        # (b, nh, t, key_length)
        # note: HF's T5 model does not use scale by \sqrt{d_kv}
        attn_scores = query @ key.transpose(-2, -1)
        if position_bias is None:
            if not self.has_relative_position_bias:
                # TODO: use batch = 1 or batch_size?
                # is it broadcastable?
                position_bias = torch.zeros(
                    (1, self.num_heads, t, key_length),
                    device=attn_scores.device,
                    dtype=attn_scores.dtype,
                )
            else:
                # (1, nh, t, key_length)
                position_bias = compute_t5_relative_position_bias(
                    self.relative_attention_bias,
                    query_length=t,
                    key_length=key_length,
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    is_causal=self.is_decoder,
                )
            # attention_mask shape (b, nh, t, key_length)
            if attention_mask is not None:
                position_bias = position_bias + attention_mask

        attn_scores += position_bias

        #if attention_mask is not None:
        #    # TODO: shape of `mask`?
        #    attn_scores = attn_scores.masked_fill(attention_mask==0, float("-inf"))

        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        out = attn_scores @ value
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.o(out)

        return out, position_bias


class ScaledDotProductAttentionForGPT(nn.Module):
    """Vanilla attention for GPT-2/GPT-J models.
    - absolute position bias
    - built-in causal attention mask + passed padding mask
    """
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)
        # causal mask
        # note: use .register_buffer() instead of nn.Parameter()
        # because the latter will update the tensor in backward-pass.
        if not config.flash:
            self.register_buffer(
                "bias", torch.tril(torch.ones(config.block_size, config.block_size))
                        .view(1, 1, config.block_size, config.block_size)
            )
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.flash =config.flash
        self.dropout_prob = config.hidden_dropout_prob

    def forward(self, x):
        # TODO: add a padding `attention_mask`? is it not needed for pretraining? for inference?
        b, t, d = x.shape
        q, k, v = self.c_attn(x).split(self.n_embed, dim=-1)

        # b, nh, t, hs -> hs x nh = d
        q = q.view(b, t, self.n_head, -1).transpose(1, 2)
        k = k.view(b, t, self.n_head, -1).transpose(1, 2)
        v = v.view(b, t, self.n_head, -1).transpose(1, 2)

        # use FLASh attention
        if self.flash:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout_prob,
                    is_causal=True,
                )
        # use slow attention
        else:
            attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn_scores = attn_scores.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
            attn_scores = nn.functional.softmax(attn_scores, dim=-1)
            attn_scores = self.attn_dropout(attn_scores)
            out = attn_scores @ v   # b, nh, t, hs

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.resid_dropout(self.c_proj(out))

        return out

