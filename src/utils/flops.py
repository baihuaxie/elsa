
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
try:
    from deepspeed.profiling.flops_profiler import get_model_profile
    has_deepspeed_profiler = True
except ImportError:
    has_deepspeed_profiler = False


def create_input_labels_dict(batch_size, seq_len, vocab_size, dtype, device):
    # Create input_ids tensor of shape (batch_size, seq_len) with random values
    input_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_len), dtype=dtype, device=device
    )

    # Create labels tensor by shifting input_ids one position to the
    # right along the seq_len dimension
    # note that the shift here is circular
    # it doesn't matter since this is only for counting flops
    labels = torch.roll(input_ids, shifts=-1, dims=1)

    # Create and return a dictionary with input_ids and labels tensors
    input_labels_dict = {"input_ids": input_ids, "labels": labels}
    return input_labels_dict


def profile_deepspeed(
    model: nn.Module,
    batch_size = 1,
    seq_len = 1024,
    vocab_size = 50257,
    dtype = torch.long,
    detailed = True,
):
    inputs = create_input_labels_dict(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=dtype,
            device=next(model.parameters()).device,
    )
    # [TD]: backward pass flops not covered?
    flops, macs, params = get_model_profile(
            model=model,
            kwargs=inputs,             # kwarg inputs to model
            print_profile=detailed,    # print the model profile
            detailed=detailed,         # print profile attached to each module
            warm_up=10,                # number of warmup steps before profiling
            as_string=False,           # print human-readable strings e.g 1k or raw numbers
            output_file=None,          # print to console
            ignore_modules=None,       # no ignored modules in profiling
    )
    return flops, macs, params


def profile_torch(
    model: nn.Module,
    batch_size = 1,
    seq_len = 1024,
    vocab_size = 50257,
    dtype = torch.long,
):
    inputs = create_input_labels_dict(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=dtype,
            device=next(model.parameters()).device,
    )
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        loss, _ = model(**inputs)
        loss.backward()
        # TODO: add optimizer step here?

    # [TD: 05-04-2023] backward pass flops coverage ok?
    # https://github.com/pytorch/pytorch/issues/69782
    total_flops = sum([int(evt.flops) for evt in prof.events()])

    return total_flops



def flops_estimate(model: nn.Module, batch_size = 1, seq_len = 1024):
    """A theoretical estimate of FLOPs based on GPT architecture.
    """
    vocab_size = model.config.vocab_size
    d_model = model.config.n_embed
    n_head = model.config.n_head
    # TODO: update GPTConfig w.t. d_kv and d_ff
    d_kv = d_model // n_head
    d_ff = d_model * 4
    n_layer = model.config.n_layer

    # token embedding layer
    wte = d_model * vocab_size * seq_len

    # abs position embedding layer
    wpe = d_model * seq_len

    # attention
    # exclude: dropout, rel_pos_bias, mask
    # assumes abs_pos_bias
    attn_qkv_proj = 2 * 3 * d_model * (d_kv * n_head) * seq_len
    attn_qk_dot = 2 * seq_len * (d_kv * n_head) * seq_len
    attn_softmax = 1 * seq_len * 4 * n_head * seq_len  # 4: sub, exp, sum, div
    attn_pool = 2 * seq_len * (d_kv * n_head) * seq_len
    attn_proj = 2 * (d_kv * n_head) * d_model * seq_len
    attn_per_layer= attn_qkv_proj + attn_qk_dot + attn_softmax + attn_pool + attn_proj

    # mlp
    # exclude: dropout
    mlp_proj = 2 * 2 * d_model * d_ff * seq_len
    mlp_act = 2 * d_ff * seq_len
    mlp_per_layer = mlp_proj + mlp_act

    # layer norm
    # 8:
    # mean: 1 add
    # var: 1 sub + 1 square + 1 add
    # norm: 1 sub + 1 div
    # scale: 1 mul
    # shift: 1 add
    # 2: two ln per layer
    ln_layer = 8 * d_model * seq_len * 2

    # total flops per layer
    total_per_layer = attn_per_layer + mlp_per_layer + ln_layer

    # final layer norm
    ln_final = 8 * d_model * seq_len

    # lm_head
    lm_head = 2 * d_model * vocab_size * seq_len

    # total fwd flops
    total_fwd_flops = wte + wpe + n_layer * total_per_layer + ln_final + lm_head

    # bwd flops ~= 2x fwd flops
    total_bwd_flops = 2 * total_fwd_flops

    # total flops
    total_flops = (total_fwd_flops + total_bwd_flops) * batch_size

    return total_flops, total_fwd_flops, total_bwd_flops


    



