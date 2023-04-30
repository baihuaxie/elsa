
import math
import torch
import torch.nn as nn


def compute_t5_relative_position_bias(
        relative_position_bias: nn.Embedding,
        query_length: int,
        key_length: int,
        num_buckets: int,
        max_distance: int,
        is_causal: bool,
    ):
    """Compute the relative position bias by buckets.
    - half of the buckets mapped to exact relative position
    - half of the buckets mapped to logarithmic larger bins
    - relative positions >= max_distance all mapped to the same bucket

    Args:
        relative_position_bias (nn.Embedding): Embedding module of shape = [num_buckets,
            num_heads]. Returns a matrix of shape = [query_length, key_length] with scalar
            positional bias values for each attention head.
        query_length (int): Number of query or context tokens.
        key_length (int): Number of key or memeory tokens.
        num_buckets (int): Number of distinct integer bucket ids (zero indexed) to map the
            relative key - query positions into.
        max_distance (int): Relative positions >= max_distance are mapped into the same
            bucket id = `num_buckets`
        is_causal (bool): For causal attention only the lower triangular part of the
            relative position bias matrix is kept.
    """
    device = relative_position_bias.weight.device
    q_pos = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    k_pos = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    relative_positions = k_pos - q_pos # shape = query_length x key_length

    relative_buckets = 0
    if is_causal:
        relative_positions = -relative_positions.tril(0)
    else:
        num_buckets //= 2
        relative_positions = torch.abs(relative_positions)
        relative_buckets = torch.full_like(
            relative_positions, fill_value=num_buckets
        ).triu(1)

    # for first half of the buckets
    # map positions to exact bucket ids
    max_exact = num_buckets // 2
    is_small = relative_positions < max_exact

    # for the other half of the buckets
    # use logarithmically larger bins to map relative positions
    relative_positions_if_large = max_exact + (
        torch.log(relative_positions / max_exact) /
        math.log(max_distance / max_exact) *
        (num_buckets - max_exact)
    ).to(torch.long)
    # relative positions >= max_distance are all mapped to bucket id = `num_buckets-1`
    relative_positions_if_large = torch.min(
        relative_positions_if_large, torch.full_like(relative_positions_if_large, num_buckets-1)
    )

    # get bucket ids
    relative_buckets += torch.where(is_small, relative_positions, relative_positions_if_large)

    # get relative position bias from buckets
    bias = relative_position_bias(relative_buckets)
    # bias shape: 1, num_heads, query_length, key_length
    bias = bias.permute(2, 0, 1).unsqueeze(0)
    return bias