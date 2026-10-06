# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed per-query FP32 reductions over sparse BF16 MLA cache entries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attention(
    Q,
    SWA,
    COMP,
    IDS,
    SINKS,
    OUT,
    SWA_OFFSET,
    COMP_OFFSET,
    H: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    ds = tl.arange(0, D)
    bs = tl.arange(0, B)
    q = tl.load(Q + (row.to(tl.int64) * H + head) * D + ds).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0, tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    if HAS_SINK:
        maximum = tl.load(SINKS + head).to(tl.float32)
        denom = tl.full((), 1, tl.float32)
    for start in range(0, TOPK, B):
        cols = start + bs
        ids = tl.load(IDS + row.to(tl.int64) * TOPK + cols, cols < TOPK, -1).to(tl.int64)
        valid = (cols < TOPK) & (ids >= 0)
        swa = tl.load(
            SWA + (ids - SWA_OFFSET)[:, None] * D + ds[None, :],
            (valid & (cols < WINDOW))[:, None],
            0,
        ).to(tl.float32)
        comp = tl.load(
            COMP + (ids - COMP_OFFSET)[:, None] * D + ds[None, :],
            (valid & (cols >= WINDOW))[:, None],
            0,
        ).to(tl.float32)
        kv = swa + comp
        scores = tl.sum(kv * q[None, :], 1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 0))
        # Empty index tiles must leave the online state unchanged.
        safe_max = tl.where(new_max == -float("inf"), 0, new_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.exp(scores - safe_max)
        denom = denom * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * kv, 0)
        maximum = new_max
    out = tl.where(denom > 0, acc / denom, 0)
    tl.store(OUT + (row.to(tl.int64) * H + head) * D + ds, out)


def deterministic_sparse_mla(
    query: torch.Tensor,
    swa_cache: torch.Tensor,
    compressed_cache: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor | None,
    output: torch.Tensor,
    *,
    swa_offset: int,
    compressed_offset: int,
    window: int,
    scale: float,
) -> None:
    """Attend to an ordered SWA/compressed index list, with negative IDs masked.

    Query has shape [tokens, heads, 512]. Each index is relative to the cache
    pool's common allocation base; offsets locate this layer's cache buffers.
    A sink contributes to the softmax denominator with a zero value vector.
    """
    tokens, heads, dim = query.shape
    assert dim == 512
    assert query.dtype == swa_cache.dtype == compressed_cache.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    assert query.is_contiguous() and indices.is_contiguous() and output.is_contiguous()
    assert swa_cache.is_contiguous() and compressed_cache.is_contiguous()
    assert indices.shape[0] == tokens and output.numel() == query.numel()
    if tokens:
        _sparse_attention[(tokens, heads)](
            query,
            swa_cache,
            compressed_cache,
            indices,
            sinks,
            output,
            swa_offset,
            compressed_offset,
            heads,
            dim,
            indices.shape[1],
            window,
            sinks is not None,
            scale,
            32,
            num_warps=4,
        )
