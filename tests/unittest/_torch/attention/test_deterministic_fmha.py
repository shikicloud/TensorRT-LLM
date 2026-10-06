# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real runtime selection must keep attention invariant to unrelated requests."""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("page", [32, 64])
def test_deterministic_attention_runtime_phase_invariance(monkeypatch, dim, page):
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        pytest.skip("Requires SM100 or SM103")
    monkeypatch.setenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC", "1")
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    torch.manual_seed(71)
    length, heads, kv_heads = 129, 8, 2
    capacity = math.ceil(length / page) * page
    qkv = torch.randn(length, (heads + 2 * kv_heads) * dim, device="cuda", dtype=torch.bfloat16)
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=capacity, enable_block_reuse=False),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=kv_heads,
        head_dim=dim,
        tokens_per_block=page,
        max_seq_len=capacity,
        max_batch_size=1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
    )
    try:
        manager.add_dummy_requests([0], [length])
        attention = TrtllmAttention(
            layer_idx=0, num_heads=heads, num_kv_heads=kv_heads, head_dim=dim
        )
        results = []
        for context in (True, False):
            tokens = length if context else 1
            metadata = TrtllmAttention.Metadata(
                num_contexts=int(context),
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[length - tokens]
                ),
                seq_lens=torch.tensor([tokens], dtype=torch.int32),
                max_num_requests=1,
                max_num_tokens=length,
                kv_cache_manager=manager,
                request_ids=[0],
                prompt_lens=[length],
            )
            metadata.prepare()
            output = torch.empty(tokens, heads * dim, device="cuda", dtype=torch.bfloat16)
            forward_args = AttentionForwardArgs(
                output=output,
                is_fused_qkv=True,
                attention_input_type=AttentionInputType.context_only
                if context
                else AttentionInputType.generation_only,
            )
            attention.forward(
                qkv[-tokens:].clone(), None, None, metadata, forward_args=forward_args
            )
            results.append(output[-1].cpu())
            del metadata
        del attention
    finally:
        manager.shutdown()
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.parametrize("length", [513, 2049, 8193])
@pytest.mark.parametrize("page", [32, 64])
def test_deterministic_attention_runtime_batch_invariance(monkeypatch, length, page):
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        pytest.skip("Requires SM100 or SM103")
    monkeypatch.setenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC", "1")
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    torch.manual_seed(20261005)
    heads, kv_heads, dim = 24, 4, 128
    pages = math.ceil(length / page)
    qkv = torch.randn(32, (heads + 2 * kv_heads) * dim, device="cuda", dtype=torch.bfloat16)
    cached = torch.randn(pages, 2, kv_heads, page, dim, device="cuda", dtype=torch.bfloat16)
    outputs = []
    for batch in (1, 32):
        manager = KVCacheManager(
            KvCacheConfig(max_tokens=batch * pages * page, enable_block_reuse=False),
            CacheType.SELF,
            num_layers=1,
            num_kv_heads=kv_heads,
            head_dim=dim,
            tokens_per_block=page,
            max_seq_len=pages * page,
            max_batch_size=batch,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.BF16,
        )
        try:
            requests = list(range(batch))
            manager.add_dummy_requests(requests, [length] * batch)
            pool = manager.get_buffers(0)
            pool.zero_()
            indices = [i for i in manager.get_batch_cache_indices(requests, 0)[0] if i >= 0]
            pool.index_copy_(
                0, torch.tensor(indices, device="cuda"), cached.view(pages, *pool.shape[1:])
            )
            metadata = TrtllmAttention.Metadata(
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[length - 1] * batch
                ),
                seq_lens=torch.ones(batch, dtype=torch.int32),
                max_num_requests=batch,
                max_num_tokens=batch,
                kv_cache_manager=manager,
                request_ids=requests,
                prompt_lens=[length] * batch,
            )
            metadata.prepare()
            attention = TrtllmAttention(
                layer_idx=0, num_heads=heads, num_kv_heads=kv_heads, head_dim=dim
            )
            output = torch.empty(batch, heads * dim, device="cuda", dtype=torch.bfloat16)
            forward_args = AttentionForwardArgs(
                output=output,
                is_fused_qkv=True,
                attention_input_type=AttentionInputType.generation_only,
            )
            attention.forward(qkv[:batch], None, None, metadata, forward_args=forward_args)
            outputs.append(output[0].cpu())
            del pool, metadata, attention
        finally:
            manager.shutdown()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
