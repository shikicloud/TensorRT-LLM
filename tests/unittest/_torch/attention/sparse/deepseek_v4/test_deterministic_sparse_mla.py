# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare one-shot context, cached chunks and decode with real paged metadata."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionInputType, MLAParams
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    DeepseekV4TrtllmAttention,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.params import SparseBackendForwardArgs
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.mapping import Mapping

from .test_deepseek_v4_sparse_mla import (
    Scenario,
    _build_compressed_topk_indices,
    _create_cache_manager,
    _create_pos_embd_params,
    _prefill_compress_buffer,
)


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
def test_sparse_mla_context_chunk_decode(monkeypatch, layer_idx):
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        pytest.skip("Requires SM100 or SM103")
    monkeypatch.setenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC", "1")
    # An exact list also catches accidental fallthrough to an ordinary backend.
    monkeypatch.setenv("TLLM_FMHA_LIBS", "deterministic_sparse_mla")
    torch.manual_seed(238)
    lengths, heads, dim = [131, 139], 8, 512
    scenario = Scenario(num_heads=heads, qk_nope_head_dim=448, max_position_embeddings=512)
    ratio = scenario.compress_ratios[layer_idx]
    queries = [torch.randn(n, heads * dim, device="cuda", dtype=torch.bfloat16) for n in lengths]
    latents = [torch.randn(n, dim, device="cuda", dtype=torch.bfloat16) for n in lengths]
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    results = []
    for mode in ("full", "chunk", "decode"):
        manager, config = _create_cache_manager(scenario, lengths, 512)
        try:
            for i, n in enumerate(lengths):
                request = LlmRequest(
                    request_id=i,
                    max_new_tokens=3,
                    input_tokens=list(range(n)),
                    sampling_config=SamplingConfig(),
                    is_streaming=False,
                )
                manager.prepare_context(request)
                manager.resize_context(request, request.context_chunk_size)
            layer = DeepseekV4TrtllmAttention(
                layer_idx=layer_idx,
                num_heads=heads,
                num_kv_heads=1,
                head_dim=dim,
                q_scaling=1.0,
                pos_embd_params=_create_pos_embd_params(scenario),
                mla_params=MLAParams(
                    q_lora_rank=512,
                    kv_lora_rank=448,
                    qk_nope_head_dim=448,
                    qk_rope_head_dim=64,
                    v_head_dim=512,
                    hidden_size=512,
                    rope_append=False,
                    predicted_tokens_per_seq=1,
                ),
                sparse_attention_config=config,
                skip_create_weights_in_init=True,
            )
            layer.attn_sink = torch.nn.Parameter(sink, requires_grad=False)
            layer.update_quant_config(None)
            if ratio > 1:
                torch.manual_seed(931 + layer_idx)
                _prefill_compress_buffer(
                    manager, layer_idx, lengths, [0, 1], dim, torch.device("cuda")
                )

            def forward(starts, counts, context, layer=layer):
                metadata = DeepseekV4TrtllmAttentionMetadata(
                    seq_lens=torch.tensor(counts, dtype=torch.int32),
                    request_ids=[0, 1],
                    max_num_requests=2,
                    num_contexts=2 if context else 0,
                    prompt_lens=lengths,
                    max_num_tokens=sum(lengths),
                    kv_cache_manager=manager,
                    kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=starts),
                    mapping=Mapping(),
                    sparse_attention_config=config,
                )
                metadata.prepare()
                q = torch.cat([t[s : s + c] for t, s, c in zip(queries, starts, counts)])
                latent = torch.cat([t[s : s + c] for t, s, c in zip(latents, starts, counts)])
                q_pe = q.view(-1, heads, dim)[..., 448:]
                indices = None
                if ratio == 4:
                    positions = [p for s, c in zip(starts, counts) for p in range(s, s + c)]
                    indices = _build_compressed_topk_indices(
                        positions, ratio, scenario.index_topk, q.device
                    )
                if not context:
                    cu_q = torch.empty(3, dtype=torch.int32, device="cuda")
                    cu_kv = torch.empty_like(cu_q)
                    counter = torch.empty(1, dtype=torch.uint32, device="cuda")
                    layer.mla_rope_generation(
                        q, q_pe, latent, metadata, cu_q, cu_kv, counter, None, None, None
                    )
                output = layer.forward(
                    q,
                    None,
                    None,
                    metadata,
                    attention_input_type=(
                        AttentionInputType.context_only
                        if context
                        else AttentionInputType.generation_only
                    ),
                    latent_cache=latent,
                    q_pe=q_pe,
                    sparse_backend_args=SparseBackendForwardArgs(topk_indices=indices),
                )
                offsets = torch.tensor(counts, device="cuda").cumsum(0) - 1
                return output[offsets].clone()

            if mode == "full":
                result = forward([0, 0], lengths, True)
            else:
                prefix = [n - 2 for n in lengths]
                forward([0, 0], prefix, True)
                if mode == "chunk":
                    result = forward(prefix, [2, 2], True)
                else:
                    forward(prefix, [1, 1], False)
                    result = forward([n - 1 for n in lengths], [1, 1], False)
            results.append(result)
            del layer
        finally:
            manager.shutdown()
    for result in results[1:]:
        torch.testing.assert_close(result, results[0], rtol=0, atol=0)
