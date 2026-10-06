# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared context/generation arithmetic for BF16 DeepSeek-V4 sparse MLA."""

import math
import os
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
)
from tensorrt_llm._torch.attention.kernels.deterministic_sparse_mla import deterministic_sparse_mla
from tensorrt_llm._utils import get_sm_version

from .interface import Fmha, FmhaPhase

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class DeterministicSparseMlaFmha(Fmha):
    """Use the same FP32 dot products and online softmax for every query."""

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        return (
            (
                os.getenv("FORCE_DETERMINISTIC") == "1"
                or os.getenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC") == "1"
            )
            and getattr(attn.sparse_params, "algorithm", None) == "deepseek_v4"
            and attn.is_mla_enable
            and attn.head_dim == 512
            and attn.qk_nope_head_dim == attn.kv_lora_rank == 448
            and attn.qk_rope_head_dim == 64
            and not attn.rope_append
            and not attn.has_fp8_kv_cache
            and not attn.has_fp4_kv_cache
            and get_sm_version() in (100, 103)
        )

    def _is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: FmhaPhase | None = None,
    ) -> bool:
        del k, v, phase
        return (
            q.dtype == torch.bfloat16
            and forward_args.output is not None
            and forward_args.output.dtype == torch.bfloat16
            and forward_args.output_sf is None
            and not forward_args.enable_dsv4_epilogue_fusion
            and forward_args.update_kv_cache
            and forward_args.attention_mask != CustomAttentionMask.CUSTOM
            and forward_args.attention_input_type
            in (AttentionInputType.context_only, AttentionInputType.generation_only)
            and not metadata.is_cross
            and metadata.beam_width == 1
            and not metadata.use_spec_decoding
            and metadata.helix_position_offsets is None
        )

    def forward(self, q, k, v, metadata, forward_args) -> None:
        from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
            DeepseekV4AttentionType,
        )

        del k, v
        attn = self.attn
        manager = metadata.kv_cache_manager
        query = q.view(-1, attn.num_heads, attn.head_dim)
        if forward_args.attention_input_type == AttentionInputType.context_only:
            # The generation caller has already rotated Q and appended its KV.
            # Context uses the standalone preprocessing op with the same RoPE.
            query = query.clone()
            contexts = metadata.num_contexts
            cached = metadata.cached_token_lens_cuda[:contexts].to(torch.int64)
            lengths = torch.diff(metadata.cu_seq_lens_cuda[: contexts + 1]).to(torch.int64)
            zero = torch.zeros(1, dtype=torch.int64, device=q.device)
            cached_indptr = torch.cat((zero, cached.cumsum(0)))
            total_indptr = torch.cat((zero, (cached + lengths).cumsum(0)))
            attn._ensure_rope_table_size(manager.max_seq_len)
            torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
                query.view(q.shape),
                forward_args.latent_cache,
                contexts,
                cached_indptr,
                total_indptr,
                q.shape[0],
                attn.rotary_cos_sin,
                attn.num_heads,
                attn.qk_nope_head_dim,
                attn.qk_rope_head_dim,
                attn.kv_lora_rank,
                metadata.kv_cache_block_offsets,
                manager.kv_cache_pool_pointers,
                manager.kv_cache_pool_mapping,
                None,
                0,
                attn.get_local_layer_idx(metadata),
                manager.tokens_per_block,
                manager.max_seq_len,
                1,
                attn.quant_mode,
            )

        swa = manager.get_buffers(attn.layer_idx, DeepseekV4AttentionType.SWA)
        token_bytes = attn.head_dim * swa.element_size()
        swa_offset = (swa.data_ptr() - metadata.sparse_mla_base_ptrs[1]) // token_bytes
        compressed, compressed_offset = swa, 0
        if attn.compress_ratio > 1:
            compressed = manager.get_buffers(attn.layer_idx, DeepseekV4AttentionType.COMPRESS)
            compressed_offset = (
                compressed.data_ptr() - metadata.sparse_mla_base_ptrs[attn.compress_ratio]
            ) // token_bytes
        deterministic_sparse_mla(
            query,
            swa,
            compressed,
            forward_args.sparse_runtime_params.sparse_attn_indices,
            forward_args.attention_sinks,
            forward_args.output,
            swa_offset=swa_offset,
            compressed_offset=compressed_offset,
            window=attn.sparse_attention_config.window_size,
            scale=1.0 / (attn.q_scaling * math.sqrt(attn.head_dim)),
        )
