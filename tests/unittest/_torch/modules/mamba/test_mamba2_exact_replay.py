# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mamba2 exact replay: prefill, resumed prefill and decode share the chunked-scan bits."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba import exact_replay
from tensorrt_llm._torch.modules.mamba.cache_manager import Mamba2ExactReplayState
from tensorrt_llm._torch.modules.mamba.exact_replay import (
    pin_ssd_autotune_configs,
    prepare_exact_replay_rows,
    replay_ssd,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    Mamba2Metadata,
    cu_seqlens_to_chunk_indices_offsets_triton,
)
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaStateLayout
from tensorrt_llm.mapping import Mapping

CHUNK = 128
HEADS, GROUPS, HEAD_DIM, DSTATE = 4, 2, 64, 128
D_INNER = HEADS * HEAD_DIM
CONV_DIM = D_INNER + 2 * GROUPS * DSTATE
WIDTH = D_INNER + CONV_DIM + HEADS


@pytest.fixture(autouse=True)
def _pinned_tiles():
    pin_ssd_autotune_configs()


def _randn(*shape, dtype=torch.bfloat16, scale=0.1):
    return torch.randn(*shape, device="cuda", dtype=dtype) * scale


def test_pinned_configurations_leave_one_tile_per_kernel():
    for kernel, kwargs, num_stages, num_warps in exact_replay._PINNED_SSD_CONFIGS:
        assert len(kernel.configs) == 1
        (config,) = kernel.configs
        assert all(config.kwargs[key] == value for key, value in kwargs.items())
        assert num_stages in (None, config.num_stages)
        assert num_warps in (None, config.num_warps)


@pytest.mark.parametrize("length", [5, 128, 300, 513])
@pytest.mark.parametrize("initial", [False, True])
def test_replay_ssd_matches_the_default_scan_of_a_lone_sequence(length, initial):
    torch.manual_seed(length + int(initial))
    x = _randn(1, length, HEADS, HEAD_DIM)
    dt = _randn(1, length, HEADS)
    B = _randn(1, length, GROUPS, DSTATE)
    C = _randn(1, length, GROUPS, DSTATE)
    A = -torch.rand(HEADS, device="cuda")
    D = torch.randn(HEADS, device="cuda")
    dt_bias = torch.randn(HEADS, device="cuda") * 0.1
    init = _randn(1, HEADS, HEAD_DIM, DSTATE, dtype=torch.float32) if initial else None
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    seq_idx = torch.zeros(1, length, device="cuda", dtype=torch.int32)
    chunk_indices = chunk_offsets = None
    if initial:
        chunk_indices, chunk_offsets = cu_seqlens_to_chunk_indices_offsets_triton(
            cu, CHUNK, total_seqlens=length
        )
    expected = torch.empty_like(x)
    mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=CHUNK,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=init,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        cu_seqlens=cu,
        seq_idx=seq_idx,
        return_varlen_states=True,
        return_final_states=False,
        out=expected,
        state_dtype=torch.float32,
    )
    out = torch.empty_like(x)
    states, final = replay_ssd(x, dt, A, B, C, CHUNK, D, dt_bias, init, True, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert states.shape == (1, -(-length // CHUNK), HEADS, HEAD_DIM, DSTATE)
    assert states.dtype == final.dtype == torch.float32
    if init is not None:
        torch.testing.assert_close(states[:, 0], init, rtol=0, atol=0)


def _make_layer():
    layer = Mamba2Mixer.__new__(Mamba2Mixer)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer.tp_nheads, layer.tp_ngroups = HEADS, GROUPS
    layer.head_dim, layer.d_state, layer.chunk_size = HEAD_DIM, DSTATE, CHUNK
    layer.tp_d_inner, layer.tp_conv_dim = D_INNER, CONV_DIM
    layer.conv1d = SimpleNamespace(weight=_randn(CONV_DIM, 4), bias=_randn(CONV_DIM))
    layer.A = -torch.rand(HEADS, device="cuda")
    layer.D = torch.randn(HEADS, device="cuda")
    layer.dt_bias = torch.randn(HEADS, device="cuda") * 0.1
    layer.delta_softplus = True
    layer._mamba_ssm_cache_dtype = torch.float32
    layer._token_major_conv = True
    layer._use_flashinfer = False
    layer._stochastic_rounding_for_flashinfer = False
    layer.selective_state_update_func = selective_state_update
    layer.norm = SimpleNamespace(is_nvfp4=False)
    layer.cache_derived_state()
    layer.aux_steram = torch.cuda.Stream()
    layer.events = [torch.cuda.Event(), torch.cuda.Event()]
    return layer


class _Cache:
    """One layer's recurrent cache plus the replay buffers of Mamba2ExactReplayState."""

    def __init__(self, slots):
        self.conv = _randn(slots, CONV_DIM, 3)
        self.temporal = torch.zeros(slots, HEADS, HEAD_DIM, DSTATE, device="cuda")
        self.replay_x = torch.zeros(
            slots, CHUNK, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        self.replay_dt = torch.zeros(slots, CHUNK, HEADS, device="cuda", dtype=torch.bfloat16)
        self.replay_B = torch.zeros(
            slots, CHUNK, GROUPS, DSTATE, device="cuda", dtype=torch.bfloat16
        )
        self.replay_num_tokens = torch.zeros(slots, dtype=torch.int32, device="cuda")

    def clone(self):
        other = _Cache.__new__(_Cache)
        for name, value in vars(self).items():
            setattr(other, name, value.clone())
        return other

    def layer_cache(self):
        return SimpleNamespace(**vars(self))

    def assert_equal(self, other, slots):
        """Bitwise equality of the state that later steps read: boundary states,
        conv states, token counts and the live part of the partial-chunk buffers."""
        torch.testing.assert_close(self.conv, other.conv, rtol=0, atol=0)
        torch.testing.assert_close(self.replay_num_tokens, other.replay_num_tokens, rtol=0, atol=0)
        for slot in slots:
            count = int(self.replay_num_tokens[slot])
            live = count % CHUNK
            if count >= CHUNK:
                torch.testing.assert_close(
                    self.temporal[slot], other.temporal[slot], rtol=0, atol=0
                )
            for name in ("replay_x", "replay_dt", "replay_B"):
                torch.testing.assert_close(
                    getattr(self, name)[slot, :live],
                    getattr(other, name)[slot, :live],
                    rtol=0,
                    atol=0,
                )


def _metadata(max_batch):
    return SimpleNamespace(
        chunk_size=CHUNK,
        state_indices=torch.zeros(max_batch, dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.zeros(max_batch + 1, dtype=torch.int32, device="cuda"),
        _arange_buffer=torch.arange(max_batch + 1, dtype=torch.int32, device="cuda"),
        _arange_buffer_long=torch.arange(max_batch + 1, dtype=torch.long, device="cuda"),
        has_initial_states=torch.zeros(max_batch, dtype=torch.bool, device="cuda"),
        replay_pos=torch.zeros(max_batch, dtype=torch.long, device="cuda"),
        replay_has_boundary=torch.zeros(max_batch, dtype=torch.bool, device="cuda"),
        replay_conv_has_initial=torch.ones(max_batch, dtype=torch.bool, device="cuda"),
        replay_prefill_n_pre=[],
        replay_prefill_has_boundary=[],
    )


def _prepare(metadata, cache, lengths, prefills, slots, initial):
    """What Mamba2Metadata.prepare does for the replay rows of one step."""
    batch = len(lengths)
    metadata.state_indices[:batch].copy_(torch.tensor(slots, dtype=torch.int32))
    cu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    metadata.cu_seqlens[: batch + 1].copy_(cu)
    metadata.has_initial_states[:batch].copy_(torch.tensor(initial))
    context_lens = torch.tensor(lengths[:prefills], dtype=torch.int32, device="cuda")
    prepare_exact_replay_rows(metadata, cache.replay_num_tokens, batch, prefills, context_lens)


def _attn(cache, lengths, prefills):
    return SimpleNamespace(
        num_contexts=prefills,
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        num_ctx_tokens=sum(lengths[:prefills]),
        num_tokens=sum(lengths),
        kv_cache_manager=SimpleNamespace(
            mamba_layer_cache=lambda _: cache.layer_cache(),
            is_speculative=lambda: False,
        ),
    )


def _run(layer, cache, projected, lengths, prefills, slots, initial, output=None):
    metadata = _metadata(len(lengths))
    _prepare(metadata, cache, lengths, prefills, slots, initial)
    if output is None:
        output = torch.empty(projected.shape[0], D_INNER, device="cuda", dtype=torch.bfloat16)
    layer.forward_core(projected, _attn(cache, lengths, prefills), metadata, None, output)
    return output


@pytest.mark.parametrize("length", [5, 128, 129, 300])
def test_resumed_prefill_and_decode_match_single_shot_prefill(length):
    torch.manual_seed(length)
    layer = _make_layer()
    projected = _randn(length, WIDTH)
    base = _Cache(3)

    full_cache = base.clone()
    full = _run(layer, full_cache, projected, [length], 1, [1], [False])

    # Prefill in three pieces; the later pieces resume from the stored
    # boundary state and partial chunk.
    split_cache = base.clone()
    cuts = sorted({0, length // 3, (2 * length) // 3, length})
    pieces = [
        _run(layer, split_cache, projected[a:b], [b - a], 1, [1], [a > 0])
        for a, b in zip(cuts, cuts[1:])
    ]
    torch.testing.assert_close(torch.cat(pieces), full, rtol=0, atol=0)
    split_cache.assert_equal(full_cache, [1])

    # Prefill a prefix, then decode token by token.
    decode_cache = base.clone()
    prefix = max(1, length // 2)
    outputs = [_run(layer, decode_cache, projected[:prefix], [prefix], 1, [1], [False])]
    for t in range(prefix, length):
        outputs.append(_run(layer, decode_cache, projected[t : t + 1], [1], 0, [1], [True]))
    torch.testing.assert_close(torch.cat(outputs), full, rtol=0, atol=0)
    decode_cache.assert_equal(full_cache, [1])
    assert int(decode_cache.replay_num_tokens[1]) == length


def test_mixed_batch_matches_isolated_rows():
    torch.manual_seed(7)
    layer = _make_layer()
    cache = _Cache(6)
    # Histories: a partial chunk without a boundary (120), a boundary with an
    # empty partial chunk (128), a full partial chunk whose next token folds
    # (255) and a boundary with a partial chunk (300).
    for slot, tokens in {1: 120, 2: 128, 3: 255, 4: 300}.items():
        _run(layer, cache, _randn(tokens, WIDTH), [tokens], 1, [slot], [False])

    lengths, slots = [7, 13, 1, 1, 1], [0, 1, 2, 3, 4]
    initial = [False, True, True, True, True]
    projected = _randn(sum(lengths) + 3, WIDTH)
    original = projected.clone()
    mixed_cache = cache.clone()
    mixed = _run(layer, mixed_cache, projected, lengths, 2, slots, initial)

    outputs, start = [], 0
    for row, length in enumerate(lengths):
        outputs.append(
            _run(
                layer,
                cache,
                projected[start : start + length],
                [length],
                int(row < 2),
                [slots[row]],
                [initial[row]],
            )
        )
        start += length
    torch.testing.assert_close(mixed[:start], torch.cat(outputs), rtol=0, atol=0)
    assert torch.count_nonzero(mixed[start:]) == 0
    torch.testing.assert_close(projected, original, rtol=0, atol=0)
    mixed_cache.assert_equal(cache, slots)
    assert mixed_cache.replay_num_tokens.tolist() == [7, 133, 129, 256, 301, 0]


def test_decode_replay_under_cuda_graph():
    torch.manual_seed(11)
    layer = _make_layer()
    cache = _Cache(4)
    for slot, tokens in {0: 127, 1: 130, 2: 5}.items():
        _run(layer, cache, _randn(tokens, WIDTH), [tokens], 1, [slot], [False])
    slots, lengths = [0, 1, 2], [1, 1, 1]
    inputs = [_randn(3, WIDTH) for _ in range(2)]
    projected = torch.empty_like(inputs[0])
    output = torch.empty(3, D_INNER, device="cuda", dtype=torch.bfloat16)

    def prepare(metadata, cache_):
        _prepare(metadata, cache_, lengths, 0, slots, [True] * 3)

    eager_cache, expected = cache.clone(), []
    metadata = _metadata(3)
    for values in inputs:
        projected.copy_(values)
        prepare(metadata, eager_cache)
        layer.forward_core(projected, _attn(eager_cache, lengths, 0), metadata, None, output)
        expected.append(output.clone())

    # Warm up on a scratch copy, then capture one decode step on a fresh copy.
    scratch, metadata = cache.clone(), _metadata(3)
    projected.copy_(inputs[0])
    prepare(metadata, scratch)
    layer.forward_core(projected, _attn(scratch, lengths, 0), metadata, None, output)
    torch.cuda.synchronize()
    graph_cache, metadata = cache.clone(), _metadata(3)
    attn = _attn(graph_cache, lengths, 0)
    prepare(metadata, graph_cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.forward_core(projected, attn, metadata, None, output)
    graph.replay()
    torch.testing.assert_close(output, expected[0], rtol=0, atol=0)
    # Slot 0 folded its chunk in the first step; the second step reads the new
    # boundary and position through the metadata buffers, not the graph.
    projected.copy_(inputs[1])
    prepare(metadata, graph_cache)
    graph.replay()
    torch.testing.assert_close(output, expected[1], rtol=0, atol=0)
    graph_cache.assert_equal(eager_cache, slots)


def test_exact_replay_state_binds_buffers_and_layer_cache():
    layout = MambaStateLayout(
        layer_mask=(True, False, True),
        pp_layers=(0, 1, 2),
        mamba_pp_layers=(0, 2),
        mapping=Mapping(),
        conv_state_shape=(CONV_DIM, 3),
        ssm_state_shape=(HEADS, HEAD_DIM, DSTATE),
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        n_groups_per_rank=GROUPS,
        max_batch_size=4,
        state_index_capacity=5,
        slot_capacity=5,
        spec_config=None,
    )
    ssm = [torch.zeros(5, HEADS, HEAD_DIM, DSTATE, device="cuda") for _ in range(2)]
    conv = [torch.zeros(5, CONV_DIM, 3, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    state = Mamba2ExactReplayState(CHUNK)
    state.bind(layout, ssm, conv)
    assert state.replay_x.shape == (2, 5, CHUNK, HEADS, HEAD_DIM)
    assert state.replay_dt.shape == (2, 5, CHUNK, HEADS)
    assert state.replay_B.shape == (2, 5, CHUNK, GROUPS, DSTATE)
    assert state.replay_x.dtype == state.replay_dt.dtype == state.replay_B.dtype == torch.bfloat16
    assert state.replay_num_tokens.shape == (5,) and state.replay_num_tokens.dtype == torch.int32
    layer_cache = state.make_layer_cache(1, conv[1], ssm[1])
    assert layer_cache.conv is conv[1] and layer_cache.temporal is ssm[1]
    assert layer_cache.replay_x.data_ptr() == state.replay_x[1].data_ptr()
    assert layer_cache.replay_num_tokens is state.replay_num_tokens
    per_token = HEADS * HEAD_DIM + HEADS + GROUPS * DSTATE
    assert Mamba2ExactReplayState.bytes_per_slot(layout, CHUNK) == 2 * CHUNK * per_token * 2
    state.replay_num_tokens[1] = 9
    state.reset_slots(torch.tensor([1], device="cuda"), [1])
    assert int(state.replay_num_tokens[1]) == 9
    state.shutdown()
    assert state.replay_x is None


def test_metadata_prepare_snapshots_and_advances_the_slot_counts():
    metadata = Mamba2Metadata(max_batch_size=4, chunk_size=8)
    counter = torch.tensor([0, 13, 8, 0], dtype=torch.int32, device="cuda")
    manager = SimpleNamespace(
        mamba_replay_num_tokens=counter,
        get_state_indices=lambda request_ids, is_padding: [2, 1, 0][: len(request_ids)],
    )
    # Row 0: a new prefill on slot 2 (its old count restarts). Row 1: a resumed
    # prefill on slot 1 (13 tokens so far). Row 2: a decode on slot 0.
    seq_lens = torch.tensor([4, 3, 1], dtype=torch.int)
    attn_metadata = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=2,
        num_ctx_tokens=7,
        kv_cache_manager=manager,
        request_ids=[10, 11, 12],
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor([0, 13], dtype=torch.int),
        ),
    )
    metadata.prepare(attn_metadata)
    assert metadata.replay_prefill_n_pre == [0, 5]
    assert metadata.replay_prefill_has_boundary == [False, True]
    assert metadata.replay_pos[:3].tolist() == [0, 5, 0]
    assert metadata.replay_has_boundary[:3].tolist() == [False, True, False]
    assert metadata.replay_conv_has_initial[:3].tolist() == [False, True, True]
    assert counter.tolist() == [1, 16, 4, 0]
