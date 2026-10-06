# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent reference and phase checks for the sparse-attention experiment."""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.deterministic_sparse_mla import deterministic_sparse_mla


@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_phase_reference_and_graph(with_sink):
    torch.manual_seed(419)
    n, heads, dim, window, topk = 17, 4, 512, 128, 160
    q = torch.randn(n, heads, dim, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(260, dim, device="cuda", dtype=torch.bfloat16)
    comp = torch.randn(70, dim, device="cuda", dtype=torch.bfloat16)
    ids = torch.cat(
        (
            torch.randint(260, (n, window), device="cuda") + 123,
            torch.randint(70, (n, topk - window), device="cuda") + 9000,
        ),
        1,
    ).int()
    ids[torch.rand(n, topk, device="cuda") < 0.2] = -1
    ids[0] = -1
    sinks = torch.randn(heads, device="cuda")
    out = torch.empty_like(q)

    def launch(query, indices, output, sink):
        deterministic_sparse_mla(
            query,
            swa,
            comp,
            indices,
            sink,
            output,
            swa_offset=123,
            compressed_offset=9000,
            window=window,
            scale=1 / math.sqrt(dim),
        )

    def reference(query, indices, sink):
        keys = torch.cat(
            (
                swa[(indices[:, :window] - 123).clamp(0)],
                comp[(indices[:, window:] - 9000).clamp(0)],
            ),
            1,
        ).double()
        scores = torch.einsum("nhd,nkd->nhk", query.double(), keys) / math.sqrt(dim)
        scores.masked_fill_(indices[:, None, :] < 0, -float("inf"))
        if sink is not None:
            scores = torch.cat((scores, sink.double()[None, :, None].expand(n, -1, -1)), -1)
        probabilities = scores.softmax(-1)[..., :topk].nan_to_num()
        return torch.einsum("nhk,nkd->nhd", probabilities, keys).to(query.dtype)

    for sink in [sinks if with_sink else None]:
        launch(q, ids, out, sink)
        ref = reference(q, ids, sink)
        torch.testing.assert_close(out, ref, rtol=0.008, atol=0.002)
        for row in [0, 1, 8, 16]:
            single = torch.empty_like(q[row : row + 1])
            launch(q[row : row + 1], ids[row : row + 1], single, sink)
            torch.testing.assert_close(out[row : row + 1], single, rtol=0, atol=0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(q, ids, out, sink)
        q.mul_(0.5)
        swa.add_(0.125)
        comp.mul_(0.75)
        sinks.sub_(0.5)
        graph.replay()
        torch.testing.assert_close(out, reference(q, ids, sink), rtol=0.008, atol=0.002)
        eager = torch.empty_like(q)
        launch(q, ids, eager, sink)
        torch.testing.assert_close(out, eager, rtol=0, atol=0)
