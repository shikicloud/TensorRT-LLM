# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP4/MXFP8 expert arithmetic across token counts and graph replays."""

import pytest
import torch
from _torch.moe.quantize_utils import MXFP4MXFP8QuantizeUtil
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoECommPlan, MoERunContext
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.parametrize(
    "experts,top_k,hidden,intermediate", [(8, 2, 512, 256), (32, 4, 1024, 512)]
)
def test_mxfp4_phase_reference_and_graph(monkeypatch, experts, top_k, hidden, intermediate):
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        pytest.skip("Requires SM100 or SM103")
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(719)
    quant = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
    routing = RenormalizeMoeRoutingMethod(top_k=top_k)
    config = ModelConfig(
        pretrained_config=PretrainedConfig(
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            torch_dtype=torch.bfloat16,
        ),
        quant_config=quant,
        mapping=Mapping(),
        moe_backend="TRTLLM",
    )
    backend = create_moe_backend(
        moe_cls=TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl,
        routing_method=routing,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        dtype=torch.bfloat16,
        reduce_results=False,
        model_config=config,
        init_load_balancer=False,
        bias=False,
    )
    utility = MXFP4MXFP8QuantizeUtil(
        num_experts=experts,
        dtype=torch.bfloat16,
        intermediate_size=intermediate,
        hidden_size=hidden,
        quant_config=quant,
        bias=False,
    )
    weights, reference_weights, reference_kwargs = utility.prepare_weights_from_backend(backend)
    backend.load_weights([weights])
    backend.post_load_weights()
    backend.cuda()
    reference = utility.create_ref_module(routing, **reference_kwargs)
    reference.load_weights([reference_weights])
    reference.cuda()
    x = torch.randn(257, hidden, dtype=torch.bfloat16, device="cuda") * 0.25
    logits = torch.randn(257, experts, device="cuda", dtype=torch.bfloat16)
    ref = reference(x[-1:], logits[-1:])

    def reject_tuning(*args, **kwargs):
        raise AssertionError("Deterministic MXFP4 must bypass the shape-specific tuning cache")

    monkeypatch.setattr(AutoTuner.get(), "choose_one", reject_tuning)

    def run(inp, scores, output=None):
        ids, scales = routing.apply(scores)
        quantized, sf = backend.quantize_input(inp, post_quant_comm=False)
        ctx = MoERunContext(
            x=quantized,
            x_sf=sf,
            token_selected_experts=ids.to(torch.int32),
            token_final_scales=scales.to(torch.bfloat16),
            output_dtype=torch.bfloat16,
            comm_plan=MoECommPlan(
                input_sf_swizzled=True,
                enable_alltoall=False,
                moe_output=output,
                payload_in_workspace=False,
            ),
        )
        return backend.run_moe(ctx)

    expected = run(x[-1:], logits[-1:])
    assert torch.isfinite(expected).all()
    error = (expected.float() - ref.float()).square().mean().sqrt()
    assert error < 0.03 * ref.float().square().mean().sqrt()
    for tokens in (2, 17, 128, 257):
        output = torch.empty_like(x[-tokens:])
        actual = run(x[-tokens:], logits[-tokens:], output)
        torch.testing.assert_close(actual[-1:], expected, rtol=0, atol=0)

    graph_x, graph_scores = x[-17:].clone(), logits[-17:].clone()
    run(graph_x, graph_scores)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(graph_x, graph_scores)
    graph_x.mul_(0.5)
    graph_scores.neg_()
    graph.replay()
    torch.testing.assert_close(captured, run(graph_x, graph_scores), rtol=0, atol=0)
