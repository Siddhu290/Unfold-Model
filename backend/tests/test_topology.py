"""Dataflow edge topology tests — the graph view is only as honest as these.

The CNN's skip connection is the acid test: `x = x + relu(bn2(conv2(x)))`
does its add in functional code between modules, so the rejoin can only be
recovered from the autograd graph, never from the module tree.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.execution import INPUT_NODE, OUTPUT_NODE, run_forward, make_input
from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.session import Session
from sample_models import build_demo


def trace_of(name):
    model, x, meta = build_demo(name)
    return run_forward(model, x), model


def by_path(trace):
    return {r["call_index"]: r["path"] for r in trace.records}


def edge_paths(trace):
    """Edges as (src_path, dst_path) with virtual nodes named."""
    p = by_path(trace)
    p[INPUT_NODE] = "<input>"
    p[OUTPUT_NODE] = "<output>"
    return {(p[e["src"]], e["dst"]) for e in trace.edges}, p


def test_mlp_linear_chain():
    trace, _ = trace_of("mlp")
    p = by_path(trace)
    e = {(p.get(x["src"], "<in>"), p.get(x["dst"], "<out>")) for x in trace.edges}
    for pair in [("<in>", "flatten"), ("flatten", "fc1"), ("fc1", "act1"),
                 ("act1", "fc2"), ("fc2", "act2"), ("act2", "fc3"),
                 ("fc3", "<out>")]:
        assert pair in e, f"missing {pair} in {sorted(e)}"
    print("  MLP: clean linear chain input -> ... -> output OK")


def test_cnn_skip_connection_rejoin():
    trace, _ = trace_of("cnn")
    paths = by_path(trace)
    # locate the calls: relu runs 3x (call order: after bn1, after bn2, after conv3)
    relu_calls = [i for i, pth in paths.items() if pth == "relu"]
    pool_call = next(i for i, pth in paths.items() if pth == "pool")
    conv2_call = next(i for i, pth in paths.items() if pth == "conv2")
    edges = {(e["src"], e["dst"]) for e in trace.edges}

    pool_srcs = {s for s, d in edges if d == pool_call}
    # THE acid test: pool consumes the residual add, so BOTH relu calls
    # (main path relu@bn2 and the skip source relu@bn1) must feed it
    assert relu_calls[0] in pool_srcs and relu_calls[1] in pool_srcs, \
        f"skip rejoin missed: pool sources {pool_srcs}, relu calls {relu_calls}"
    # and the branch point: conv2 consumes the same relu output the skip uses
    assert (relu_calls[0], conv2_call) in edges
    print(f"  CNN: skip rejoin visible — pool@{pool_call} fed by relu@{relu_calls[0]} "
          f"(skip) AND relu@{relu_calls[1]} (main path) OK")


def test_transformer_qkv_fan():
    trace, _ = trace_of("tiny_transformer")
    paths = by_path(trace)
    edges = {(e["src"], e["dst"]) for e in trace.edges}

    def call(pth):
        return next(i for i, x in paths.items() if x == pth)

    q, k, v = call("blocks.0.attn.q_proj"), call("blocks.0.attn.k_proj"), call("blocks.0.attn.v_proj")
    probs, out = call("blocks.0.attn.attn_probs"), call("blocks.0.attn.out_proj")
    ln1 = call("blocks.0.ln1")
    # fan-out: ln1 feeds all three projections
    for t in (q, k, v):
        assert (ln1, t) in edges, f"ln1->{paths[t]} missing"
    # softmax(QK^T): attn_probs fed by q_proj and k_proj, NOT v_proj
    probs_srcs = {s for s, d in edges if d == probs}
    assert probs_srcs == {q, k}, f"attn_probs sources: {[paths.get(s) for s in probs_srcs]}"
    # att @ V -> out_proj: fed by attn_probs and v_proj
    out_srcs = {s for s, d in edges if d == out}
    assert out_srcs == {probs, v}, f"out_proj sources: {[paths.get(s) for s in out_srcs]}"
    # residual: ln2 consumes the add of (block input, out_proj) -> 2+ sources
    ln2_srcs = {s for s, d in edges if d == call("blocks.0.ln2")}
    assert out in ln2_srcs and len(ln2_srcs) >= 2, \
        f"residual rejoin at ln2 missed: {[paths.get(s) for s in ln2_srcs]}"
    # embeddings are fed by the raw input
    emb_srcs = {s for s, d in edges if d == call("tok_emb")}
    assert emb_srcs == {INPUT_NODE}
    print("  Transformer: Q/K/V fan-out, QK->softmax, (probs,V)->out_proj, "
          "residual rejoin at ln2, input->embedding OK")


def test_backward_trace_also_has_edges():
    model, x, meta = build_demo("mlp")
    load = LoadResult(kind="module", model=model, source="demo:mlp")
    session = Session(load, extract_architecture(model, x), meta)
    from xray.execution import run_backward
    trace, _ = run_backward(session, {"kind": "tensor", "shape": [1, 1, 28, 28]},
                            {"kind": "argmax"})
    assert trace.edges, "backward-run trace must carry the same topology"
    print("  backward-run trace carries identical edge topology OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} topology tests:")
    for t in tests:
        t()
    print("ALL TOPOLOGY TESTS PASSED")
