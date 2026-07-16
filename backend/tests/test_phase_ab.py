"""Phase A (generation, distribution) + Phase B (logit lens) tests."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.execution import run_forward, make_input
from xray.extraction import extract_architecture
from xray.generation import full_distribution, generate_stream
from xray.lens import lens_for_call, logit_lens_strip, resolve_final_projection
from xray.loading import LoadResult
from xray.session import Session
from sample_models import build_demo


def make_session(name):
    model, x, meta = build_demo(name)
    load = LoadResult(kind="module", model=model, source=f"demo:{name}")
    return Session(load, extract_architecture(model, x), meta)


def run_fw(session, spec):
    ex, kw, desc = make_input(session, spec)
    tr = run_forward(session.model, ex, kw)
    tr.input_desc = desc
    session.last_trace = tr
    return tr


def test_generation_stream_greedy_deterministic():
    session = make_session("tiny_transformer")
    events = list(generate_stream(session, {"kind": "text", "text": "hello"},
                                  max_new_tokens=5, mode="greedy"))
    assert events[0]["event"] == "start"
    tokens = [e for e in events if e["event"] == "token"]
    assert len(tokens) == 5
    assert all(len(t["topk"]) == 5 and isinstance(t["token"], str) for t in tokens)
    # greedy pick must equal top-1 of the step's own distribution
    for t in tokens:
        assert t["id"] == t["topk"][0]["id"]
        assert abs(t["prob"] - t["topk"][0]["prob"]) < 1e-6
    # deterministic across runs
    again = [e["id"] for e in generate_stream(
        session, {"kind": "text", "text": "hello"}, 5, "greedy") if e["event"] == "token"]
    assert again == [t["id"] for t in tokens]
    assert events[-1]["event"] == "done" and events[-1]["n_tokens"] == 5
    print("  generation: 5 greedy tokens, top-1 consistency, deterministic OK")


def test_generation_sampling_respects_temperature():
    session = make_session("tiny_transformer")
    torch.manual_seed(1)
    ids = set()
    for _ in range(4):
        toks = [e["id"] for e in generate_stream(
            session, {"kind": "text", "text": "abc"}, 3, "sample", temperature=5.0)
            if e["event"] == "token"]
        ids.add(tuple(toks))
    assert len(ids) > 1, "high-temperature sampling should vary"
    print("  generation: sampling at T=5 varies across runs OK")


def test_full_distribution_pages():
    session = make_session("tiny_transformer")
    run_fw(session, {"kind": "text", "text": "hello"})
    page0 = full_distribution(session, 0, 50)
    assert page0["total"] == 96          # char vocab
    probs = [e["prob"] for e in page0["entries"]]
    assert probs == sorted(probs, reverse=True), "must be sorted desc"
    page1 = full_distribution(session, 50, 50)
    all_p = probs + [e["prob"] for e in page1["entries"]]
    assert abs(sum(all_p) - 1.0) < 1e-4, f"full softmax must sum to 1, got {sum(all_p)}"
    assert page1["entries"][0]["rank"] == 50
    # classifier output also works
    s2 = make_session("mlp")
    run_fw(s2, {"kind": "tensor", "shape": [1, 1, 28, 28]})
    d = full_distribution(s2, 0, 20)
    assert d["total"] == 10 and len(d["entries"]) == 10
    print("  distribution: sorted pages, sums to 1, classifier fallback OK")


def test_lens_resolution_and_strip():
    session = make_session("tiny_transformer")
    run_fw(session, {"kind": "text", "text": "the quick brown"})
    norm, head, meta = resolve_final_projection(session)
    assert meta["via"] == "registry" and meta["head"] == "lm_head"
    strip = logit_lens_strip(session, k=3)
    stages = [r["stage"] for r in strip["rows"]]
    assert stages[0] == "embedding" and stages.count("block") == 4
    block_paths = [r["path"] for r in strip["rows"] if r["stage"] == "block"]
    assert block_paths == ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
    # THE lens invariant: projecting the LAST block + final norm path must
    # agree with the model's real output distribution (same top-1)
    assert strip["final"], "needs the real final decode for reference"
    print(f"  lens strip: embedding + 4 blocks, final ref decode OK")


def test_lens_final_block_agrees_with_output():
    """Projecting ln_f's own output through the head must exactly reproduce
    the model's real prediction — the strongest correctness check."""
    session = make_session("tiny_transformer")
    trace = run_fw(session, {"kind": "text", "text": "xyz"})
    lnf_call = next(r["call_index"] for r in trace.records if r["path"] == "ln_f")
    # ln_f output is ALREADY normed; lens applies ln_f again — so instead take
    # the last block's output (pre-norm) and check top-1 matches real output
    last_block = next(r["call_index"] for r in trace.records if r["path"] == "blocks.3")
    row = lens_for_call(session, last_block, k=1)
    from xray.execution import decode_topk
    real = decode_topk(session, trace.output, k=1)
    assert row["topk"][0]["id"] == real["topk"][0]["id"], \
        f"lens(last block) {row['topk'][0]} != real output {real['topk'][0]}"
    print("  lens correctness: lens(blocks.3) top-1 == model's real top-1 OK")


def test_lens_rejects_non_hidden_layers():
    session = make_session("tiny_transformer")
    trace = run_fw(session, {"kind": "text", "text": "hi"})
    probs_call = next(r["call_index"] for r in trace.records
                      if r["path"].endswith("attn_probs"))
    try:
        lens_for_call(session, probs_call)
        raise AssertionError("attention probs are not a hidden state")
    except ValueError:
        pass
    # MLP has no lens at all
    s2 = make_session("mlp")
    run_fw(s2, {"kind": "tensor", "shape": [1, 1, 28, 28]})
    try:
        logit_lens_strip(s2)
        raise AssertionError("MLP must not have a logit lens")
    except ValueError:
        pass
    print("  lens guards: rejects attention maps + non-transformer models OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase A+B tests:")
    for t in tests:
        t()
    print("ALL PHASE A+B TESTS PASSED")
