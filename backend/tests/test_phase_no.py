"""Phase N (aggregation) + Phase O (robustness) tests."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.aggregate import aggregate_attribution_stream, aggregate_heads_stream
from xray.attribution import attribute
from xray.comparative import ablate_heads
from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.robustness import fgsm_sweep, token_substitution_stream
from xray.session import Session
from xray.training import train_stream
from sample_models import build_demo


def make_session(name):
    model, x, meta = build_demo(name)
    load = LoadResult(kind="module", model=model, source=f"demo:{name}")
    s = Session(load, extract_architecture(model, x), meta)
    s.last_input_spec = ({"kind": "text", "text": "hello"}
                         if name == "tiny_transformer"
                         else {"kind": "tensor", "shape": [1, 1, 28, 28]})
    return s


def test_aggregate_heads_batch_of_one_matches_single():
    """THE consistency regression: batch aggregation with one prompt must
    equal the Phase F single-example numbers exactly."""
    s = make_session("tiny_transformer")
    single = ablate_heads(s, {"kind": "text", "text": "hello world"},
                          "blocks.0.attn")
    agg = list(aggregate_heads_stream(s, "blocks.0.attn",
                                      prompts=["hello world"]))[-1]
    for h in agg["heads"]:
        ref = next(x["delta"] for x in single["heads"] if x["head"] == h["head"])
        assert abs(h["mean_delta"] - ref) < 1e-12, (h, ref)
        assert h["min_delta"] == h["max_delta"] == h["mean_delta"]
    print("  N consistency: aggregate(batch=1) == single ablate_heads exactly OK")


def test_aggregate_heads_multi_prompt():
    s = make_session("tiny_transformer")
    events = list(aggregate_heads_stream(
        s, "blocks.1.attn",
        prompts=["hello world", "abc def ghi", "one two three"]))
    start = events[0]
    assert start["estimate_s"] >= 0 and start["n_prompts"] == 3
    assert sum(1 for e in events if e["event"] == "prompt") == 3
    done = events[-1]
    assert len(done["heads"]) == 4
    assert all(0 <= h["top3_frac"] <= 1 for h in done["heads"])
    assert any("aggregate_heads" == x["kind"] for x in s.analysis_log)
    print(f"  N heads: 3 prompts aggregated, top3_frac per head "
          f"{[h['top3_frac'] for h in done['heads']]} OK")


def test_aggregate_attribution():
    s = make_session("tiny_transformer")
    done = list(aggregate_attribution_stream(
        s, prompts=["hello world", "xyz abc"]))[-1]
    assert done["n_prompts"] == 2
    assert 0 < done["mean_top_frac"] <= 1
    assert all("top_token" in r for r in done["rows"])
    print(f"  N attribution: mean concentration "
          f"{done['mean_top_frac']:.2f} across 2 prompts OK")


def test_fgsm_degradation_curve():
    s = make_session("mlp")
    r = fgsm_sweep(s, {"kind": "tensor", "shape": [1, 1, 28, 28]})
    ps = [c["p_top1"] for c in r["curve"]]
    assert ps[-1] < r["p_top1"], "large-ε FGSM must reduce top-1 confidence"
    assert ps == sorted(ps, reverse=True) or ps[0] > ps[-1], \
        f"confidence should broadly degrade with ε: {ps}"
    assert r["curve"][-1]["flipped"], "ε=0.3 on an untrained MLP must flip"
    print(f"  O FGSM: p(top1) {r['p_top1']:.3f} → {ps[-1]:.3f} across ε, "
          f"flip at ε=0.3 OK")


def test_token_substitution_finds_planted_fragility():
    """Plant ground truth: train the char-LM to memorize 'xyzxyz…' so that
    after 'xy' it predicts 'z' BECAUSE of those exact tokens — substitution
    search must find flips, and the most fragile position must be a real one."""
    s = make_session("tiny_transformer")
    list(train_stream(s, steps=60, optimizer="adam", lr=0.005,
                      source={"kind": "corpus", "text": "xyzxyzxyzxyz " * 30}))
    events = list(token_substitution_stream(
        s, {"kind": "text", "text": "xyzxy"}, k_neighbors=8))
    start, done = events[0], events[-1]
    assert start["n_runs"] == 5 * 8 and start["estimate_s"] >= 0
    assert done["baseline_top1"] == "z", f"model should predict z, got {done['baseline_top1']}"
    n_flips = sum(p["flips"] for p in done["positions"])
    assert n_flips >= 1, "substituting memorized-context tokens must flip"
    print(f"  O substitution: trained LM predicts 'z', {n_flips}/5 positions "
          f"flippable, most fragile: {done['positions'][0]['token']!r} OK")


def test_cross_check_with_attribution():
    s = make_session("tiny_transformer")
    list(train_stream(s, steps=60, optimizer="adam", lr=0.005,
                      source={"kind": "corpus", "text": "xyzxyzxyzxyz " * 30}))
    spec = {"kind": "text", "text": "xyzxy"}
    s.last_attribution = attribute(s, spec, method="saliency")
    done = list(token_substitution_stream(s, spec, k_neighbors=6))[-1]
    cc = done["cross_check"]
    assert cc is not None, "cross-check must run when attribution matches prompt"
    assert 0 <= cc["overlap_top5"] <= 5 and "note" in cc
    print(f"  O cross-check: {cc['overlap_top5']}/5 overlap between "
          f"attribution and fragility, note surfaced OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase N+O tests:")
    for t in tests:
        t()
    print("ALL PHASE N+O TESTS PASSED")
