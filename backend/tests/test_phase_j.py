"""Phase J tests: saliency + integrated gradients on all three demo models."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.attribution import attribute
from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.session import Session
from sample_models import build_demo


def make_session(name):
    model, x, meta = build_demo(name)
    load = LoadResult(kind="module", model=model, source=f"demo:{name}")
    return Session(load, extract_architecture(model, x), meta)


def test_vision_saliency():
    for name in ("mlp", "cnn"):
        s = make_session(name)
        r = attribute(s, {"kind": "tensor", "shape": [1, 1, 28, 28]},
                      method="saliency")
        assert r["kind"] == "vision"
        assert r["map"]["shape"] == [28, 28]
        assert r["map"]["stats"]["max"] > 0, "saliency must be non-zero"
    print("  vision saliency: 28x28 maps on MLP + CNN OK")


def test_vision_ig_completeness():
    """MLP is piecewise-linear, so IG must satisfy the completeness axiom
    almost exactly: sum(attributions) == logit(x) - logit(baseline)."""
    s = make_session("mlp")
    x = torch.randn(1, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    r = attribute(s, {"kind": "tensor", "shape": [1, 1, 28, 28]},
                  method="ig", steps=32)
    total = r["map"]["stats"]["mean"] * r["map"]["numel"]
    # recompute f(x) - f(0) directly
    with torch.no_grad():
        from xray.execution import make_input
        ex, _, _ = make_input(s, {"kind": "tensor", "shape": [1, 1, 28, 28]})
        tid = r["target"]["id"]
        fx = s.model(ex)[0, tid].item()
        f0 = s.model(torch.zeros_like(ex))[0, tid].item()
    diff = fx - f0
    assert abs(total - diff) < max(0.05 * abs(diff), 0.05), \
        f"completeness violated: sum(attr)={total:.4f} vs f(x)-f(0)={diff:.4f}"
    print(f"  IG completeness (MLP): sum(attr)={total:.4f} ≈ f(x)−f(0)={diff:.4f} OK")


def test_text_saliency():
    s = make_session("tiny_transformer")
    r = attribute(s, {"kind": "text", "text": "hello world"}, method="saliency")
    assert r["kind"] == "text" and r["embedding_layer"] == "tok_emb"
    assert len(r["scores"]) == 11 and len(r["grad_norms"]) == 11
    fracs = [row["frac"] for row in r["scores"]]
    assert abs(sum(fracs) - 1.0) < 1e-5
    assert max(fracs) > 1.5 / 11, "attribution should not be uniform"
    print(f"  text saliency: 11 token scores, peak frac {max(fracs):.2f} OK")


def test_text_ig_reports_completeness():
    s = make_session("tiny_transformer")
    r = attribute(s, {"kind": "text", "text": "abc def"}, method="ig", steps=24)
    c = r["completeness"]
    assert all(k in c for k in
               ("sum_attributions", "logit_input", "logit_baseline", "difference"))
    # midpoint Riemann over 24 steps on a small transformer: should be close
    rel = abs(c["sum_attributions"] - c["difference"]) / max(abs(c["difference"]), 1e-6)
    assert rel < 0.30, f"IG completeness off by {rel*100:.1f}%"
    assert len(r["scores"]) == 7
    print(f"  text IG: completeness within {rel*100:.1f}% "
          f"(Σattr={c['sum_attributions']:.3f}, Δlogit={c['difference']:.3f}) OK")


def test_target_spec_respected():
    s = make_session("tiny_transformer")
    r = attribute(s, {"kind": "text", "text": "xy"},
                  target_spec={"kind": "token", "text": "z"}, method="saliency")
    assert r["target"]["label"] == "z"
    print("  target spec: attribution computed for requested token OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase J tests:")
    for t in tests:
        t()
    print("ALL PHASE J TESTS PASSED")
