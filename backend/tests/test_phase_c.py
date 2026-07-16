"""Phase C tests: hook-level activation overrides + causal patching."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.execution import make_input, run_forward
from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.patching import causal_patch
from xray.session import Session
from xray.summarize import first_tensor
from sample_models import build_demo


def make_session(name):
    model, x, meta = build_demo(name)
    load = LoadResult(kind="module", model=model, source=f"demo:{name}")
    return Session(load, extract_architecture(model, x), meta)


def test_override_replaces_output_exactly():
    """Patching a layer's output with zeros must change downstream compute,
    and the recorded output must be the override, not the computed value."""
    session = make_session("mlp")
    x = torch.randn(1, 1, 28, 28)
    base = run_forward(session.model, x)
    fc2_ci = next(r["call_index"] for r in base.records if r["path"] == "fc2")
    zeros = torch.zeros(1, 64)
    patched = run_forward(session.model, x, overrides={fc2_ci: zeros})
    assert patched.records[fc2_ci]["patched"] is True
    assert torch.equal(patched.tensors[fc2_ci], zeros), "record must show the override"
    # downstream: with fc2 output = 0, logits = fc3.bias exactly (ReLU(0)=0)
    expected = dict(session.model.named_parameters())["fc3.bias"].detach()
    got = first_tensor(patched.output)[0]
    assert torch.allclose(got, expected, atol=1e-6), \
        "zero-patch at fc2 must make output == fc3.bias"
    print("  override: replaces output, downstream math exact (logits == fc3.bias) OK")


def test_override_shape_mismatch_rejected():
    session = make_session("mlp")
    x = torch.randn(1, 1, 28, 28)
    base = run_forward(session.model, x)
    ci = next(r["call_index"] for r in base.records if r["path"] == "fc2")
    try:
        run_forward(session.model, x, overrides={ci: torch.zeros(1, 63)})
        raise AssertionError("wrong-shape override must be rejected")
    except (ValueError, RuntimeError):
        pass
    print("  override: wrong-shape patch rejected OK")


def test_full_position_patch_restores_exactly():
    """THE invariant: splicing the clean activation at ALL positions into any
    residual-stream block makes the output identical to the clean run."""
    session = make_session("tiny_transformer")
    out = causal_patch(
        session,
        {"kind": "text", "text": "hello world"},
        {"kind": "text", "text": "jjjjj world"},
        positions="all",
    )
    assert len(out["results"]) == 4
    for r in out["results"]:
        assert r.get("restoration") is not None, r
        assert abs(r["restoration"] - 1.0) < 1e-3, \
            f"{r['path']}: full patch restoration {r['restoration']} != 1.0"
        assert abs(r["p_target"] - out["clean"]["p_target"]) < 1e-5
    print("  causal patch: positions='all' restores clean output EXACTLY at "
          "every block (restoration == 1.0) OK")


def test_diff_position_patch_varies_by_depth():
    session = make_session("tiny_transformer")
    out = causal_patch(
        session,
        {"kind": "text", "text": "hello world"},
        {"kind": "text", "text": "jjjjj world"},
        positions="diff",
    )
    assert [d["pos"] for d in out["diff_tokens"]] == [0, 1, 2, 3, 4]
    vals = [r["restoration"] for r in out["results"] if r.get("restoration") is not None]
    assert len(vals) == 4 and all(isinstance(v, float) for v in vals)
    assert len(set(round(v, 6) for v in vals)) > 1, \
        f"restoration should vary across depth, got {vals}"
    # patching the differing positions right after the FIRST block leaves the
    # later blocks free to re-attend to clean states -> more restoration than
    # patching after the LAST block (where contamination already spread)
    assert vals[0] > vals[-1] - 1e-6, f"expected depth decay-ish: {vals}"
    print(f"  causal patch: positions='diff' varies by depth {['%.3f' % v for v in vals]} OK")


def test_unequal_lengths_rejected():
    session = make_session("tiny_transformer")
    try:
        causal_patch(session, {"kind": "text", "text": "short"},
                     {"kind": "text", "text": "much longer prompt"})
        raise AssertionError("unequal token counts must be rejected")
    except ValueError as e:
        assert "SAME length" in str(e)
    print("  causal patch: unequal-length prompts rejected with clear message OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase C tests:")
    for t in tests:
        t()
    print("ALL PHASE C TESTS PASSED")
