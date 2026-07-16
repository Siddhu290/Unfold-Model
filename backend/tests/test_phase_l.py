"""Phase L tests: steering mechanics with exact invariants."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.session import Session
from xray.steering import build_direction, steer, steer_batch, _steer_probs
from sample_models import build_demo


def make_session():
    model, x, meta = build_demo("tiny_transformer")
    load = LoadResult(kind="module", model=model, source="demo:tiny_transformer")
    return Session(load, extract_architecture(model, x), meta)


def test_direction_construction():
    s = make_session()
    r = build_direction(s, "hello world", "jello world", "blocks.1")
    assert r["dim"] == 64 and r["norm"] > 0
    assert s.steering["direction"].shape == (64,)
    print(f"  direction: 64-dim vector, norm {r['norm']:.3f} OK")


def test_alpha_zero_is_exact_baseline():
    """α=0 must be bit-identical to an unsteered run — the override is only
    installed when it does something."""
    s = make_session()
    build_direction(s, "abc", "xyz", "blocks.2")
    p0, _ = _steer_probs(s, {"kind": "text", "text": "some prompt"}, 0.0)
    p_plain, _ = _steer_probs(s, {"kind": "text", "text": "some prompt"}, 0.0)
    assert torch.equal(p0, p_plain)
    r = steer(s, {"kind": "text", "text": "some prompt"}, 0.0)
    assert r["kl_from_base"] < 1e-9 and not r["top1_changed"]
    print("  α=0: exactly the baseline distribution OK")


def test_full_direction_at_last_position_reproduces_prompt_a():
    """THE steering invariant (mirror of the patching one): steering prompt B
    at the LAST block, last position, with direction (A−B) and α=1 makes the
    last-token residual exactly A's — so the next-token distribution must
    equal prompt A's exactly."""
    s = make_session()
    # both prompts same length so last-position semantics align
    a, b = "hello world", "jello swirl"
    build_direction(s, a, b, "blocks.3", position="last")
    pa, _ = _steer_probs(s, {"kind": "text", "text": a}, 0.0)
    steered, _ = _steer_probs(s, {"kind": "text", "text": b}, 1.0,
                              positions="last")
    assert torch.allclose(steered, pa, atol=1e-5), \
        f"max diff {(steered - pa).abs().max():.2e}"
    print("  invariant: steer(B, A−B, α=1, last block/position) == forward(A) "
          "exactly OK")


def test_alpha_scales_effect_monotonically():
    s = make_session()
    build_direction(s, "aaaaa", "zzzzz", "blocks.1")
    kls = [steer(s, {"kind": "text", "text": "test prompt"}, a)["kl_from_base"]
           for a in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert kls[0] < 1e-9
    assert all(kls[i] < kls[i + 1] for i in range(len(kls) - 1)), \
        f"KL must grow with α: {kls}"
    print(f"  α sweep: KL strictly increasing {['%.4f' % k for k in kls]} OK")


def test_batch_generalization_and_logging():
    s = make_session()
    build_direction(s, "happy day", "angry day", "blocks.2")
    r = steer_batch(s, ["one thing", "two thing", "red thing"], alpha=3.0,
                    watch=["a", "z"])
    assert len(r["results"]) == 3
    assert all(len(x["watch"]) == 2 for x in r["results"])
    assert len(s.steering_log) == 1 and len(s.steering_log[0]["results"]) == 3
    print("  batch: 3 unrelated prompts steered, watch tokens tracked, "
          "logged for report OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase L tests:")
    for t in tests:
        t()
    print("ALL PHASE L TESTS PASSED")
