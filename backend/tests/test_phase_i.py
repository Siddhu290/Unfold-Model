"""Phase I tests: real training loop, checkpoints, restore, diff."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.session import Session
from xray.training import checkpoint_diff, restore_checkpoint, train_stream
from sample_models import build_demo


def make_session(name):
    model, x, meta = build_demo(name)
    load = LoadResult(kind="module", model=model, source=f"demo:{name}")
    return Session(load, extract_architecture(model, x), meta)


def test_vision_training_converges():
    """The toy quadrant task is learnable — loss must genuinely drop."""
    s = make_session("mlp")
    events = list(train_stream(s, steps=60, optimizer="adam", lr=0.005,
                               checkpoint_every=20))
    assert events[0]["event"] == "start" and events[0]["task"] == "quadrant"
    losses = [e["loss"] for e in events if e["event"] == "step"]
    assert len(losses) == 60
    first5, last5 = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last5 < first5 * 0.5, \
        f"training must converge: first5={first5:.3f} last5={last5:.3f}"
    done = events[-1]
    assert done["checkpoints"] == [0, 20, 40, 60]
    print(f"  vision training: loss {first5:.3f} -> {last5:.3f} over 60 real "
          f"steps, checkpoints at {done['checkpoints']} OK")


def test_lm_training_memorizes():
    s = make_session("tiny_transformer")
    events = list(train_stream(s, steps=40, optimizer="adam", lr=0.003,
                               source={"kind": "corpus",
                                       "text": "abcabcabcabc " * 20},
                               checkpoint_every=10))
    losses = [e["loss"] for e in events if e["event"] == "step"]
    assert losses[-1] < losses[0] * 0.7, \
        f"LM must start memorizing a trivial corpus: {losses[0]:.3f} -> {losses[-1]:.3f}"
    print(f"  LM training: CE {losses[0]:.3f} -> {losses[-1]:.3f} on trivial corpus OK")


def test_restore_and_diff_checkpoints():
    s = make_session("mlp")
    w0 = dict(s.model.named_parameters())["fc1.weight"].detach().clone()
    list(train_stream(s, steps=20, optimizer="sgd", lr=0.05, checkpoint_every=10))
    w_trained = dict(s.model.named_parameters())["fc1.weight"].detach().clone()
    assert not torch.equal(w0, w_trained), "training must change weights"

    # diff current (trained) model vs step-0 checkpoint via the shared pipeline
    d = checkpoint_diff(s, 0)
    assert d["other"] == "training step 0"
    assert d["param_diffs"]["fc1.weight"]["update_norm"] > 0
    # the diff detail endpoint (reused UI) now works against this checkpoint
    from xray.comparative import diff_against_detail
    detail = diff_against_detail(s, "fc1.weight")
    assert detail["delta"]["stats"]["l2_norm"] > 0

    # scrub back to step 0: weights must be EXACTLY the pre-training state
    restore_checkpoint(s, 0)
    w_restored = dict(s.model.named_parameters())["fc1.weight"].detach()
    assert torch.equal(w_restored, w0), "restore must be exact"
    print("  checkpoints: diff vs step-0, detail heatmaps, exact restore OK")


def test_wrong_task_rejected():
    s = make_session("tiny_transformer")
    try:
        list(train_stream(s, steps=2, source={"kind": "quadrant"}))
        raise AssertionError("LM on vision task must be rejected")
    except ValueError:
        pass
    print("  guards: task/model mismatch rejected OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase I tests:")
    for t in tests:
        t()
    print("ALL PHASE I TESTS PASSED")
