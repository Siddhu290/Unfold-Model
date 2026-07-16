"""Milestone 3/4 tests: real forward trace, real gradients, real weight update."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.execution import (
    decode_topk, make_input, optimizer_step, run_backward, run_forward, undo_step,
)
from xray.session import Session
from xray.theory import get_theory
from sample_models import build_demo


def make_session(name):
    model, x, meta = build_demo(name)
    load = LoadResult(kind="module", model=model, source=f"demo:{name}")
    arch = extract_architecture(model, x)
    return Session(load, arch, meta), x


def test_forward_trace_matches_execution():
    session, x = make_session("cnn")
    trace = run_forward(session.model, x)
    # every record is a real module call in execution order
    paths = [r["path"] for r in trace.records]
    assert paths[0] == "conv1"
    assert paths.count("relu") == 3, "relu is called 3x in forward()"
    assert paths[-1] == "", "root module finishes last"
    final = trace.records[-1]
    assert final["out_shape"] == [1, 10]
    # captured values are the real computation: re-run and compare
    with torch.no_grad():
        expected = session.model(x)
    got = trace.tensors[len(trace.records) - 1]
    assert torch.allclose(expected, got), "hooked output != real output"
    print("  forward: hook order, module reuse, captured values == real output OK")


def test_forward_llm_decode():
    session, _ = make_session("tiny_transformer")
    x, kwargs, desc = make_input(session, {"kind": "text", "text": "the cat sat"})
    assert desc["tokens"][0] == "t"
    trace = run_forward(session.model, x, kwargs)
    top = decode_topk(session, trace.output, k=5)
    assert len(top["topk"]) == 5
    assert abs(sum(e["prob"] for e in top["topk"])) <= 1.0
    assert all(isinstance(e["label"], str) for e in top["topk"])
    print(f"  LLM decode: top-5 next chars {[e['label'] for e in top['topk']]} OK")


def test_backward_real_gradients():
    session, _ = make_session("mlp")
    trace, result = run_backward(
        session,
        {"kind": "tensor", "shape": [1, 1, 28, 28]},
        {"kind": "class", "index": 3},
    )
    assert result["loss"] > 0
    assert result["loss_desc"]["loss_fn"] == "cross_entropy"
    # gradient check against manual computation
    g = result["param_grads"]
    assert set(g) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                      "fc3.weight", "fc3.bias"}
    # activation grads captured at intermediate layers
    assert len(result["activation_grads"]) >= 4
    # verify the actual grad tensor on the model is the real dL/dW:
    # re-run manually and compare fc3.bias grad = softmax(logits) - onehot
    model = session.model
    x, _, _ = make_input(session, {"kind": "tensor", "shape": [1, 1, 28, 28]})
    with torch.no_grad():
        logits = model(x)
        expected_bias_grad = torch.softmax(logits[0], -1)
        expected_bias_grad[3] -= 1.0
    actual = dict(model.named_parameters())["fc3.bias"].grad
    assert torch.allclose(actual, expected_bias_grad, atol=1e-5), \
        "fc3.bias grad != softmax(logits) - onehot  (the analytic CE gradient)"
    print("  backward: real grads on all 6 params, matches analytic CE gradient OK")


def test_optimizer_step_diff_and_undo():
    session, _ = make_session("mlp")
    run_backward(session, {"kind": "tensor", "shape": [1, 1, 28, 28]},
                 {"kind": "class", "index": 0})
    model = session.model
    w_before = dict(model.named_parameters())["fc1.weight"].detach().clone()
    g = dict(model.named_parameters())["fc1.weight"].grad.clone()

    result = optimizer_step(session, "sgd", lr=0.1)
    w_after = dict(model.named_parameters())["fc1.weight"].detach()
    # SGD math: w_after == w_before - lr * grad, exactly
    assert torch.allclose(w_after, w_before - 0.1 * g, atol=1e-7)
    d = result["param_diffs"]["fc1.weight"]
    assert abs(d["update_norm"] - (0.1 * g).norm().item()) < 1e-5
    assert d["grad_norm"] > 0

    undo_step(session)
    w_restored = dict(model.named_parameters())["fc1.weight"].detach()
    assert torch.equal(w_restored, w_before)
    print("  optimizer: SGD step == w - lr*grad exactly, diff norms correct, undo OK")


def test_adam_vs_sgd_update_shape():
    session, _ = make_session("mlp")
    run_backward(session, {"kind": "tensor", "shape": [1, 1, 28, 28]},
                 {"kind": "argmax"})
    result = optimizer_step(session, "adam", lr=0.001)
    diffs = result["param_diffs"]
    # first Adam step is ~lr * sign(grad): max_abs_change ≈ lr for every param
    for name, d in diffs.items():
        assert d["max_abs_change"] <= 0.001 + 1e-6, name
        assert d["max_abs_change"] > 0.0009, name
    print("  Adam: first-step normalization visible (all updates ~= lr) OK")


def test_backward_on_lm_target_token():
    session, _ = make_session("tiny_transformer")
    trace, result = run_backward(
        session, {"kind": "text", "text": "hello worl"},
        {"kind": "token", "text": "d"},
    )
    assert result["loss_desc"]["target_label"] == "d"
    assert "tok_emb.weight" in result["param_grads"]
    norms = result["layer_grad_norms"]
    assert len(norms) > 10
    print(f"  LM backward: CE on next-token 'd', {len(norms)} layers with grad norms OK")


def test_theory_lookup():
    assert get_theory("Linear")["key"] == "Linear"
    assert get_theory("Conv1D")["key"] == "Linear"        # GPT-2 alias
    assert get_theory("LlamaRMSNorm")["key"] == "LayerNorm"
    assert get_theory("SomeExoticLayer")["key"] == "_generic"
    assert "chain rule" in get_theory("_backprop")["what"] or \
           "chain rule" in get_theory("_backprop")["formula"]
    print("  theory: direct, alias, substring, and generic fallback OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} execution tests:")
    for t in tests:
        t()
    print("ALL EXECUTION TESTS PASSED")
