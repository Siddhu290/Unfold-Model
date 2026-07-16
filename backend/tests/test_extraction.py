"""Milestone 1 tests: loader + architecture extractor against real models."""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.extraction import extract_architecture, extract_from_state_dict, flatten_tree
from xray.loading import UnsafePickleError, load_any, load_torch_file
from xray.summarize import summarize_tensor
from sample_models import MNISTClassifierCNN, build_demo


def test_mlp_extraction():
    model, x, _ = build_demo("mlp")
    arch = extract_architecture(model, x)
    assert arch["mode"] == "module"
    assert arch["total_params"] == 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10
    nodes = flatten_tree(arch["tree"])
    fc1 = nodes["fc1"]
    assert fc1["class"] == "Linear"
    assert fc1["in_shape"] == [1, 784]
    assert fc1["out_shape"] == [1, 128]
    assert fc1["own_param_count"] == 784 * 128 + 128
    # execution order was captured
    assert nodes["flatten"]["call_order"] < nodes["fc1"]["call_order"] < nodes["fc3"]["call_order"]
    print("  MLP: params, shapes, call order OK")


def test_cnn_extraction_with_skip():
    model, x, _ = build_demo("cnn")
    arch = extract_architecture(model, x)
    nodes = flatten_tree(arch["tree"])
    assert nodes["conv1"]["out_shape"] == [1, 16, 28, 28]
    assert nodes["pool"]["out_shape"] == [1, 16, 14, 14]
    # 'relu' is called 3 times in forward() — n_calls must reflect real execution
    assert nodes["relu"]["n_calls"] == 3
    # nested Sequential head resolved with children
    assert nodes["head.2"]["class"] == "Linear"
    assert nodes["head.2"]["out_shape"] == [1, 10]
    print("  CNN: skip connection, module reuse (n_calls=3), nested Sequential OK")


def test_transformer_extraction_and_repeat_groups():
    model, x, _ = build_demo("tiny_transformer")
    arch = extract_architecture(model, x)
    nodes = flatten_tree(arch["tree"])
    groups = nodes["blocks"].get("repeat_groups")
    assert groups and groups[0]["count"] == 4, f"expected x4 repeat group, got {groups}"
    attn = nodes["blocks.0.attn"]
    assert attn["class"] == "CausalSelfAttention"
    # Q/K/V projections individually visible
    assert nodes["blocks.0.attn.q_proj"]["class"] == "Linear"
    # attention-prob tap captured a (B, H, T, T) shape
    assert nodes["blocks.0.attn.attn_probs"]["out_shape"] == [1, 4, x.shape[1], x.shape[1]]
    assert nodes["lm_head"]["out_shape"] == [1, x.shape[1], model.VOCAB]
    print(f"  Transformer: repeat group x{groups[0]['count']}, Q/K/V + attn-prob tap, lm_head shapes OK")


def test_safetensors_roundtrip():
    from safetensors.torch import save_file

    model = MNISTClassifierCNN()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cnn.safetensors")
        save_file(model.state_dict(), path)
        result = load_any(path)
        assert result.kind == "state_dict"
        arch = extract_from_state_dict(result.state_dict)
        assert arch["mode"] == "state_dict"
        assert arch["total_params"] == sum(p.numel() for p in model.parameters()) + \
            sum(b.numel() for n, b in model.named_buffers() if "running" in n or "batches" in n)
        nodes = flatten_tree(arch["tree"])
        assert any(p["name"] == "weight" and p["shape"] == [16, 1, 3, 3]
                   for p in nodes["conv1"]["params"])
    print("  safetensors: save -> safe load -> inferred tree OK")


def test_state_dict_pt_safe_load():
    model = MNISTClassifierCNN()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cnn.pt")
        torch.save(model.state_dict(), path)
        result = load_torch_file(path)  # no allow_pickle needed
        assert result.kind == "state_dict"
        assert not result.warnings
    print("  .pt state dict: weights_only safe load OK")


def test_pickled_module_requires_opt_in():
    model = MNISTClassifierCNN()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "full_model.pt")
        torch.save(model, path)
        try:
            load_torch_file(path)
            raise AssertionError("should have refused pickle without opt-in")
        except UnsafePickleError:
            pass
        result = load_torch_file(path, allow_pickle=True)
        assert result.kind == "module"
        assert result.warnings, "pickle load must carry a warning"
        arch = extract_architecture(result.model, torch.randn(1, 1, 28, 28))
        assert arch["total_params"] > 0
    print("  pickled full model: refused by default, loads with opt-in + warning OK")


def test_summarize_large_tensor():
    t = torch.randn(4096, 4096)
    s = summarize_tensor(t)
    assert "values" not in s
    assert s["heatmap"]["rows"] <= 64 and s["heatmap"]["cols"] <= 64
    assert len(s["preview"]) <= 128
    assert len(s["histogram"]["counts"]) == 40
    import json
    payload = json.dumps(s)
    assert len(payload) < 200_000, f"summary too large: {len(payload)} bytes"
    small = summarize_tensor(torch.arange(6.0))
    assert small["values"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"  summarize: 16.7M-element tensor -> {len(payload)//1024}KB payload OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} milestone-1 tests:")
    for t in tests:
        t()
    print("ALL MILESTONE 1 TESTS PASSED")
