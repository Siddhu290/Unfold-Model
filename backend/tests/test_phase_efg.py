"""Phase E (analysis), F (comparative), G (profiling) tests."""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.analysis import dead_neurons, prune_sim, quantize_sim, weight_svd
from xray.comparative import ablate_heads, diff_against, diff_against_detail, max_activating
from xray.execution import make_input, run_forward
from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.profiling import profile
from xray.session import Session
from sample_models import build_demo, MNISTClassifierCNN


def make_session(name):
    model, x, meta = build_demo(name)
    load = LoadResult(kind="module", model=model, source=f"demo:{name}")
    s = Session(load, extract_architecture(model, x), meta)
    s.last_input_spec = ({"kind": "text", "text": "hello world"}
                         if name == "tiny_transformer"
                         else {"kind": "tensor", "shape": [1, 1, 28, 28]})
    ex, kw, desc = make_input(s, s.last_input_spec)
    tr = run_forward(s.model, ex, kw)
    tr.input_desc = desc
    s.last_trace = tr
    return s


def test_svd_rank_detection():
    s = make_session("mlp")
    r = weight_svd(s, "fc1", "weight")
    assert r["full_rank"] == 128 and len(r["singular_values"]) == 128
    assert 0 < r["effective_rank"] <= 128
    # plant a GENUINELY low-rank matrix and confirm detection
    with torch.no_grad():
        u = torch.randn(128, 8)
        v = torch.randn(8, 784)
        dict(s.model.named_parameters())["fc1.weight"].copy_(u @ v)
    r2 = weight_svd(s, "fc1", "weight")
    assert r2["rank_1pct"] <= 8, f"rank-8 matrix detected as {r2['rank_1pct']}"
    assert r2["effective_rank"] < 12
    print(f"  SVD: random eff.rank={r['effective_rank']:.1f}, planted rank-8 "
          f"detected as {r2['rank_1pct']} OK")


def test_dead_neurons_detects_planted_dead_unit():
    s = make_session("mlp")
    # kill unit 5 of fc1: negative bias + zero weights -> ReLU never fires
    with torch.no_grad():
        p = dict(s.model.named_parameters())
        p["fc1.weight"][5].zero_()
        p["fc1.bias"][5] = -10.0
    r = dead_neurons(s, "act1", n_inputs=16)
    assert r["total_units"] == 128
    assert 5 in r["dead_indices"], f"planted dead unit missed: {r['dead_indices']}"
    print(f"  dead neurons: planted unit found, {r['dead_count']}/128 dead OK")


def test_quantize_sim_restores_weights():
    s = make_session("cnn")
    w_before = dict(s.model.named_parameters())["conv3.weight"].detach().clone()
    r8 = quantize_sim(s, "conv3", bits=8)
    r2 = quantize_sim(s, "conv3", bits=2)
    w_after = dict(s.model.named_parameters())["conv3.weight"].detach()
    assert torch.equal(w_before, w_after), "weights must be restored"
    assert r2["kl_divergence"] > r8["kl_divergence"], \
        f"int2 must drift more than int8: {r2['kl_divergence']} vs {r8['kl_divergence']}"
    print(f"  quantize: KL int8={r8['kl_divergence']:.5f} < int2={r2['kl_divergence']:.5f}, restored OK")


def test_prune_curve_monotonic_degradation():
    s = make_session("mlp")
    w_before = dict(s.model.named_parameters())["fc1.weight"].detach().clone()
    r = prune_sim(s, "fc1")
    w_after = dict(s.model.named_parameters())["fc1.weight"].detach()
    assert torch.equal(w_before, w_after)
    kls = [c["kl_divergence"] for c in r["curve"]]
    assert len(kls) == 5 and kls[-1] > kls[0], f"pruning 90% must hurt more than 10%: {kls}"
    print(f"  prune: KL curve {['%.4f' % k for k in kls]}, restored OK")


def test_head_ablation():
    s = make_session("tiny_transformer")
    r = ablate_heads(s, {"kind": "text", "text": "hello world"}, "blocks.0.attn")
    assert r["n_heads"] == 4 and len(r["heads"]) == 4
    deltas = [h["delta"] for h in r["heads"]]
    assert deltas == sorted(deltas, reverse=True), "must be ranked by impact"
    assert any(abs(d) > 1e-9 for d in deltas), "ablating heads must change something"
    print(f"  head ablation: 4 heads ranked, deltas={['%.4f' % d for d in deltas]} OK")


def test_max_activating():
    s = make_session("tiny_transformer")
    r = max_activating(s, "blocks.0.mlp.1")   # the GELU
    assert len(r["results"]) >= 5
    scores = [x["score"] for x in r["results"]]
    assert scores == sorted(scores, reverse=True)
    r2 = max_activating(s, "blocks.0.mlp.1", neuron=7)
    assert len(r2["results"]) >= 5
    print(f"  max-activating: ranked, top={r['results'][0]['input']!r} OK")


def test_model_diff():
    s = make_session("cnn")
    # a second checkpoint = same arch, one layer perturbed
    other = MNISTClassifierCNN()
    other.load_state_dict(s.model.state_dict())
    with torch.no_grad():
        other.conv2.weight.add_(0.5)
    with tempfile.TemporaryDirectory() as d:
        pt = os.path.join(d, "other.pt")
        torch.save(other.state_dict(), pt)
        r = diff_against(s, pt)
    assert r["n_params_compared"] > 0
    assert r["param_diffs"]["conv2.weight"]["update_norm"] > 1.0
    identical = [k for k, v in r["param_diffs"].items() if v["update_norm"] == 0]
    assert "conv1.weight" in identical, "untouched layers must diff to zero"
    detail = diff_against_detail(s, "conv2.weight")
    assert abs(detail["delta"]["stats"]["mean"] - 0.5) < 1e-5, \
        "delta heatmap must show the +0.5 shift"
    print("  model diff: perturbed layer flagged, identical layers zero, "
          "delta mean == +0.5 exactly OK")


def test_profile_real_timings():
    s = make_session("cnn")
    r = profile(s)
    assert r["total_ms"] > 0
    rows = {x["path"]: x for x in r["rows"]}
    assert "conv1" in rows and rows["conv1"]["ms"] > 0
    assert rows["relu"]["calls"] == 3, "reused ReLU must aggregate 3 calls"
    # FLOPs from real shapes: conv1 = 2 * out_elems(1*16*28*28) * (1*3*3)
    assert rows["conv1"]["flops"] == 2 * 16 * 28 * 28 * 9
    assert rows["conv1"]["param_bytes"] == rows["conv1"]["params"] * 4
    leaf_sum = sum(x["ms"] for x in r["rows"] if x["is_leaf"])
    assert leaf_sum <= r["total_ms"] * 1.5
    print(f"  profile: total={r['total_ms']:.2f}ms, relu×3 aggregated, "
          f"conv1 FLOPs exact OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase E/F/G tests:")
    for t in tests:
        t()
    print("ALL PHASE E/F/G TESTS PASSED")
