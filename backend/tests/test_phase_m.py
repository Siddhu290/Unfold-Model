"""Phase M tests: SAE mechanics + planted-ground-truth recovery."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.sae import SparseAutoencoder, decompose, train_sae_stream
from xray.session import Session
from sample_models import build_demo


def make_session(name):
    model, x, meta = build_demo(name)
    load = LoadResult(kind="module", model=model, source=f"demo:{name}")
    return Session(load, extract_architecture(model, x), meta)


def test_synthetic_dictionary_recovery():
    """THE acid test: data generated from K known sparse directions — the
    SAE must recover each direction as a decoder column (cosine > 0.9)."""
    torch.manual_seed(0)
    d, K, N = 32, 6, 4000
    true_dirs = F.normalize(torch.randn(K, d), dim=1)
    coeffs = torch.rand(N, K) * (torch.rand(N, K) < 0.25).float() * 3.0
    X = coeffs @ true_dirs + 0.01 * torch.randn(N, d)

    # tuned recipe (matches sae.py's pipeline): center-only + strong L1 +
    # moderate expansion — per-dim standardization shears the geometry, and
    # weak L1 causes feature splitting; both empirically break recovery
    sae = SparseAutoencoder(d, 2 * d)
    opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
    Xn = X - X.mean(0)
    gen = torch.Generator().manual_seed(0)
    for _ in range(3000):
        xb = Xn[torch.randint(0, N, (256,), generator=gen)]
        xhat, c = sae(xb)
        loss = F.mse_loss(xhat, xb) + 1e-2 * c.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm_decoder()

    W = F.normalize(sae.dec.weight.data, dim=0)          # (d, m)
    cos_best = []
    for k in range(K):
        t = F.normalize(true_dirs[k], dim=0)
        cos_best.append(float((t @ W).abs().max()))
    assert min(cos_best) > 0.9, f"recovery failed: cosines {cos_best}"
    print(f"  synthetic recovery: all {K} planted directions found, "
          f"min cosine {min(cos_best):.3f} OK")


def test_end_to_end_quadrant_features():
    """SAE on the MLP's first hidden layer over the toy quadrant dataset must
    learn quadrant-selective features (the true generating factor)."""
    s = make_session("mlp")
    torch.manual_seed(0)
    events = list(train_sae_stream(s, "act1", {"n": 128}, expansion=2,
                                   l1=3e-3, steps=800))
    assert events[0]["event"] == "start"
    done = events[-1]
    assert done["event"] == "done" and done["alive"] > 4
    # mean feature activation per quadrant from the cached codes
    st = s.sae
    C = st["C"]
    quad = torch.tensor([int(l.split("-")[1].split(" ")[0]) for l in
                         [st["labels"][p["input"]] for p in st["prov"]]])
    top_per_quadrant = []
    for q in range(4):
        mean_on = C[quad == q].mean(0)
        f = int(mean_on.argmax())
        on = float(mean_on[f])
        off = float(C[quad != q][:, f].mean())
        top_per_quadrant.append((f, on / max(off, 1e-6)))
    distinct = len(set(f for f, _ in top_per_quadrant))
    sel = [r for _, r in top_per_quadrant]
    assert distinct >= 3, f"quadrant features collapsed: {top_per_quadrant}"
    assert sum(r > 2 for r in sel) >= 3, f"selectivity too weak: {sel}"
    print(f"  quadrant recovery: {distinct}/4 distinct top features, "
          f"selectivity ratios {['%.1f' % r for r in sel]} OK")


def test_decompose_and_provenance():
    s = make_session("tiny_transformer")
    torch.manual_seed(0)
    list(train_sae_stream(s, "blocks.2", {"texts": [
        "aaaa bbbb", "cccc dddd", "hello world", "xyz xyz xyz"]},
        expansion=2, steps=200))
    r = decompose(s, {"kind": "text", "text": "hello world"})
    assert r["l0"] >= 1 and r["l0"] < r["total_features"], \
        "decomposition must be sparse but non-empty"
    assert r["active"][0]["strength"] > 0
    assert all("label" in ex for f in r["active"] for ex in f["examples"])
    print(f"  decompose: L0={r['l0']}/{r['total_features']}, "
          f"recon R²={r['recon_r2']:.2f}, provenance attached OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase M tests:")
    for t in tests:
        t()
    print("ALL PHASE M TESTS PASSED")
