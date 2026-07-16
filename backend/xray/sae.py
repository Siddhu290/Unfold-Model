"""Phase M: toy sparse autoencoder over one layer's activation stream.

The real dictionary-learning technique at honest toy scale: an overcomplete
UNTIED linear autoencoder (encoder d→m, decoder m→d with its own weights,
decoder columns renormalized to unit norm every step) trained with
reconstruction + L1 sparsity on activations captured through the existing
forward pipeline. It demonstrates the mechanics; it does not claim
publication-grade monosemantic features.

    c    = ReLU(W_e (x − b_dec) + b_e)
    x̂    = W_d c + b_dec
    loss = ‖x − x̂‖² + λ‖c‖₁
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn

from .execution import make_input

DEFAULT_PROMPTS = [
    "The Eiffel Tower is in Paris", "Once upon a time there was a cat",
    "def main(): return 0", "2 + 2 = 4 and 3 + 3 = 6",
    "The weather today is cold and rainy", "I love this movie so much",
    "I hate waiting in long lines", "The capital of Italy is Rome",
    "She sells sea shells by the shore", "The stock market fell sharply",
    "import numpy as np", "Happy birthday to you",
]


class SparseAutoencoder(nn.Module):
    def __init__(self, d: int, m: int):
        super().__init__()
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d)
        with torch.no_grad():   # standard init: decoder ~ unit columns
            self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)
            self.enc.weight.data = self.dec.weight.data.t().clone()

    def encode(self, x):
        return F.relu(self.enc(x - self.dec.bias))

    def forward(self, x):
        c = self.encode(x)
        return self.dec(c), c

    @torch.no_grad()
    def renorm_decoder(self):
        self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0, eps=1e-8)


def collect_activations(session, layer_path: str, source: dict = None,
                        max_rows: int = 8192):
    """Capture one layer's outputs over a batch of real inputs, with
    provenance (which input / which token) per activation row."""
    model = session.model
    mod = dict(model.named_modules()).get(layer_path)
    if mod is None:
        raise ValueError(f"No module at {layer_path!r}")
    is_text = session.tokenizer is not None or hasattr(model, "encode")

    if is_text:
        texts = (source or {}).get("texts") or DEFAULT_PROMPTS
        specs = [{"kind": "text", "text": t} for t in texts]
        labels = texts
    else:
        n = min(int((source or {}).get("n", 64)), 256)
        shape = session.meta.get("input_shape")
        if not shape:
            raise ValueError("No input shape known.")
        from .training import _quadrant_batch
        gen = torch.Generator().manual_seed(3)
        x, y = _quadrant_batch(shape, n, gen)
        specs = None
        labels = [f"quadrant-{int(q)} sample {i}" for i, q in enumerate(y)]

    rows, prov = [], []
    grabbed = []
    h = mod.register_forward_hook(lambda m, a, o: grabbed.append(o.detach()))
    try:
        with torch.no_grad():
            if is_text:
                for i, spec in enumerate(specs):
                    grabbed.clear()
                    ex, kw, desc = make_input(session, spec)
                    model(**kw) if kw else model(ex)
                    out = grabbed[0].float()
                    if out.dim() != 3:
                        raise ValueError(f"{layer_path} output is not "
                                         "(batch, seq, d) — pick a block.")
                    toks = desc.get("tokens") or []
                    for pos in range(out.shape[1]):
                        rows.append(out[0, pos])
                        prov.append({"input": i, "pos": pos,
                                     "token": toks[pos] if pos < len(toks) else "?"})
            else:
                model(x)
                out = grabbed[0].float()
                flat = out.reshape(out.shape[0], -1) if out.dim() > 2 else out
                for i in range(flat.shape[0]):
                    rows.append(flat[i])
                    prov.append({"input": i, "pos": None, "token": None})
    finally:
        h.remove()
    X = torch.stack(rows)[:max_rows]
    prov = prov[:max_rows]
    return X, prov, labels


def train_sae_stream(session, layer_path: str, source: dict = None,
                     expansion: int = 2, l1: float = 5e-3, steps: int = 500,
                     lr: float = 1e-3, batch: int = 256, seed: int = 0):
    """Stream a real SAE training run (same NDJSON pattern as Phase I)."""
    X, prov, labels = collect_activations(session, layer_path, source)
    d = X.shape[1]
    m = max(8, int(expansion) * d)
    if m > 8192:
        raise ValueError(f"Dictionary of {m} features is beyond toy scale "
                         f"here; lower the expansion (layer d={d}).")
    steps = max(10, min(int(steps), 2000))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    sae = SparseAutoencoder(d, m)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    # center only: per-dimension standardization would shear the geometry
    # and distort the very directions dictionary learning must recover
    mu = X.mean(0)
    Xn = X - mu

    t0 = time.perf_counter()
    yield {"event": "start", "layer": layer_path, "rows": X.shape[0],
           "d": d, "features": m, "steps": steps, "l1": l1,
           "note": "toy-scale dictionary learning — demonstrates the real "
                   "technique's mechanics, not publication-grade features"}

    for i in range(1, steps + 1):
        idx = torch.randint(0, Xn.shape[0], (min(batch, Xn.shape[0]),),
                            generator=gen)
        xb = Xn[idx]
        xhat, c = sae(xb)
        recon = F.mse_loss(xhat, xb)
        sparsity = c.abs().mean()
        loss = recon + l1 * sparsity
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.renorm_decoder()
        if i % max(1, steps // 100) == 0 or i == steps:
            l0 = float((c > 1e-6).float().sum(-1).mean())
            yield {"event": "step", "i": i, "loss": float(loss),
                   "recon": float(recon), "l0": l0}

    # feature statistics + max-activating provenance over the training rows
    with torch.no_grad():
        C = sae.encode(Xn)                          # (N, m)
    freq = (C > 1e-6).float().mean(0)
    alive = int((freq > 0).sum())
    top_examples = {}
    strengths, order = C.max(dim=0)
    active_feats = (strengths > 1e-6).nonzero().flatten().tolist()
    for f in active_feats:
        vals, idxs = torch.topk(C[:, f], min(8, C.shape[0]))
        top_examples[f] = [
            {"strength": float(v), **prov[int(j)],
             "label": labels[prov[int(j)]["input"]]}
            for v, j in zip(vals, idxs) if v > 1e-6]

    session.sae = {
        "sae": sae, "layer_path": layer_path, "mu": mu,
        "labels": labels, "prov": prov, "C": C,
        "top_examples": top_examples,
        "meta": {"d": d, "features": m, "rows": X.shape[0], "l1": l1,
                 "steps": steps, "alive": alive,
                 "mean_l0": float((C > 1e-6).float().sum(-1).mean())},
    }
    yield {"event": "done", "elapsed_s": round(time.perf_counter() - t0, 1),
           **session.sae["meta"]}


def decompose(session, input_spec: dict, position: str = "last",
              k: int = 12) -> dict:
    """Sparse feature decomposition of the current input's activation."""
    st = getattr(session, "sae", None)
    if st is None:
        raise ValueError("Train an SAE on a layer first.")
    model = session.model
    mod = dict(model.named_modules())[st["layer_path"]]
    grabbed = []
    h = mod.register_forward_hook(lambda m, a, o: grabbed.append(o.detach()))
    try:
        ex, kw, desc = make_input(session, input_spec)
        with torch.no_grad():
            model(**kw) if kw else model(ex)
    finally:
        h.remove()
    out = grabbed[0].float()
    if out.dim() == 3:
        vec = out[0, -1] if position == "last" else out[0].mean(0)
    else:
        vec = out.reshape(out.shape[0], -1)[0]
    xn = vec - st["mu"]
    sae = st["sae"]
    with torch.no_grad():
        c = sae.encode(xn.unsqueeze(0))[0]
        xhat = sae.dec(c) + 0 * xn      # dec includes bias
        recon_r2 = 1.0 - float(F.mse_loss(sae(xn.unsqueeze(0))[0][0], xn)
                               / xn.var().clamp_min(1e-9))
    vals, idxs = torch.topk(c, min(k, c.numel()))
    feats = []
    for v, f in zip(vals, idxs):
        if v <= 1e-6:
            continue
        f = int(f)
        feats.append({"feature": f, "strength": float(v),
                      "examples": st["top_examples"].get(f, [])[:5]})
    return {"layer": st["layer_path"], "position": position,
            "input": desc.get("text") or "tensor input",
            "l0": int((c > 1e-6).sum()), "total_features": c.numel(),
            "recon_r2": recon_r2, "active": feats,
            "meta": st["meta"]}
