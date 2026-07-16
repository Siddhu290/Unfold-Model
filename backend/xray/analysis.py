"""Phase E: weight & structure diagnostics on real weights + real re-runs.

Quantization and pruning simulations mutate the live weights in-memory,
re-run the last input, measure the drift, and ALWAYS restore the originals
(try/finally) — the reported degradation is measured, never estimated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .editing import _eval_probs, _default_input_spec, _label
from .summarize import _to_2d


def _spec_of(session):
    return getattr(session, "last_input_spec", None) or _default_input_spec(session)


def _module_own_params(session, path):
    mod = dict(session.model.named_modules()).get(path)
    if mod is None:
        raise ValueError(f"No module at {path!r}")
    params = list(mod.named_parameters(recurse=False))
    if not params:
        raise ValueError(f"{path} has no parameters of its own.")
    return mod, params


def weight_svd(session, path: str, param: str = "weight") -> dict:
    """Singular value spectrum + effective rank of a weight matrix."""
    t = session.get_param_tensor(path, param).detach().float()
    m = _to_2d(t)
    if min(m.shape) < 2:
        raise ValueError(f"{path}.{param} ({list(t.shape)}) is not a matrix.")
    approx = False
    with torch.no_grad():
        if min(m.shape) > 4096:
            q = min(512, min(m.shape))
            _, s, _ = torch.svd_lowrank(m, q=q)
            approx = True
        else:
            s = torch.linalg.svdvals(m)
    s = s.clamp_min(0)
    total = float((s ** 2).sum())
    p = (s ** 2) / max(total, 1e-30)
    entropy = float(-(p * (p + 1e-30).log()).sum())
    effective_rank = float(torch.exp(torch.tensor(entropy)))
    thresh_rank = int((s > 0.01 * s[0]).sum()) if s.numel() else 0
    cum = torch.cumsum(p, 0)
    r90 = int((cum < 0.90).sum()) + 1
    vals = s.tolist()
    if len(vals) > 256:   # downsample for the plot
        idx = torch.linspace(0, len(vals) - 1, 256, dtype=torch.float64).long()
        vals = [vals[i] for i in idx]
    return {
        "path": path, "param": param, "shape": list(t.shape),
        "matrix_shape": list(m.shape), "full_rank": min(m.shape),
        "approximate": approx,
        "effective_rank": effective_rank,      # exp(spectral entropy)
        "rank_1pct": thresh_rank,              # svals above 1% of max
        "rank_90pct_energy": r90,              # svals holding 90% of energy
        "singular_values": vals,
    }


def dead_neurons(session, path: str, n_inputs: int = 32) -> dict:
    """Units of an activation layer that NEVER fire across a real batch."""
    model = session.model
    mod = dict(model.named_modules()).get(path)
    if mod is None:
        raise ValueError(f"No module at {path!r}")
    captured = []
    h = mod.register_forward_hook(lambda m, a, o: captured.append(o.detach()))
    try:
        with torch.no_grad():
            if session.tokenizer is not None or hasattr(model, "encode"):
                prompts = ["Hello world", "The quick brown fox", "1 + 1 =",
                           "Once upon a time", "import numpy as np",
                           "El rapido zorro", "AAAAAA", "the the the the"]
                from .execution import make_input
                for text in prompts[:n_inputs]:
                    ex, kw, _ = make_input(session, {"kind": "text", "text": text})
                    model(**kw) if kw else model(ex)
            else:
                shape = session.meta.get("input_shape")
                if not shape:
                    raise ValueError("No input shape known for a batch probe.")
                gen = torch.Generator().manual_seed(7)
                x = torch.randn([n_inputs] + shape[1:], generator=gen)
                model(x)
    finally:
        h.remove()
    if not captured:
        raise ValueError(f"{path} did not fire during the probe.")
    # unit = channel (conv: dim 1) or feature (last dim otherwise)
    ever_active = None
    for out in captured:
        if out.dim() == 4:      # (B, C, H, W): unit = channel
            act = (out > 0).permute(1, 0, 2, 3).reshape(out.shape[1], -1).any(dim=1)
        else:                   # (B, ..., D): unit = feature
            act = (out > 0).reshape(-1, out.shape[-1]).any(dim=0)
        ever_active = act if ever_active is None else (ever_active | act)
    dead = (~ever_active).nonzero().flatten().tolist()
    return {
        "path": path, "n_inputs": n_inputs, "total_units": int(ever_active.numel()),
        "dead_count": len(dead), "dead_frac": len(dead) / ever_active.numel(),
        "dead_indices": dead[:64],
    }


def _quantize_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric round-to-nearest quantization."""
    qmax = 2 ** (bits - 1) - 1
    m = w.reshape(w.shape[0], -1)
    scale = (m.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)
    q = (m / scale).round().clamp(-qmax - 1, qmax) * scale
    return q.reshape(w.shape)


def quantize_sim(session, path: str, bits: int = 8) -> dict:
    """Quantize one layer's weights in-memory, re-run, measure drift, restore."""
    if bits not in (8, 4, 2):
        raise ValueError("bits must be 8, 4 or 2")
    mod, params = _module_own_params(session, path)
    spec = _spec_of(session)
    base = _eval_probs(session, session.model, spec)
    saved = [(p, p.detach().clone()) for _, p in params]
    try:
        with torch.no_grad():
            for _, p in params:
                if p.dim() >= 2:
                    p.copy_(_quantize_tensor(p, bits))
        quant = _eval_probs(session, session.model, spec)
    finally:
        with torch.no_grad():
            for p, orig in saved:
                p.copy_(orig)
    kl = float(F.kl_div(quant.clamp_min(1e-12).log(), base, reduction="sum"))
    b1, q1 = int(base.argmax()), int(quant.argmax())
    return {
        "path": path, "bits": bits, "kl_divergence": kl,
        "top1_before": _label(session, b1), "top1_after": _label(session, q1),
        "top1_changed": b1 != q1,
        "p_top1_before": float(base[b1]), "p_top1_after": float(quant[b1]),
        "max_prob_shift": float((quant - base).abs().max()),
    }


def prune_sim(session, path: str, fractions=None) -> dict:
    """Zero the smallest-magnitude X% of one layer's weights, re-run, measure
    degradation as X grows, restore."""
    fractions = fractions or [0.1, 0.25, 0.5, 0.75, 0.9]
    mod, params = _module_own_params(session, path)
    spec = _spec_of(session)
    base = _eval_probs(session, session.model, spec)
    b1 = int(base.argmax())
    saved = [(p, p.detach().clone()) for _, p in params]
    curve = []
    try:
        for frac in sorted(fractions):
            with torch.no_grad():
                for (_, p), (_, orig) in zip(params, saved):
                    if p.dim() < 2:
                        continue
                    flat = orig.abs().flatten()
                    k = int(flat.numel() * frac)
                    if k == 0:
                        continue
                    thresh = flat.kthvalue(k).values
                    p.copy_(torch.where(orig.abs() <= thresh,
                                        torch.zeros_like(orig), orig))
            pr = _eval_probs(session, session.model, spec)
            curve.append({
                "fraction": frac,
                "kl_divergence": float(F.kl_div(pr.clamp_min(1e-12).log(), base,
                                                reduction="sum")),
                "p_top1": float(pr[b1]),
                "top1_changed": int(pr.argmax()) != b1,
            })
    finally:
        with torch.no_grad():
            for p, orig in saved:
                p.copy_(orig)
    return {"path": path, "baseline_top1": _label(session, b1),
            "p_top1_baseline": float(base[b1]), "curve": curve}
