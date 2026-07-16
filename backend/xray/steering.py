"""Phase L: activation steering with contrastive concept vectors.

A direction is the difference between two prompts' residual-stream
activations at a chosen layer (the exact tensors the patching pipeline
already captures). Steering adds α × direction to that layer's output
during a real forward on an UNRELATED prompt — Phase C's override mechanism
with addition instead of replacement. Generalization to unrelated prompts is
what separates a concept direction from a prompt-specific artifact.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .execution import make_input, run_forward
from .summarize import first_tensor


def _layer_first_call(trace, layer_path: str):
    for rec in trace.records:
        if rec["path"] == layer_path:
            return rec["call_index"]
    raise ValueError(f"Layer {layer_path!r} did not fire in the forward pass.")


def _act_at(session, prompt: str, layer_path: str, position: str):
    ex, kw, desc = make_input(session, {"kind": "text", "text": prompt})
    trace = run_forward(session.model, ex, kw)
    ci = _layer_first_call(trace, layer_path)
    t = trace.tensors.get(ci)
    if t is None or t.dim() != 3:
        raise ValueError(f"{layer_path} has no retained (batch, seq, d) "
                         "activation — pick a residual-stream layer (a block).")
    vec = t[0, -1] if position == "last" else t[0].mean(dim=0)
    return vec, ci, desc


def build_direction(session, prompt_a: str, prompt_b: str, layer_path: str,
                    position: str = "last") -> dict:
    """direction = act(prompt_a) − act(prompt_b) at layer_path."""
    if not session.runnable:
        raise ValueError("State-dict-only session cannot steer.")
    va, ci_a, _ = _act_at(session, prompt_a, layer_path, position)
    vb, ci_b, _ = _act_at(session, prompt_b, layer_path, position)
    if ci_a != ci_b:
        raise ValueError("Layer call index differs between prompts — "
                         "unexpected model structure.")
    direction = (va - vb).detach()
    session.steering = {
        "layer_path": layer_path, "call_index": ci_a, "position": position,
        "prompt_a": prompt_a, "prompt_b": prompt_b,
        "direction": direction,
    }
    return {
        "layer_path": layer_path, "position": position,
        "dim": direction.numel(),
        "norm": float(direction.norm()),
        "prompt_a": prompt_a, "prompt_b": prompt_b,
        "note": "direction = activation(A) − activation(B) at the last token "
                "of this layer's residual stream" if position == "last" else
                "direction = mean-over-tokens activation difference",
    }


def _decode(session, tid):
    from .generation import _decode_one
    return _decode_one(session, int(tid))


def _steer_probs(session, input_spec: dict, alpha: float,
                 positions: str = "all"):
    st = getattr(session, "steering", None)
    if st is None:
        raise ValueError("Build a steering direction first.")
    ex, kw, desc = make_input(session, input_spec)
    d = st["direction"]

    def add_direction(t):
        if t is None or t.dim() != 3 or t.shape[-1] != d.numel():
            raise ValueError("Steered layer produced an unexpected shape.")
        out = t.clone()
        if positions == "last":
            out[:, -1, :] += alpha * d.to(t.dtype)
        else:
            out += alpha * d.to(t.dtype)
        return out

    overrides = {st["call_index"]: add_direction} if alpha != 0 else {}
    trace = run_forward(session.model, ex, kw, overrides=overrides, light=True)
    logits = first_tensor(trace.output)
    last = logits[0, -1] if logits.dim() == 3 else logits[0]
    return F.softmax(last.float(), dim=-1), desc


def steer(session, input_spec: dict, alpha: float, k: int = 5,
          positions: str = "all", watch: list = None) -> dict:
    """One steered forward + the unsteered baseline for comparison."""
    base, desc = _steer_probs(session, input_spec, 0.0, positions)
    probs, _ = _steer_probs(session, input_spec, alpha, positions)
    kk = min(k, probs.numel())
    top = torch.topk(probs, kk)
    out = {
        "alpha": alpha, "positions": positions,
        "prompt": desc.get("text"),
        "topk": [{"id": int(i), "label": _decode(session, i), "prob": float(p),
                  "prob_base": float(base[i])}
                 for p, i in zip(top.values, top.indices)],
        "top1_base": _decode(session, int(base.argmax())),
        "top1_steered": _decode(session, int(probs.argmax())),
        "top1_changed": int(base.argmax()) != int(probs.argmax()),
        "kl_from_base": float(F.kl_div(probs.clamp_min(1e-12).log(), base,
                                       reduction="sum")),
    }
    if watch:
        rows = []
        for text in watch:
            try:
                from .attribution import _token_id
                tid = _token_id(session, text)
                rows.append({"token": text, "id": tid,
                             "p_base": float(base[tid]),
                             "p_steered": float(probs[tid]),
                             "delta": float(probs[tid] - base[tid])})
            except Exception:
                continue
        out["watch"] = rows
    return out


def steer_batch(session, prompts: list, alpha: float, k: int = 3,
                positions: str = "all", watch: list = None) -> dict:
    """The generalization test: the same direction applied to several
    unrelated prompts. A real concept direction shifts them all the same way."""
    results = [steer(session, {"kind": "text", "text": p}, alpha, k,
                     positions, watch) for p in prompts[:12]]
    log = getattr(session, "steering_log", None)
    if log is None:
        log = session.steering_log = []
    st = session.steering
    log.append({
        "layer": st["layer_path"], "alpha": alpha,
        "prompt_a": st["prompt_a"], "prompt_b": st["prompt_b"],
        "norm": float(st["direction"].norm()),
        "results": [{"prompt": r["prompt"], "top1_base": r["top1_base"],
                     "top1_steered": r["top1_steered"],
                     "kl": r["kl_from_base"],
                     "watch": r.get("watch")} for r in results],
    })
    del log[:-10]
    return {"alpha": alpha, "results": results}
