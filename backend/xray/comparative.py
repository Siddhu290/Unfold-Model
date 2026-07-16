"""Phase F: attention head ablation, max-activating examples, model diffing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .editing import _label
from .execution import make_input, run_forward
from .summarize import first_tensor, summarize_tensor


def _attn_probs_call(trace, layer_path: str):
    """Find the (B, H, T, T) attention-probability carrier inside a layer —
    the module whose retained output holds per-head attention weights."""
    for rec in trace.records:
        if not rec["path"].startswith(layer_path):
            continue
        s = rec.get("out_shape")
        if (s and len(s) == 4 and s[-1] == s[-2]
                and rec["call_index"] in trace.tensors):
            return rec["call_index"], trace.tensors[rec["call_index"]]
    raise ValueError(
        f"No retained (batch, heads, seq, seq) attention tensor found under "
        f"{layer_path!r} — run a forward pass first, and pick an attention layer.")


def ablate_heads(session, input_spec: dict, layer_path: str) -> dict:
    """Zero one attention head at a time (a real patched re-run per head) and
    rank heads by how much the output degrades."""
    model = session.model
    example, kwargs, _ = make_input(session, input_spec)
    base_trace = run_forward(model, example, kwargs)
    ci, attn = _attn_probs_call(base_trace, layer_path)
    n_heads = attn.shape[1]

    base_logits = first_tensor(base_trace.output)
    base_last = base_logits[0, -1] if base_logits.dim() == 3 else base_logits[0]
    base_probs = F.softmax(base_last.float(), dim=-1)
    top1 = int(base_probs.argmax())
    p_base = float(base_probs[top1])

    heads = []
    for h in range(n_heads):
        patched = attn.clone()
        patched[:, h] = 0.0
        tr = run_forward(model, example, kwargs, overrides={ci: patched}, light=True)
        logits = first_tensor(tr.output)
        last = logits[0, -1] if logits.dim() == 3 else logits[0]
        probs = F.softmax(last.float(), dim=-1)
        heads.append({
            "head": h,
            "p_top1": float(probs[top1]),
            "delta": p_base - float(probs[top1]),
            "top1_changed": int(probs.argmax()) != top1,
        })
    heads.sort(key=lambda x: -x["delta"])
    return {"layer": layer_path, "call_index": ci, "n_heads": n_heads,
            "baseline_top1": _label(session, top1), "p_top1_baseline": p_base,
            "heads": heads}


def max_activating(session, path: str, texts: list = None,
                   n_random: int = 16, neuron: int = None) -> dict:
    """Rank candidate inputs by how strongly they drive one layer (or one
    specific unit of it)."""
    model = session.model
    mod = dict(model.named_modules()).get(path)
    if mod is None:
        raise ValueError(f"No module at {path!r}")

    is_text = session.tokenizer is not None or hasattr(model, "encode")
    if is_text:
        candidates = [{"kind": "text", "text": t} for t in (texts or [
            "Hello world", "The Eiffel Tower is in Paris",
            "def main():", "2 + 2 = 4", "Once upon a time",
            "!!!???", "The cat sat on the mat", "E = mc^2",
        ])]
    else:
        shape = session.meta.get("input_shape")
        if not shape:
            raise ValueError("No input shape known.")
        candidates = [{"kind": "tensor", "shape": shape, "seed": i,
                       "fill": "randn"} for i in range(n_random)]

    grabbed = []
    handle = mod.register_forward_hook(lambda m, a, o: grabbed.append(o.detach()))
    results = []
    try:
        for spec in candidates:
            grabbed.clear()
            ex, kw, desc = make_input(session, spec)
            with torch.no_grad():
                model(**kw) if kw else model(ex)
            if not grabbed:
                continue
            out = grabbed[0].float()
            if neuron is not None:
                unit = (out[..., neuron] if out.dim() != 4
                        else out[:, neuron])
                score = float(unit.abs().max())
            else:
                score = float(out.abs().max())
            results.append({
                "input": desc.get("text") or f"randn(seed={spec.get('seed')})",
                "score": score,
                "mean_abs": float(out.abs().mean()),
            })
    finally:
        handle.remove()
    results.sort(key=lambda r: -r["score"])
    return {"path": path, "neuron": neuron, "metric": "max |activation|",
            "results": results}


def diff_against(session, other_ref: str, allow_pickle: bool = False) -> dict:
    """Load a second checkpoint of the same architecture and diff every
    weight tensor against the currently loaded model."""
    from .loading import load_any

    other = load_any(other_ref, allow_pickle=allow_pickle)
    if other.kind == "state_dict":
        other_sd = other.state_dict
    else:
        other_sd = {k: v.detach() for k, v in other.model.state_dict().items()}
    result = diff_state_dicts(session, other_sd, other.source)
    result["warnings"] = other.warnings
    return result


def diff_state_dicts(session, other_sd: dict, source_name: str) -> dict:
    """Diff the live model against any state dict (uploaded checkpoint or an
    in-memory training checkpoint) — one implementation for both."""
    mine = dict(session.model.named_parameters())
    diffs, missing, shape_mismatch = {}, [], []
    for name, p in mine.items():
        o = other_sd.get(name)
        if o is None:
            missing.append(name)
            continue
        if list(o.shape) != list(p.shape):
            shape_mismatch.append({"name": name, "mine": list(p.shape),
                                   "other": list(o.shape)})
            continue
        delta = (o.float() - p.detach().float())
        pn = p.detach().norm().item()
        diffs[name] = {
            "shape": list(p.shape),
            "update_norm": float(delta.norm()),
            "weight_norm_before": pn,
            "weight_norm_after": float(o.float().norm()),
            "relative_update": float(delta.norm()) / pn if pn > 0 else None,
            "max_abs_change": float(delta.abs().max()),
            "grad_norm": None,
        }
    session.model_diff_sd = other_sd
    n_identical = sum(1 for d in diffs.values() if d["update_norm"] == 0)
    return {"other": source_name, "n_params_compared": len(diffs),
            "n_identical": n_identical, "missing_in_other": missing[:20],
            "shape_mismatch": shape_mismatch[:20], "param_diffs": diffs,
            "warnings": []}


def diff_against_detail(session, name: str) -> dict:
    """Before/current vs other-checkpoint heatmaps for one parameter —
    same shape as the optimizer-step diff payload, so the UI is reused."""
    sd = getattr(session, "model_diff_sd", None)
    if sd is None or name not in sd:
        raise ValueError("Run a model diff first (or unknown parameter).")
    mine = dict(session.model.named_parameters()).get(name)
    if mine is None:
        raise ValueError(f"Unknown parameter {name!r}")
    cur = mine.detach().cpu().float()
    oth = sd[name].detach().cpu().float()
    return {
        "name": name,
        "before": summarize_tensor(cur, name=f"{name} (this model)"),
        "after": summarize_tensor(oth, name=f"{name} (other checkpoint)"),
        "delta": summarize_tensor(oth - cur, name=f"{name} (other − this)"),
    }
