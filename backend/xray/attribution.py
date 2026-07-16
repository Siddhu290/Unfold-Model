"""Phase J: input attribution — which parts of the input drove the output.

Both methods run through the existing run_forward pipeline:
  - saliency: one forward with grad capture; ∂(target logit)/∂(embedding or
    pixel) read from the SAME grad hooks the Backward tab uses.
  - integrated gradients: K interpolated passes. For text the interpolation
    happens in embedding space via the Phase-C override mechanism (the
    embedding call's output is replaced by α·emb with requires_grad, and the
    gradient is read off that override tensor).
"""

from __future__ import annotations

import torch

from .execution import make_input, run_forward
from .summarize import first_tensor, summarize_tensor


def _token_id(session, text: str) -> int:
    if session.load_kind == "hf":
        ids = session.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"{text!r} produced no tokens.")
        return ids[0]
    if hasattr(session.model, "encode"):
        return int(session.model.encode(text)[0, 0].item())
    raise ValueError("No tokenizer to resolve a token.")


def _target_logit(session, logits: torch.Tensor, target_spec: dict,
                  contrast_id: int = None):
    """The scalar being attributed. With a contrast token the score is
    logit(target) − logit(contrast): 'why THIS answer rather than that one' —
    which suppresses attribution to tokens that merely set up the answer's
    grammatical category."""
    last = logits[0, -1] if logits.dim() == 3 else logits[0]
    spec = target_spec or {}
    if spec.get("id") is not None:
        tid = int(spec["id"])
    elif spec.get("text"):
        tid = _token_id(session, spec["text"])
    else:
        tid = int(last.argmax())
    score = last[tid] - last[contrast_id] if contrast_id is not None else last[tid]
    return score, tid


def _embedding_call(trace, seq_len: int):
    """First module call whose output is the (1, T, d) token representation."""
    for rec in trace.records:
        s = rec.get("out_shape")
        if s and len(s) == 3 and s[0] == 1 and s[1] == seq_len \
                and rec["call_index"] in trace.tensors:
            return rec["call_index"], rec["path"]
    raise ValueError("No (1, seq, d) embedding activation found — is this a "
                     "text model?")


def _label_of(session, tid):
    from .generation import _decode_one
    return _decode_one(session, tid)


def _token_rows(tokens, scores):
    total = sum(abs(s) for s in scores) or 1.0
    return [{"pos": i, "token": t, "score": float(s),
             "frac": float(abs(s) / total)}
            for i, (t, s) in enumerate(zip(tokens, scores))]


def attribute(session, input_spec: dict, target_spec: dict = None,
              method: str = "saliency", steps: int = 16,
              contrast: str = None) -> dict:
    if not session.runnable:
        raise ValueError("State-dict-only session cannot run attribution.")
    example, kwargs, desc = make_input(session, input_spec)
    is_text = desc.get("kind") == "text"
    contrast_id = _token_id(session, contrast) if (contrast and is_text) else None

    if is_text:
        return _attribute_text(session, example, kwargs, desc, target_spec,
                               method, steps, contrast_id)
    return _attribute_vision(session, example, desc, target_spec, method, steps)


def _baseline_embedding(session, emb: torch.Tensor):
    """IG baseline: the pad/eos token's embedding when one exists — a zero
    vector is far outside the LayerNorm-normalized manifold and wrecks the
    path integral on real LMs."""
    tok = session.tokenizer
    base_id = None
    if tok is not None:
        base_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if base_id is None:
        return torch.zeros_like(emb), "zeros"
    wte = session.model.get_input_embeddings() if hasattr(
        session.model, "get_input_embeddings") else None
    if wte is None:
        return torch.zeros_like(emb), "zeros"
    with torch.no_grad():
        v = wte(torch.tensor([base_id]))
    return v.expand_as(emb).clone(), f"token {base_id} embedding"


def _attribute_text(session, example, kwargs, desc, target_spec, method,
                    steps, contrast_id=None):
    model = session.model
    tokens = desc["tokens"]
    T = len(tokens)

    # baseline pass with grad capture — the same hooks the Backward tab uses
    trace = run_forward(model, example, kwargs, capture_grads=True)
    logits = first_tensor(trace.output)
    logit, tid = _target_logit(session, logits, target_spec, contrast_id)
    model.zero_grad(set_to_none=True)
    logit.backward()
    emb_ci, emb_path = _embedding_call(trace, T)
    emb = trace.tensors[emb_ci]                    # (1, T, d)
    grad = trace.grad_tensors.get(emb_ci)
    if grad is None:
        raise ValueError(f"No gradient captured at {emb_path}.")
    model.zero_grad(set_to_none=True)

    result = {
        "kind": "text", "method": method, "embedding_layer": emb_path,
        "target": {"id": tid, "label": _label_of(session, tid),
                   "logit": float(logit.detach())},
        "contrast": _label_of(session, contrast_id) if contrast_id is not None else None,
        "tokens": tokens,
    }

    if method == "saliency":
        gxi = (grad * emb).sum(-1)[0]              # gradient × input
        gnorm = grad.norm(dim=-1)[0]
        result["scores"] = _token_rows(tokens, gxi.tolist())
        result["grad_norms"] = [float(v) for v in gnorm]
        return result

    if method != "ig":
        raise ValueError("method must be 'saliency' or 'ig'")

    # integrated gradients: path integral in embedding space via the
    # override mechanism, from a pad/eos-embedding baseline
    steps = max(4, min(int(steps), 64))
    baseline, baseline_desc = _baseline_embedding(session, emb)
    delta = emb - baseline
    acc = torch.zeros_like(emb)
    for k in range(steps):
        alpha = (k + 0.5) / steps
        ov = (baseline + alpha * delta).clone().requires_grad_(True)
        tr = run_forward(model, example, kwargs, overrides={emb_ci: ov},
                         light=True, detach_output=False)
        lg = first_tensor(tr.output)
        step_logit, _ = _target_logit(session, lg, {"id": tid}, contrast_id)
        (g,) = torch.autograd.grad(step_logit, ov)
        acc += g
    attr = delta * acc / steps                     # (x − baseline) ⊙ mean-grad
    scores = attr.sum(-1)[0]
    # completeness axiom: Σ attributions ≈ f(x) − f(baseline)
    tr0 = run_forward(model, example, kwargs, overrides={emb_ci: baseline},
                      light=True)
    base_logit, _ = _target_logit(session, first_tensor(tr0.output),
                                  {"id": tid}, contrast_id)
    result.update({
        "steps": steps,
        "baseline": baseline_desc,
        "scores": _token_rows(tokens, scores.tolist()),
        "completeness": {
            "sum_attributions": float(scores.sum()),
            "logit_input": float(logit.detach()),
            "logit_baseline": float(base_logit.detach()),
            "difference": float(logit.detach() - base_logit.detach()),
        },
    })
    return result


def _attribute_vision(session, example, desc, target_spec, method, steps):
    model = session.model

    def grad_for(x, spec):
        x = x.clone().requires_grad_(True)
        tr = run_forward(model, x, None, light=True, detach_output=False)
        logit, tid = _target_logit(session, first_tensor(tr.output), spec)
        (g,) = torch.autograd.grad(logit, x)
        return g, logit, tid

    if method == "saliency":
        g, logit, tid = grad_for(example, target_spec)
        sal = g[0].abs().amax(dim=0) if g.dim() == 4 else g.abs()
        return {
            "kind": "vision", "method": "saliency",
            "target": {"id": tid, "label": f"class {tid}",
                       "logit": float(logit.detach())},
            "map": summarize_tensor(sal, name="saliency |∂logit/∂pixel|"),
        }

    steps = max(4, min(int(steps), 64))
    # fix the target ONCE at the real input — interpolated inputs must all
    # attribute the same logit, or the path integral is meaningless
    _, logit_x, tid = grad_for(example, target_spec)
    acc = torch.zeros_like(example)
    for k in range(steps):
        alpha = (k + 0.5) / steps
        g, _, _ = grad_for(alpha * example, {"id": tid})
        acc += g
    attr = example * acc / steps
    amap = attr[0].sum(dim=0) if attr.dim() == 4 else attr
    with torch.no_grad():
        out0 = model(torch.zeros_like(example))
        logit_0 = first_tensor(out0)[0]
        logit_0 = logit_0[tid] if logit_0.dim() == 1 else logit_0[-1, tid]
    return {
        "kind": "vision", "method": "ig", "steps": steps,
        "target": {"id": tid, "label": f"class {tid}",
                   "logit": float(logit_x.detach())},
        "completeness": {
            "sum_attributions": float(attr.sum()),
            "logit_input": float(logit_x.detach()),
            "logit_baseline": float(logit_0),
            "difference": float(logit_x.detach() - logit_0),
        },
        "map": summarize_tensor(amap, name="integrated gradients"),
    }
