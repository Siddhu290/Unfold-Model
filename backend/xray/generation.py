"""Streaming token generation + full output distribution.

Generation is a plain autoregressive loop over the SAME model the hooks
inspect — each step is one real forward call. The step loop itself runs
un-hooked for speed; when the user pauses to inspect a specific token, the
frontend re-runs the standard hooked /forward on that exact prefix, so the
graph/tree state is the genuine computation that produced the token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .execution import make_input
from .summarize import first_tensor


def _step_logits(session, ids: torch.Tensor) -> torch.Tensor:
    """One real forward call -> last-position logits (vocab,)."""
    model = session.model
    with torch.no_grad():
        if session.load_kind == "hf":
            out = model(input_ids=ids)
            logits = first_tensor(out)
        else:
            max_len = getattr(model, "MAX_LEN", None)
            window = ids[:, -max_len:] if max_len else ids
            logits = model(window)
    return logits[0, -1].float()


def _decode_one(session, tid: int) -> str:
    if session.load_kind == "hf" and session.tokenizer is not None:
        return session.tokenizer.decode([tid])
    if hasattr(session.model, "decode"):
        return session.model.decode([tid])
    return str(tid)


def generate_stream(session, input_spec: dict, max_new_tokens: int = 20,
                    mode: str = "greedy", temperature: float = 1.0,
                    topk: int = 5):
    """Yield one dict per generated token (consumed as NDJSON by the UI)."""
    if not session.runnable:
        raise ValueError("State-dict-only session cannot generate.")
    example, kwargs, desc = make_input(session, input_spec)
    if desc.get("kind") != "text":
        raise ValueError("Generation needs a text input (language model).")
    if session.load_kind == "hf":
        ids = torch.tensor([desc["token_ids"]], dtype=torch.long)
    else:
        ids = example
    eos_id = None
    if session.load_kind == "hf" and session.tokenizer is not None:
        eos_id = session.tokenizer.eos_token_id

    yield {"event": "start", "prompt_tokens": desc["tokens"],
           "prompt_text": desc.get("text", "")}

    generated = []
    for step in range(max_new_tokens):
        logits = _step_logits(session, ids)
        temp = max(1e-4, float(temperature))
        probs = F.softmax(logits / (temp if mode == "sample" else 1.0), dim=-1)
        if mode == "sample":
            tid = int(torch.multinomial(probs, 1).item())
        else:
            tid = int(probs.argmax().item())
        k = min(topk, probs.numel())
        top = torch.topk(probs, k)
        entry = {
            "event": "token",
            "step": step,
            "id": tid,
            "token": _decode_one(session, tid),
            "prob": float(probs[tid].item()),
            "topk": [{"id": int(i), "label": _decode_one(session, int(i)),
                      "prob": float(p)} for p, i in zip(top.values, top.indices)],
        }
        generated.append(entry["token"])
        yield entry
        ids = torch.cat([ids, torch.tensor([[tid]], dtype=torch.long)], dim=1)
        if eos_id is not None and tid == eos_id:
            break

    yield {"event": "done", "text": "".join(generated), "n_tokens": len(generated)}


def full_distribution(session, offset: int = 0, limit: int = 500):
    """A sorted page of the complete softmax over the final logits of the
    last forward pass. Sorting is cached on the session so paging is cheap."""
    trace = session.last_trace
    if trace is None or trace.output is None:
        raise ValueError("Run a forward pass first.")
    logits = first_tensor(trace.output)
    if logits.dim() == 3:
        last = logits[0, -1]
    elif logits.dim() == 2:
        last = logits[0]
    else:
        raise ValueError("Final output is not a logits vector.")

    cache = getattr(session, "_dist_cache", None)
    key = id(trace)
    if cache is None or cache["key"] != key:
        probs = F.softmax(last.float(), dim=-1)
        order = torch.argsort(probs, descending=True)
        cache = {"key": key, "probs": probs, "order": order}
        session._dist_cache = cache

    probs, order = cache["probs"], cache["order"]
    total = probs.numel()
    offset = max(0, min(offset, total))
    idx = order[offset: offset + limit]
    entries = [{"rank": offset + j, "id": int(i),
                "label": _decode_one(session, int(i)),
                "prob": float(probs[i])} for j, i in enumerate(idx)]
    return {"total": total, "offset": offset, "entries": entries}
