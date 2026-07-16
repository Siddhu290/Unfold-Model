"""Activation patching (causal tracing).

Run the model on a clean prompt and a corrupted prompt, then re-run the
corrupted prompt with one layer's hidden state spliced from the clean run at
chosen token positions. If the answer flips back, that (layer, positions)
site is CAUSALLY responsible for the behavior — not merely correlated.

Positions default to the tokens where the two prompts differ (the corrupted
subject). Patching ALL positions of any residual-stream layer must restore
the clean output exactly — that invariant is what validates the override
machinery itself.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .execution import make_input, run_forward
from .lens import _block_paths
from .summarize import first_tensor


def _last_pos_probs(trace) -> torch.Tensor:
    logits = first_tensor(trace.output)
    last = logits[0, -1] if logits.dim() == 3 else logits[0]
    return F.softmax(last.float(), dim=-1)


def _resolve_target_id(session, target_spec, clean_probs) -> int:
    spec = target_spec or {}
    if spec.get("id") is not None:
        return int(spec["id"])
    text = spec.get("text")
    if text:
        if session.load_kind == "hf":
            ids = session.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"Target {text!r} produced no tokens.")
            return ids[0]
        if hasattr(session.model, "encode"):
            return int(session.model.encode(text)[0, 0].item())
    return int(clean_probs.argmax().item())


def _decode(session, tid: int) -> str:
    from .generation import _decode_one
    return _decode_one(session, tid)


class PatchContext:
    """Shared setup for single patches AND circuit sweeps — one code path,
    so sweep numbers can never contradict the single-layer results."""

    def __init__(self, session, clean_spec, corrupted_spec, target_spec,
                 positions="diff"):
        if not session.runnable:
            raise ValueError("State-dict-only session cannot run patches.")
        import time as _t
        self.session = session
        self.model = session.model
        self.clean_ex, self.clean_kw, self.clean_desc = make_input(session, clean_spec)
        self.corr_ex, self.corr_kw, self.corr_desc = make_input(session, corrupted_spec)
        clean_ids = self.clean_desc.get("token_ids")
        corr_ids = self.corr_desc.get("token_ids")
        if clean_ids is None or corr_ids is None:
            raise ValueError("Activation patching needs text inputs.")
        for name, ids, text in (("clean", clean_ids, self.clean_desc.get("text")),
                                ("corrupted", corr_ids, self.corr_desc.get("text"))):
            if not ids or not (text or "").strip():
                raise ValueError(
                    f"The {name} prompt is empty. Both boxes need real text — "
                    "the grey text shown in an empty box is only an example. "
                    "Use two prompts that differ in one subject and tokenize "
                    "to the same length, e.g. “The city of Paris is located "
                    "in the country of” vs the same sentence with Rome.")
        if len(clean_ids) != len(corr_ids):
            raise ValueError(
                f"Prompts must tokenize to the SAME length for position-aligned "
                f"patching: clean={len(clean_ids)} vs corrupted={len(corr_ids)} "
                "tokens. Reword one prompt.")

        self.clean_trace = run_forward(self.model, self.clean_ex, self.clean_kw)
        t0 = _t.perf_counter()
        self.corr_trace = run_forward(self.model, self.corr_ex, self.corr_kw)
        self.fwd_seconds = _t.perf_counter() - t0

        self.clean_probs = _last_pos_probs(self.clean_trace)
        self.corr_probs = _last_pos_probs(self.corr_trace)
        self.target_id = _resolve_target_id(session, target_spec, self.clean_probs)
        self.p_clean = float(self.clean_probs[self.target_id])
        self.p_corr = float(self.corr_probs[self.target_id])
        self.denom = self.p_clean - self.p_corr

        if positions == "diff":
            self.pos = [i for i, (a, b) in enumerate(zip(clean_ids, corr_ids))
                        if a != b]
            if not self.pos:
                raise ValueError("The prompts are identical — nothing to patch.")
        elif positions == "all":
            self.pos = None
        elif positions == "last":
            self.pos = [len(clean_ids) - 1]
        else:
            self.pos = [int(p) for p in positions]

        self.first_call = {}
        for rec in self.clean_trace.records:
            self.first_call.setdefault(rec["path"], rec["call_index"])
        self.diff_positions = [i for i, (a, b) in
                               enumerate(zip(clean_ids, corr_ids)) if a != b]

    def run_override(self, ci: int, patched_act) -> dict:
        """One patched corrupted-input run; the restoration metric."""
        tr = run_forward(self.model, self.corr_ex, self.corr_kw,
                         overrides={ci: patched_act}, light=True)
        probs = _last_pos_probs(tr)
        p_patch = float(probs[self.target_id])
        top_id = int(probs.argmax())
        return {
            "p_target": p_patch,
            "restoration": ((p_patch - self.p_corr) / self.denom
                            if abs(self.denom) > 1e-9 else None),
            "top1": _decode(self.session, top_id),
            "flipped_back": top_id == int(self.clean_probs.argmax()),
        }

    def patch_layer(self, path: str) -> dict:
        """Splice the clean hidden state at self.pos into one layer."""
        ci = self.first_call.get(path)
        entry = {"path": path, "call_index": ci}
        t_clean = self.clean_trace.tensors.get(ci) if ci is not None else None
        t_corr = self.corr_trace.tensors.get(ci) if ci is not None else None
        if t_clean is None or t_corr is None or t_clean.dim() != 3:
            entry["error"] = "no retained (batch, seq, d) activation at this layer"
            return entry
        if self.pos is None:
            patched_act = t_clean.clone()
        else:
            patched_act = t_corr.clone()
            patched_act[:, self.pos, :] = t_clean[:, self.pos, :]
        entry.update(self.run_override(ci, patched_act))
        return entry

    def summary(self) -> dict:
        return {
            "target": {"id": self.target_id,
                       "label": _decode(self.session, self.target_id)},
            "clean": {"text": self.clean_desc.get("text"), "p_target": self.p_clean,
                      "top1": _decode(self.session, int(self.clean_probs.argmax()))},
            "corrupted": {"text": self.corr_desc.get("text"), "p_target": self.p_corr,
                          "top1": _decode(self.session, int(self.corr_probs.argmax()))},
        }


def causal_patch(session, clean_spec: dict, corrupted_spec: dict,
                 target_spec: dict = None, layer_paths: list = None,
                 positions="diff", progress=None) -> dict:
    """Patch clean activations into the corrupted run, one layer at a time."""
    ctx = PatchContext(session, clean_spec, corrupted_spec, target_spec, positions)
    if layer_paths is None:
        layer_paths = _block_paths(session.arch)
        if not layer_paths:
            raise ValueError("No repeated blocks found to patch; pass layer_paths.")

    results = []
    for i, path in enumerate(layer_paths):
        results.append(ctx.patch_layer(path))
        if progress:
            progress(i + 1, len(layer_paths))

    clean_desc, corr_desc = ctx.clean_desc, ctx.corr_desc
    clean_ids = clean_desc.get("token_ids")
    corr_ids = corr_desc.get("token_ids")
    pos, p_clean, p_corr = ctx.pos, ctx.p_clean, ctx.p_corr
    clean_probs, corr_probs = ctx.clean_probs, ctx.corr_probs
    target_id = ctx.target_id

    return {
        "target": {"id": target_id, "label": _decode(session, target_id)},
        "positions": pos if pos is not None else "all",
        "diff_tokens": [
            {"pos": i, "clean": clean_desc["tokens"][i], "corrupted": corr_desc["tokens"][i]}
            for i, (a, b) in enumerate(zip(clean_ids, corr_ids)) if a != b],
        "clean": {"text": clean_desc.get("text"), "p_target": p_clean,
                  "top1": _decode(session, int(clean_probs.argmax()))},
        "corrupted": {"text": corr_desc.get("text"), "p_target": p_corr,
                      "top1": _decode(session, int(corr_probs.argmax()))},
        "results": results,
    }
