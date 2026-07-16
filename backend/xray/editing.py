"""Architecture editing with live re-run validation and diffing.

Every edit mutates the REAL loaded model, is immediately validated by
running the last-used input through it (a failing edit is auto-undone and
reported), and is diffed against the pre-edit model's output so the change
is falsifiable, not cosmetic. Structural edits keep their own undo stack —
separate from the optimizer-step snapshot.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .execution import make_input
from .summarize import first_tensor

ACTIVATION_SWAPS = {
    "relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
}
# keep a deepcopy 'original' for side-by-side re-runs below this size
ORIGINAL_COPY_PARAM_LIMIT = 60_000_000


def _resolve_parent(model: nn.Module, path: str):
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() and isinstance(
            parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, p)
    return parent, parts[-1]


def _get_module(model: nn.Module, path: str) -> nn.Module:
    mod = dict(model.named_modules()).get(path)
    if mod is None:
        raise ValueError(f"No module at path {path!r}")
    return mod


def _set_module(model: nn.Module, path: str, new: nn.Module):
    parent, name = _resolve_parent(model, path)
    if name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(name)] = new
    else:
        setattr(parent, name, new)


def _reset_parameters_recursive(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def _node_shapes(session, path):
    from .extraction import flatten_tree
    node = flatten_tree(session.arch["tree"]).get(path)
    return (node.get("in_shape"), node.get("out_shape")) if node else (None, None)


def _eval_probs(session, model, input_spec):
    example, kwargs, _ = make_input(session, input_spec)
    with torch.no_grad():
        out = model(**kwargs) if kwargs else model(example)
    logits = first_tensor(out)
    last = logits[0, -1] if logits.dim() == 3 else logits[0]
    return F.softmax(last.float(), dim=-1)


def _label(session, tid):
    from .generation import _decode_one
    return _decode_one(session, int(tid))


def ensure_edit_state(session):
    if not hasattr(session, "edits"):
        session.edits = []
        session.original_model = None


def _snapshot_original(session):
    """Keep the pre-edit model for side-by-side comparison (size permitting)."""
    ensure_edit_state(session)
    if session.edits or session.original_model is not None:
        return None
    n = sum(p.numel() for p in session.model.parameters())
    if n <= ORIGINAL_COPY_PARAM_LIMIT:
        session.original_model = copy.deepcopy(session.model).eval()
        return None
    return (f"Model has {n/1e6:.0f}M params — keeping a full pre-edit copy "
            "would double memory, so side-by-side comparison uses the edited "
            "model only against the recorded pre-edit output of the current input.")


def apply_edit(session, edit: dict) -> dict:
    """Apply one structural edit; validate by a real forward; undo on failure."""
    if not session.runnable:
        raise ValueError("State-dict-only session cannot be edited.")
    ensure_edit_state(session)
    model = session.model
    op = edit.get("op")
    path = edit.get("path", "")
    module = _get_module(model, path) if op != "noop" else None
    warnings = []

    warn = _snapshot_original(session)
    if warn:
        warnings.append(warn)

    if op == "swap_activation":
        to = edit.get("to")
        if to not in ACTIVATION_SWAPS:
            raise ValueError(f"Unknown activation {to!r}; options: {list(ACTIVATION_SWAPS)}")
        old = module
        if not any(a in type(old).__name__.lower()
                   for a in ("relu", "gelu", "tanh", "sigmoid", "elu", "silu",
                             "activation", "identity", "mish")):
            raise ValueError(f"{path} is a {type(old).__name__}, not an "
                             "activation — swap refused.")
        _set_module(model, path, ACTIVATION_SWAPS[to]())
        desc = f"swap {path}: {type(old).__name__} → {to.upper()}"
        undo = lambda: _set_module(model, path, old)

    elif op == "remove":
        old = module
        in_s, out_s = _node_shapes(session, path)
        if in_s and out_s and in_s != out_s:
            raise ValueError(
                f"Cannot remove {path}: it maps {in_s} → {out_s}. Removing it "
                "would feed the wrong shape downstream. Only shape-preserving "
                "layers can be removed.")
        _set_module(model, path, nn.Identity())
        desc = f"remove {path} ({type(old).__name__} → Identity)"
        undo = lambda: _set_module(model, path, old)

    elif op == "duplicate":
        parent, name = _resolve_parent(model, path)
        if not (name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList))):
            raise ValueError("Duplicate works on members of Sequential/ModuleList "
                             f"containers; {path} is a named attribute.")
        in_s, out_s = _node_shapes(session, path)
        if in_s and out_s and in_s != out_s:
            raise ValueError(
                f"Cannot duplicate {path}: it maps {in_s} → {out_s}, so its "
                "copy cannot consume its own output.")
        init = edit.get("init", "copy")            # explicit modeling decision
        clone = copy.deepcopy(module)
        if init == "random":
            _reset_parameters_recursive(clone)
        idx = int(name)
        mods = list(parent)
        mods.insert(idx + 1, clone)
        _replace_container_contents(parent, mods)
        desc = f"duplicate {path} ({'copied' if init == 'copy' else 'random'} weights)"

        def undo(parent=parent, idx=idx):
            mods = list(parent)
            mods.pop(idx + 1)
            _replace_container_contents(parent, mods)

    elif op == "reorder":
        parent, name = _resolve_parent(model, path)
        if not (name.isdigit() and isinstance(parent, nn.Sequential)):
            raise ValueError("Reorder works inside nn.Sequential containers "
                             "(where child order IS execution order).")
        idx = int(name)
        j = idx + (1 if edit.get("direction", "down") == "down" else -1)
        if not (0 <= j < len(parent)):
            raise ValueError("Already at the edge of its container.")
        mods = list(parent)
        mods[idx], mods[j] = mods[j], mods[idx]
        _replace_container_contents(parent, mods)
        desc = f"reorder: swap positions {idx} ↔ {j} in {path.rsplit('.', 1)[0] or 'model'}"

        def undo(parent=parent, idx=idx, j=j):
            mods = list(parent)
            mods[idx], mods[j] = mods[j], mods[idx]
            _replace_container_contents(parent, mods)

    else:
        raise ValueError(f"Unknown edit op {op!r}")

    # validate with a REAL forward on the last-used input; auto-undo on failure
    spec = getattr(session, "last_input_spec", None) or _default_input_spec(session)
    try:
        _eval_probs(session, model, spec)
    except Exception as e:
        undo()
        raise ValueError(
            f"Edit rejected — the model no longer runs ({type(e).__name__}: {e}). "
            "It was automatically reverted.")

    session.edits.append({"desc": desc, "undo": undo})
    return {"desc": desc, "warnings": warnings,
            "history": [e["desc"] for e in session.edits]}


def _replace_container_contents(parent, mods):
    if isinstance(parent, nn.Sequential):
        for k in list(parent._modules):
            del parent._modules[k]
        for i, m in enumerate(mods):
            parent.add_module(str(i), m)
    else:  # ModuleList
        parent._modules.clear()
        for i, m in enumerate(mods):
            parent._modules[str(i)] = m


def undo_edit(session) -> dict:
    ensure_edit_state(session)
    if not session.edits:
        raise ValueError("No structural edits to undo.")
    last = session.edits.pop()
    last["undo"]()
    if not session.edits:
        session.original_model = None   # back to pristine
    return {"undone": last["desc"], "history": [e["desc"] for e in session.edits]}


def _default_input_spec(session):
    if session.tokenizer is not None or hasattr(session.model, "encode"):
        return {"kind": "text", "text": "Hello world"}
    shape = session.meta.get("input_shape")
    if shape:
        return {"kind": "tensor", "shape": shape}
    raise ValueError("No input available to validate the edit — run a forward "
                     "pass first.")


def compare_outputs(session, input_spec=None, k: int = 5) -> dict:
    """Original-vs-edited output distributions for the same real input."""
    ensure_edit_state(session)
    spec = input_spec or getattr(session, "last_input_spec", None) \
        or _default_input_spec(session)
    if session.original_model is None:
        raise ValueError("No pre-edit model copy available for comparison "
                         "(model too large, or no edits made).")
    p_orig = _eval_probs(session, session.original_model, spec)
    p_edit = _eval_probs(session, session.model, spec)
    kl = float(F.kl_div(p_edit.clamp_min(1e-12).log(), p_orig,
                        reduction="sum").item())
    idx = torch.unique(torch.cat([torch.topk(p_orig, k).indices,
                                  torch.topk(p_edit, k).indices]))
    rows = sorted(
        ({"id": int(i), "label": _label(session, i),
          "p_original": float(p_orig[i]), "p_edited": float(p_edit[i]),
          "delta": float(p_edit[i] - p_orig[i])} for i in idx),
        key=lambda r: -max(r["p_original"], r["p_edited"]))
    return {
        "input": spec,
        "kl_divergence": kl,
        "top1_original": _label(session, int(p_orig.argmax())),
        "top1_edited": _label(session, int(p_edit.argmax())),
        "top1_changed": int(p_orig.argmax()) != int(p_edit.argmax()),
        "rows": rows,
    }
