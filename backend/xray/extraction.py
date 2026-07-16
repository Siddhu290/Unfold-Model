"""Architecture extraction.

Walks any nn.Module generically via named_modules()/named_parameters() and
produces a JSON-safe tree:
  - every module: path, class, own/total param counts, extra_repr, children
  - in/out shapes filled in by a hook-based shape probe when an example
    input is available (execution order can differ from tree order)
  - repeated identical blocks (e.g. 32 transformer layers) are detected by
    structural signature so the UI can render "block ×32"

Also handles the weights-only case (state dict from .safetensors or a
weights_only .pt) where no architecture object exists: the tree is inferred
from parameter key prefixes.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import torch
from torch import nn

from .summarize import shape_of


def _own_params(module: nn.Module):
    return [
        {"name": name, "shape": list(p.shape), "numel": p.numel(),
         "trainable": p.requires_grad}
        for name, p in module.named_parameters(recurse=False)
    ]


def _total_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_module_tree(model: nn.Module) -> dict:
    """Recursive tree of every module in the model."""

    def visit(module: nn.Module, path: str) -> dict:
        own = _own_params(module)
        node = {
            "path": path,
            "class": type(module).__name__,
            "module_type": f"{type(module).__module__}.{type(module).__name__}",
            "extra_repr": module.extra_repr(),
            "params": own,
            "own_param_count": sum(p["numel"] for p in own),
            "total_param_count": _total_param_count(module),
            "is_leaf": len(list(module.children())) == 0,
            "in_shape": None,
            "out_shape": None,
            "call_order": None,
            "children": [
                visit(child, f"{path}.{name}" if path else name)
                for name, child in module.named_children()
            ],
        }
        node["signature"] = _signature(node)
        return node

    return visit(model, "")


def _signature(node: dict) -> str:
    """Structural fingerprint: class + own param shapes + child signatures.
    Two modules with the same signature are architecturally identical."""
    parts = [node["class"], node["extra_repr"]]
    parts += [f'{p["name"]}:{p["shape"]}' for p in node["params"]]
    parts += [c["signature"] for c in node["children"]]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def mark_repeat_groups(node: dict) -> dict:
    """Group consecutive structurally-identical children into repeat groups.

    Adds node["repeat_groups"] = [{"start": i, "count": n, "signature": s}]
    for every run of >= 3 identical children (typical transformer stacks).
    Only integer-named children (ModuleList/Sequential members) qualify:
    identically-shaped NAMED siblings like q_proj/k_proj/v_proj are parallel
    roles, not a repeated stack, and must stay individually visible.
    """
    children = node["children"]
    groups = []
    i = 0
    while i < len(children):
        j = i
        while (j + 1 < len(children)
               and children[j + 1]["signature"] == children[i]["signature"]):
            j += 1
        run = j - i + 1
        indexed = all(c["path"].rsplit(".", 1)[-1].isdigit()
                      for c in children[i:j + 1])
        if run >= 3 and indexed:
            groups.append({
                "start": i,
                "count": run,
                "signature": children[i]["signature"],
                "class": children[i]["class"],
            })
        i = j + 1
    if groups:
        node["repeat_groups"] = groups
    for c in children:
        mark_repeat_groups(c)
    return node


def probe_shapes(model: nn.Module, example_input, input_kwargs: Optional[dict] = None):
    """Run one forward pass with hooks to record real in/out shapes and the
    true execution order of every module (which can differ from tree order)."""
    records = {}
    order = [0]
    handles = []

    name_of = {id(m): n for n, m in model.named_modules()}

    def hook(module, args, output):
        path = name_of.get(id(module))
        if path is None:
            return
        rec = records.setdefault(path, {})
        if "call_order" not in rec:  # keep first call (modules can be re-entered)
            rec["call_order"] = order[0]
            rec["in_shape"] = shape_of(args)
            rec["out_shape"] = shape_of(output)
        rec["n_calls"] = rec.get("n_calls", 0) + 1
        order[0] += 1

    for _, m in model.named_modules():
        handles.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            if input_kwargs:
                model(**input_kwargs)
            else:
                model(example_input)
    finally:
        for h in handles:
            h.remove()
    return records


def apply_shape_records(tree: dict, records: dict):
    def visit(node):
        rec = records.get(node["path"])
        if rec:
            node["in_shape"] = rec.get("in_shape")
            node["out_shape"] = rec.get("out_shape")
            node["call_order"] = rec.get("call_order")
            node["n_calls"] = rec.get("n_calls")
        for c in node["children"]:
            visit(c)

    visit(tree)
    return tree


def extract_architecture(
    model: nn.Module,
    example_input=None,
    input_kwargs: Optional[dict] = None,
) -> dict:
    """Full architecture extraction; shape probe is best-effort."""
    tree = build_module_tree(model)
    mark_repeat_groups(tree)
    probe_error = None
    if example_input is not None or input_kwargs:
        try:
            records = probe_shapes(model, example_input, input_kwargs)
            apply_shape_records(tree, records)
        except Exception as e:
            probe_error = f"{type(e).__name__}: {e}"
    return {
        "mode": "module",
        "root_class": type(model).__name__,
        "total_params": _total_param_count(model),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "num_modules": sum(1 for _ in model.modules()),
        "shape_probe_error": probe_error,
        "tree": tree,
    }


# ---------------------------------------------------------------------------
# Weights-only mode: infer a tree from state-dict key prefixes
# ---------------------------------------------------------------------------

def extract_from_state_dict(sd: dict) -> dict:
    """Build a display tree from parameter names alone (no forward pass is
    possible — there is no architecture object, just named tensors)."""
    root = {
        "path": "", "class": "(state dict)", "module_type": "state_dict",
        "extra_repr": "", "params": [], "own_param_count": 0,
        "total_param_count": 0, "is_leaf": False, "in_shape": None,
        "out_shape": None, "call_order": None, "children": [], "signature": "",
        "_index": {},
    }

    def get_child(parent, name):
        idx = parent.setdefault("_index", {})
        if name not in idx:
            child = {
                "path": f'{parent["path"]}.{name}' if parent["path"] else name,
                "class": "(module)", "module_type": "inferred", "extra_repr": "",
                "params": [], "own_param_count": 0, "total_param_count": 0,
                "is_leaf": True, "in_shape": None, "out_shape": None,
                "call_order": None, "children": [], "signature": "", "_index": {},
            }
            idx[name] = child
            parent["children"].append(child)
            parent["is_leaf"] = False
        return idx[name]

    total = 0
    for key, tensor in sd.items():
        if not torch.is_tensor(tensor):
            continue
        parts = key.split(".")
        node = root
        for part in parts[:-1]:
            node = get_child(node, part)
        node["params"].append({
            "name": parts[-1], "shape": list(tensor.shape),
            "numel": tensor.numel(), "trainable": True,
        })
        node["own_param_count"] += tensor.numel()
        total += tensor.numel()

    def finalize(node):
        node.pop("_index", None)
        node["total_param_count"] = node["own_param_count"] + sum(
            finalize(c) for c in node["children"]
        )
        node["signature"] = _signature(node)
        return node["total_param_count"]

    finalize(root)
    mark_repeat_groups(root)
    return {
        "mode": "state_dict",
        "root_class": "(state dict — weights only, architecture inferred from key names)",
        "total_params": total,
        "trainable_params": total,
        "num_modules": None,
        "shape_probe_error": None,
        "tree": root,
    }


def flatten_tree(tree: dict) -> dict:
    """path -> node lookup (children excluded) for quick access."""
    out = {}

    def visit(node):
        out[node["path"]] = node
        for c in node["children"]:
            visit(c)

    visit(tree)
    return out
