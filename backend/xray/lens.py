"""Logit lens: project intermediate hidden states through the model's OWN
final norm + unembedding, showing what each layer 'would have predicted'.

The final-projection path differs by model family, so it is resolved through
a small registry (same spirit as theory.py) with two fallbacks: the HF
get_output_embeddings() API, and — since we always have a real trace — the
last *Norm*-class call before the output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# root class name -> (final norm path or None, head path)
FINAL_PROJECTION_REGISTRY: dict[str, tuple] = {
    "GPT2LMHeadModel": ("transformer.ln_f", "lm_head"),
    "GPTNeoXForCausalLM": ("gpt_neox.final_layer_norm", "embed_out"),
    "LlamaForCausalLM": ("model.norm", "lm_head"),
    "TinyTransformerLM": ("ln_f", "lm_head"),
}


def register_final_projection(root_class: str, norm_path, head_path):
    FINAL_PROJECTION_REGISTRY[root_class] = (norm_path, head_path)


def resolve_final_projection(session):
    """Returns (norm_module_or_None, head_module, meta) or raises."""
    model = session.model
    modules = dict(model.named_modules())
    root = type(model).__name__

    if root in FINAL_PROJECTION_REGISTRY:
        norm_path, head_path = FINAL_PROJECTION_REGISTRY[root]
        norm = modules.get(norm_path) if norm_path else None
        head = modules.get(head_path)
        if head is not None:
            return norm, head, {"norm": norm_path, "head": head_path, "via": "registry"}

    # fallback 1: HF's own API for the unembedding
    head = None
    head_path = None
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            for n, m in modules.items():
                if m is head:
                    head_path = n
                    break
    if head is None:
        raise ValueError(
            f"No final projection known for {root}. Register one with "
            "lens.register_final_projection(root_class, norm_path, head_path).")

    # fallback 2: the last Norm-class module call before the output,
    # taken from the REAL execution order of the last trace
    norm, norm_path = None, None
    trace = session.last_trace
    if trace is not None:
        for rec in reversed(trace.records):
            if "norm" in rec["class"].lower() and rec["path"] in modules:
                norm, norm_path = modules[rec["path"]], rec["path"]
                break
    return norm, head, {"norm": norm_path, "head": head_path, "via": "fallback"}


def _head_in_features(head) -> int:
    w = getattr(head, "weight", None)
    if w is None:
        raise ValueError("Unembedding module has no weight.")
    return w.shape[-1]


def _block_paths(arch: dict) -> list:
    """Paths of repeat-group members (transformer blocks) — the natural rows
    of the lens strip."""
    out = []

    def visit(node):
        for g in node.get("repeat_groups", []):
            for c in node["children"][g["start"]: g["start"] + g["count"]]:
                out.append(c["path"])
        for c in node["children"]:
            visit(c)

    visit(arch["tree"])
    return out


def project_hidden(session, hidden: torch.Tensor, k: int = 3):
    """Apply the model's own final norm + head to one hidden vector."""
    norm, head, _ = resolve_final_projection(session)
    with torch.no_grad():
        x = hidden.float()
        if norm is not None:
            x = norm(x)
        logits = head(x)
        probs = F.softmax(logits, dim=-1)
        top = torch.topk(probs, min(k, probs.numel()))
    from .generation import _decode_one
    return [{"id": int(i), "label": _decode_one(session, int(i)),
             "prob": float(p)} for p, i in zip(top.values, top.indices)]


def lens_for_call(session, call_index: int, k: int = 3):
    """Logit-lens projection of one retained activation (last position)."""
    trace = session.last_trace
    if trace is None:
        raise ValueError("Run a forward pass first.")
    t = trace.tensors.get(call_index)
    if t is None:
        raise ValueError(f"No retained activation for call {call_index}.")
    d = _head_in_features(resolve_final_projection(session)[1])
    if t.dim() != 3 or t.shape[-1] != d:
        raise ValueError(
            f"Activation shape {list(t.shape)} is not a (batch, seq, {d}) "
            "hidden state — logit lens does not apply to this layer.")
    rec = trace.records[call_index]
    return {"call_index": call_index, "path": rec["path"], "class": rec["class"],
            "topk": project_hidden(session, t[0, -1], k)}


def logit_lens_strip(session, k: int = 3):
    """Lens rows for the embedding stage, every repeated block, and the final
    output — the classic 'answer crystallizing across depth' view."""
    trace = session.last_trace
    if trace is None:
        raise ValueError("Run a forward pass first.")
    norm, head, meta = resolve_final_projection(session)
    d = _head_in_features(head)

    blocks = set(_block_paths(session.arch))
    first_call_of = {}
    for rec in trace.records:
        first_call_of.setdefault(rec["path"], rec["call_index"])

    rows = []
    # embedding stage: first retained (B, T, d) activation in execution order
    for rec in trace.records:
        t = trace.tensors.get(rec["call_index"])
        if t is not None and t.dim() == 3 and t.shape[-1] == d:
            rows.append({"stage": "embedding", **lens_for_call(session, rec["call_index"], k)})
            break
    for path in sorted(blocks, key=lambda p: first_call_of.get(p, 1 << 30)):
        ci = first_call_of.get(path)
        if ci is None or ci not in trace.tensors:
            continue
        t = trace.tensors[ci]
        if t.dim() == 3 and t.shape[-1] == d:
            rows.append({"stage": "block", **lens_for_call(session, ci, k)})
    if not rows:
        raise ValueError("No block-level hidden states retained — is this a "
                         "transformer with (batch, seq, d_model) activations?")
    # true final output for reference
    from .execution import decode_topk
    final = decode_topk(session, trace.output, k=k)
    return {"rows": rows, "final": final["topk"] if final else None,
            "projection": meta, "input": trace.input_desc}
