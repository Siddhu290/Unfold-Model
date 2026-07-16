"""Real forward/backward execution with hook-based capture.

Nothing here is simulated: the forward trace is captured with
register_forward_hook on every submodule during an actual model call, the
gradients come from a real loss.backward(), and the optimizer step mutates
the real weights (with a pre-step snapshot so the diff — and an undo — are
possible).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .summarize import first_tensor, shape_of, summarize_tensor_light

# retain activation tensors for drill-down only below this size
ACTIVATION_RETAIN_MAX = 4_000_000
# allow a pre-step weight snapshot (for diff/undo) only below this many params
# (~2 GB of fp32 CPU RAM at the limit)
SNAPSHOT_PARAM_LIMIT = 500_000_000


def _replace_first_tensor(obj, new):
    """Swap the first tensor inside a (possibly nested tuple) module output,
    keeping the container structure intact for downstream code."""
    if torch.is_tensor(obj):
        return new
    if isinstance(obj, tuple):
        out, done = [], False
        for item in obj:
            if not done and first_tensor(item) is not None:
                out.append(_replace_first_tensor(item, new))
                done = True
            else:
                out.append(item)
        return tuple(out)
    return new


class ForwardTrace:
    def __init__(self):
        self.records: list[dict] = []       # one per module call, execution order
        self.tensors: dict[int, torch.Tensor] = {}      # call_index -> activation
        self.act_grads: dict[int, dict] = {}            # call_index -> grad summary
        self.grad_tensors: dict[int, torch.Tensor] = {} # call_index -> full grad
        self.output = None                  # final model output tensor
        self.output_summary = None
        self.attentions = None              # list[tensor] for HF models
        self.input_desc = None
        self.llm = None                     # top-k decode info
        self.edges = None                   # true dataflow edges between calls


def run_forward(
    model: nn.Module,
    example_input=None,
    input_kwargs: Optional[dict] = None,
    capture_grads: bool = False,
    overrides: Optional[dict] = None,
    light: bool = False,
    detach_output: bool = True,
) -> ForwardTrace:
    """Execute model(input) with a forward hook on every submodule.

    capture_grads=True additionally attaches tensor hooks to each module
    output so a later .backward() fills trace.act_grads with the real
    dL/d(activation) at every layer.

    overrides maps call_index -> tensor: when that module call happens, its
    computed output is REPLACED by the stored tensor (forward hooks that
    return a value substitute the module output), and everything downstream
    consumes the patched value. This is the activation-patching primitive —
    call indices are stable across runs with identically-shaped inputs.

    light=True skips summaries/retention/topology (used by patch sweeps and
    ablations that only need the final output of a patched run).
    """
    import time as _time

    trace = ForwardTrace()
    name_of = {id(m): n for n, m in model.named_modules()}
    handles = []
    overrides = overrides or {}
    # per-call live tensor refs, kept only until edge topology is computed
    # (holding the refs keeps id() stable and the autograd graph reachable)
    flow = []   # (call_index, is_leaf, [input tensors], output tensor)
    tstack = []  # pre/post hook pairs nest LIFO in a synchronous forward

    def pre_hook(module, args, kwargs):
        tstack.append(_time.perf_counter())

    def hook(module, args, kwargs, output):
        path = name_of.get(id(module))
        if path is None:
            return
        duration_ms = (_time.perf_counter() - tstack.pop()) * 1000 if tstack else None
        idx = len(trace.records)
        patched = idx in overrides
        if patched:
            ov = overrides[idx]
            got = first_tensor(output)
            if callable(ov):
                # transform override (steering): computed -> modified tensor.
                # Same substitution point as a replacement patch.
                new = ov(got)
            else:
                if got is not None and list(ov.shape) != list(got.shape):
                    raise ValueError(
                        f"Override for call {idx} ({path}) has shape "
                        f"{list(ov.shape)}, module produced {list(got.shape)}.")
                new = ov.to(got.dtype) if got is not None else ov
            # preserve tuple/ModelOutput structure: swap only the first tensor
            output = _replace_first_tensor(output, new)
        out_t = first_tensor(output)
        if light:
            trace.records.append({"call_index": idx, "path": path,
                                  "class": type(module).__name__,
                                  "patched": patched})
            return output if patched else None
        rec = {
            "call_index": idx,
            "path": path,
            "class": type(module).__name__,
            "in_shape": shape_of(args),
            "out_shape": list(out_t.shape) if out_t is not None else None,
            "output": summarize_tensor_light(out_t) if out_t is not None else None,
            "retained": False,
            "patched": patched,
            "duration_ms": duration_ms,
        }
        if out_t is not None:
            if out_t.numel() <= ACTIVATION_RETAIN_MAX:
                trace.tensors[idx] = out_t.detach().cpu().clone()
                rec["retained"] = True
            if capture_grads and out_t.requires_grad:
                def grad_hook(g, idx=idx, path=path):
                    trace.act_grads[idx] = summarize_tensor_light(g, name=path)
                    if g.numel() <= ACTIVATION_RETAIN_MAX:
                        trace.grad_tensors[idx] = g.detach().cpu().clone()
                out_t.register_hook(grad_hook)
        is_leaf = len(module._modules) == 0
        in_ts = [t for t in list(args) + list(kwargs.values()) if torch.is_tensor(t)]
        flow.append((idx, is_leaf, in_ts, out_t))
        trace.records.append(rec)
        return output if patched else None

    for _, m in model.named_modules():
        handles.append(m.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(m.register_forward_hook(hook, with_kwargs=True))

    # Always run with grad enabled: fused inference fast paths (e.g.
    # nn.TransformerEncoderLayer, nn.MultiheadAttention in no-grad eval mode)
    # bypass submodules entirely, which would blind the hooks. The autograd
    # graph is dropped right after the pass when capture_grads is False.
    try:
        with torch.enable_grad():
            if input_kwargs:
                output = model(**input_kwargs)
            else:
                output = model(example_input)
    finally:
        for h in handles:
            h.remove()

    out_t = first_tensor(output)
    if light:
        if out_t is None:
            trace.output = None
        else:
            # detach_output=False keeps the autograd graph alive so callers
            # (integrated gradients) can backprop through an override tensor
            trace.output = out_t.detach() if detach_output else out_t
        return trace
    try:
        trace.edges = _compute_edges(flow, out_t)
    except Exception:
        trace.edges = None
    flow.clear()  # release tensor refs
    trace.output_summary = summarize_tensor_light(out_t) if out_t is not None else None
    if hasattr(output, "attentions") and output.attentions is not None:
        trace.attentions = [a.detach().cpu() for a in output.attentions]
    else:
        # models with explicit attn_probs taps (see demo transformer)
        taps = [trace.tensors[r["call_index"]] for r in trace.records
                if r["path"].endswith("attn_probs") and r["call_index"] in trace.tensors
                and r["out_shape"] and len(r["out_shape"]) == 4]
        if taps:
            trace.attentions = taps
    if capture_grads or not detach_output:
        trace.output = output
    else:
        trace.output = out_t.detach() if out_t is not None else None  # free the graph
    return trace


# ---------------------------------------------------------------------------
# Dataflow edge topology
# ---------------------------------------------------------------------------

INPUT_NODE = -1    # virtual source: the model's input
OUTPUT_NODE = -2   # virtual sink: the model's final output


def _latest_before(writers, before_idx):
    """Last call index in a (sorted) writer list that ran before before_idx.
    Pass-through modules (nn.Identity, eval-mode Dropout) re-emit the SAME
    tensor object as their input, so one tensor/grad_fn can have several
    'writers'; the honest producer for a consumer is the most recent one
    that ran earlier."""
    for w in reversed(writers):
        if w < before_idx:
            return w
    return None


def _find_producers(t, prod_by_tid, prod_by_fn, before_idx, max_visits=400):
    """Which earlier module calls produced tensor `t`?

    Direct identity is checked first; otherwise walk the autograd graph
    upward through the functional ops that sit between modules (residual
    adds, concats, reshapes, softmax, ...) and stop at the first grad_fn
    that is some module call's output. This is what recovers branch/rejoin
    topology that the static module tree cannot express.
    """
    writers = prod_by_tid.get(id(t))
    if writers:
        src = _latest_before(writers, before_idx)
        if src is not None:
            return {src}
    fn = getattr(t, "grad_fn", None)
    if fn is None:
        return set()
    found, seen, stack, visits = set(), {fn}, [fn], 0
    while stack and visits < max_visits:
        f = stack.pop()
        visits += 1
        hit = _latest_before(prod_by_fn.get(f, ()), before_idx)
        if hit is not None:
            found.add(hit)
            continue  # a producer boundary: don't traverse past it
        for nf, _ in getattr(f, "next_functions", ()):
            if nf is not None and nf not in seen \
                    and type(nf).__name__ != "AccumulateGrad":
                seen.add(nf)
                stack.append(nf)
    return found


def _compute_edges(flow, final_out):
    """Leaf-call-level dataflow edges [{"src": i, "dst": j}] in execution
    order, plus INPUT_NODE/OUTPUT_NODE virtual endpoints. Containers are
    excluded — their calls overlap their children's."""
    prod_by_tid, prod_by_fn = {}, {}
    for idx, is_leaf, _ins, out_t in flow:  # flow is in call order
        if is_leaf and out_t is not None:
            prod_by_tid.setdefault(id(out_t), []).append(idx)
            if out_t.grad_fn is not None:
                prod_by_fn.setdefault(out_t.grad_fn, []).append(idx)

    edges = set()
    for idx, is_leaf, in_ts, _out in flow:
        if not is_leaf:
            continue
        srcs = set()
        for t in in_ts:
            srcs |= _find_producers(t, prod_by_tid, prod_by_fn, before_idx=idx)
        srcs = {s for s in srcs if s < idx}
        if srcs:
            edges |= {(s, idx) for s in srcs}
        else:
            edges.add((INPUT_NODE, idx))

    if final_out is not None:
        for s in _find_producers(final_out, prod_by_tid, prod_by_fn,
                                 before_idx=float("inf")):
            edges.add((s, OUTPUT_NODE))
    return [{"src": s, "dst": d} for s, d in sorted(edges)]


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def make_input(session, spec: dict):
    """Turn a frontend input spec into real model input.

    spec: {"kind": "text", "text": ...} or
          {"kind": "tensor", "shape": [...], "values": optional flat list,
           "fill": "randn"|"zeros"|"ones"}
    Returns (example_input, input_kwargs, input_desc).
    """
    kind = spec.get("kind", "tensor")
    if kind == "text":
        text = spec.get("text", "Hello world")
        if session.load_kind == "hf":
            if session.tokenizer is None:
                raise ValueError("This model has no tokenizer; use tensor input.")
            enc = session.tokenizer(text, return_tensors="pt")
            kwargs = {k: v for k, v in enc.items()}
            if session.supports_attentions:
                kwargs["output_attentions"] = True
            tokens = session.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            desc = {"kind": "text", "text": text, "tokens": tokens,
                    "token_ids": enc["input_ids"][0].tolist()}
            return None, kwargs, desc
        if hasattr(session.model, "encode"):   # demo char-LM
            ids = session.model.encode(text)
            desc = {"kind": "text", "text": text,
                    "tokens": list(text[: ids.shape[1]]),
                    "token_ids": ids[0].tolist()}
            return ids, None, desc
        raise ValueError("Text input needs a tokenizer (HF model) or encode().")

    shape = spec.get("shape") or session.meta.get("input_shape")
    if not shape:
        raise ValueError("No input shape given and none known for this model.")
    values = spec.get("values")
    if values is not None:
        x = torch.tensor(values, dtype=torch.float32).reshape(shape)
    else:
        fill = spec.get("fill", "randn")
        gen = torch.Generator().manual_seed(spec.get("seed", 0))
        if fill == "zeros":
            x = torch.zeros(shape)
        elif fill == "ones":
            x = torch.ones(shape)
        else:
            x = torch.randn(shape, generator=gen)
    return x, None, {"kind": "tensor", "shape": shape, "fill": spec.get("fill", "randn")}


def decode_topk(session, logits: torch.Tensor, k: int = 5):
    """Top-k next-token/class decode from final logits, for the last position."""
    if logits.dim() == 3:          # (batch, seq, vocab) — language model
        last = logits[0, -1]
    elif logits.dim() == 2:        # (batch, classes)
        last = logits[0]
    else:
        return None
    probs = F.softmax(last.float(), dim=-1)
    k = min(k, probs.numel())
    top = torch.topk(probs, k)
    entries = []
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        if session.load_kind == "hf" and session.tokenizer is not None:
            label = session.tokenizer.decode([i])
        elif hasattr(session.model, "decode"):
            label = session.model.decode([i])
        else:
            label = f"class {i}"
        entries.append({"id": i, "label": label, "prob": p})
    return {"topk": entries, "position": "last"}


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def resolve_target(session, trace: ForwardTrace, target_spec: dict):
    """Build (loss, description) from the user's target spec.

    kinds: "class" {index}, "token" {text or id}, "argmax" (self-target),
           "vector" {values}.
    """
    logits = first_tensor(trace.output)
    if logits is None:
        raise ValueError("Model produced no tensor output to compute a loss on.")
    spec = target_spec or {"kind": "argmax"}
    kind = spec.get("kind", "argmax")

    if logits.dim() == 3:      # LM: loss at last position
        last = logits[0, -1].unsqueeze(0)
        if kind == "token":
            if "id" in spec and spec["id"] is not None:
                tid = int(spec["id"])
            else:
                text = spec.get("text", "")
                if session.load_kind == "hf":
                    ids = session.tokenizer(text, add_special_tokens=False)["input_ids"]
                    if not ids:
                        raise ValueError(f"Target text {text!r} produced no tokens.")
                    tid = ids[0]
                elif hasattr(session.model, "encode"):
                    tid = session.model.encode(text)[0, 0].item()
                else:
                    raise ValueError("No tokenizer to resolve target token.")
        else:
            tid = last.argmax(dim=-1).item()
        target = torch.tensor([tid])
        loss = F.cross_entropy(last, target)
        label = (session.tokenizer.decode([tid]) if session.load_kind == "hf"
                 and session.tokenizer else
                 session.model.decode([tid]) if hasattr(session.model, "decode")
                 else str(tid))
        return loss, {"loss_fn": "cross_entropy", "target_kind": "next_token",
                      "target_id": tid, "target_label": label}

    if logits.dim() == 2:      # classifier
        if kind == "class" and spec.get("index") is not None:
            tid = int(spec["index"])
        else:
            tid = logits[0].argmax().item()
        loss = F.cross_entropy(logits, torch.tensor([tid]))
        return loss, {"loss_fn": "cross_entropy", "target_kind": "class",
                      "target_id": tid, "target_label": f"class {tid}"}

    # generic: MSE against user vector or zeros
    if kind == "vector" and spec.get("values") is not None:
        tgt = torch.tensor(spec["values"], dtype=torch.float32).reshape(logits.shape)
    else:
        tgt = torch.zeros_like(logits)
    loss = F.mse_loss(logits, tgt)
    return loss, {"loss_fn": "mse", "target_kind": "vector", "target_id": None,
                  "target_label": "user vector" if kind == "vector" else "zeros"}


def run_backward(session, input_spec: dict, target_spec: dict):
    """Fresh forward with grad capture, real loss, real backward.

    Returns (trace, result_dict). Gradients stay on the parameters so a
    subsequent optimizer step uses exactly these.
    """
    model = session.model
    model.zero_grad(set_to_none=True)
    example_input, input_kwargs, input_desc = make_input(session, input_spec)
    was_training = model.training
    trace = run_forward(model, example_input, input_kwargs, capture_grads=True)
    trace.input_desc = input_desc

    loss, loss_desc = resolve_target(session, trace, target_spec)
    loss.backward()
    if was_training != model.training:
        model.train(was_training)

    param_grads = {}
    layer_grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        param_grads[name] = summarize_tensor_light(p.grad, name=name)
        layer = name.rsplit(".", 1)[0]
        layer_grad_norms[layer] = (
            layer_grad_norms.get(layer, 0.0) ** 2 + p.grad.norm().item() ** 2
        ) ** 0.5

    return trace, {
        "loss": loss.item(),
        "loss_desc": loss_desc,
        "param_grads": param_grads,
        "layer_grad_norms": layer_grad_norms,
        "activation_grads": trace.act_grads,
    }


# ---------------------------------------------------------------------------
# Optimizer step + before/after diff
# ---------------------------------------------------------------------------

def optimizer_step(session, optimizer_name: str = "sgd", lr: float = 0.01):
    """Apply one real optimizer step using the gradients currently on the
    model, and report a per-parameter before/after diff."""
    model = session.model
    grads_present = any(p.grad is not None for p in model.parameters())
    if not grads_present:
        raise ValueError("No gradients present — run a backward pass first.")

    n_params = sum(p.numel() for p in model.parameters())
    snapshot = None
    if n_params <= SNAPSHOT_PARAM_LIMIT:
        snapshot = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer_name = "sgd"
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    opt.step()

    diffs = {}
    if snapshot is not None:
        session.pre_step_weights = snapshot
        for name, p in model.named_parameters():
            old = snapshot[name]
            delta = p.detach().cpu() - old
            dn = delta.norm().item()
            on = old.norm().item()
            diffs[name] = {
                "shape": list(p.shape),
                "update_norm": dn,
                "weight_norm_before": on,
                "weight_norm_after": p.detach().norm().item(),
                "relative_update": dn / on if on > 0 else None,
                "max_abs_change": delta.abs().max().item(),
                "grad_norm": p.grad.norm().item() if p.grad is not None else None,
            }
    return {
        "optimizer": optimizer_name,
        "lr": lr,
        "stepped": True,
        "snapshot_kept": snapshot is not None,
        "param_diffs": diffs,
    }


def undo_step(session):
    """Restore the pre-step snapshot."""
    if not getattr(session, "pre_step_weights", None):
        raise ValueError("No pre-step snapshot to restore.")
    with torch.no_grad():
        for name, p in session.model.named_parameters():
            p.copy_(session.pre_step_weights[name].to(p.device))
    session.pre_step_weights = None
    return {"restored": True}
