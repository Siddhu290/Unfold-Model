"""Model X-Ray API server.

The frontend never touches PyTorch: everything crosses this API as
JSON-safe tensor summaries. Run with:
    python3.10 server.py [--port 8321]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

from xray.extraction import (
    apply_shape_records, extract_architecture, extract_from_state_dict,
)
from xray.loading import UnsafePickleError, load_any
from xray.execution import (
    decode_topk, make_input, optimizer_step, run_backward, run_forward, undo_step,
)
from xray.session import STORE, Session
from xray.summarize import summarize_tensor
from xray.theory import all_theory, get_theory
from sample_models import DEMO_MODELS, build_demo

app = FastAPI(title="Model X-Ray", version="0.1.0")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "model-xray-uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 8 * 1024 ** 3  # refuse absurd uploads outright


def _err(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


def _get_session(session_id: str) -> Session:
    try:
        return STORE.get(session_id)
    except KeyError:
        _err(404, f"Unknown session {session_id!r}")


def _apply_trace_to_arch(session, trace):
    """Fold a real execution's shapes/order back into the stored tree, and
    refresh the session's dataflow topology — a single source of truth for
    both the tree view and the graph view."""
    records = {}
    for r in trace.records:
        rec = records.setdefault(r["path"], {})
        if "call_order" not in rec:
            rec.update(call_order=r["call_index"], in_shape=r["in_shape"],
                       out_shape=r["out_shape"])
        rec["n_calls"] = rec.get("n_calls", 0) + 1
    apply_shape_records(session.arch["tree"], records)
    if trace.edges is not None:
        session.topology = {
            "edges": trace.edges,
            "calls": [{"call_index": r["call_index"], "path": r["path"],
                       "class": r["class"]} for r in trace.records],
        }


def _make_session_from_load(load, example_input=None, input_kwargs=None, meta=None):
    if load.kind == "state_dict":
        arch = extract_from_state_dict(load.state_dict)
    else:
        # one real forward (below) supplies shapes + topology; no separate probe
        arch = extract_architecture(load.model)
    session = Session(load, arch, meta)
    session.topology = None
    if session.runnable and (example_input is not None or input_kwargs):
        try:
            trace = run_forward(session.model, example_input, input_kwargs)
            session.last_trace = trace
            _apply_trace_to_arch(session, trace)
        except Exception as e:
            session.warnings.append(
                f"Shape/topology probe failed ({type(e).__name__}: {e}); "
                "shapes and the graph view appear after the first forward pass.")
    STORE.add(session)
    return session


class LoadRequest(BaseModel):
    demo: str | None = None
    hf_id: str | None = None
    path: str | None = None          # server-local file path
    allow_pickle: bool = False


@app.post("/api/load")
def api_load(req: LoadRequest):
    from xray.loading import LoadResult

    try:
        if req.demo:
            model, example, meta = build_demo(req.demo)
            load = LoadResult(kind="module", model=model, source=f"demo:{req.demo}")
            if req.demo == "tiny_transformer":
                load.kind = "module"
            session = _make_session_from_load(load, example, meta=meta)
        elif req.hf_id:
            load = load_any(req.hf_id, allow_pickle=req.allow_pickle)
            kwargs = None
            if load.tokenizer is not None:
                enc = load.tokenizer("Hello world", return_tensors="pt")
                kwargs = dict(enc)
            session = _make_session_from_load(load, input_kwargs=kwargs,
                                              meta={"input_kind": "text"})
        elif req.path:
            load = load_any(req.path, allow_pickle=req.allow_pickle)
            session = _make_session_from_load(load)
        else:
            _err(400, "Provide one of: demo, hf_id, path")
    except UnsafePickleError as e:
        _err(403, str(e))
    except HTTPException:
        raise
    except Exception as e:
        _err(400, f"{type(e).__name__}: {e}")

    return {"session": session.info(), "arch": session.arch,
            "topology": session.topology}


@app.post("/api/upload")
def api_upload(file: UploadFile = File(...), allow_pickle: bool = Form(False)):
    name = os.path.basename(file.filename or "upload.pt")
    if not name.endswith((".pt", ".pth", ".safetensors", ".bin", ".ckpt")):
        _err(400, "Supported extensions: .pt .pth .safetensors .bin .ckpt")
    dest = os.path.join(UPLOAD_DIR, name)
    size = 0
    with open(dest, "wb") as out:
        while chunk := file.file.read(16 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                os.unlink(dest)
                _err(413, "Upload exceeds 8 GB limit.")
            out.write(chunk)
    try:
        load = load_any(dest, allow_pickle=allow_pickle)
    except UnsafePickleError as e:
        _err(403, str(e))
    except Exception as e:
        _err(400, f"{type(e).__name__}: {e}")
    session = _make_session_from_load(load)
    return {"session": session.info(), "arch": session.arch,
            "topology": session.topology}


@app.get("/api/demos")
def api_demos():
    return DEMO_MODELS


@app.get("/api/sessions")
def api_sessions():
    return STORE.list()


@app.get("/api/session/{session_id}/arch")
def api_arch(session_id: str):
    return _get_session(session_id).arch


@app.get("/api/session/{session_id}/topology")
def api_topology(session_id: str):
    session = _get_session(session_id)
    if session.topology is None:
        _err(404, "No topology yet — run a forward pass first.")
    return session.topology


@app.get("/api/session/{session_id}/weight")
def api_weight(session_id: str, path: str = "", param: str = "weight"):
    session = _get_session(session_id)
    try:
        t = session.get_param_tensor(path, param)
    except KeyError as e:
        _err(404, str(e))
    return summarize_tensor(t, name=f"{path}.{param}" if path else param)


class ForwardRequest(BaseModel):
    input: dict
    topk: int = 5


@app.post("/api/session/{session_id}/forward")
def api_forward(session_id: str, req: ForwardRequest):
    session = _get_session(session_id)
    if not session.runnable:
        _err(400, "This session holds only a state dict (no architecture object); "
                  "forward execution is unavailable. Load a full model or HF ID.")
    try:
        example, kwargs, desc = make_input(session, req.input)
        trace = run_forward(session.model, example, kwargs)
    except Exception as e:
        _err(400, f"Forward failed — {type(e).__name__}: {e}")
    session.last_input_spec = req.input
    trace.input_desc = desc
    trace.llm = decode_topk(session, trace.output, k=req.topk)
    session.last_trace = trace
    _apply_trace_to_arch(session, trace)

    attn_meta = None
    if trace.attentions:
        attn_meta = {
            "num_layers": len(trace.attentions),
            "num_heads": trace.attentions[0].shape[1],
            "seq_len": trace.attentions[0].shape[-1],
        }
    return {
        "input": desc,
        "records": trace.records,
        "output": trace.output_summary,
        "llm": trace.llm,
        "attention": attn_meta,
        "edges": trace.edges,
    }


@app.get("/api/session/{session_id}/activation/{call_index}")
def api_activation(session_id: str, call_index: int):
    session = _get_session(session_id)
    trace = session.last_trace
    if trace is None:
        _err(400, "Run a forward pass first.")
    t = trace.tensors.get(call_index)
    if t is None:
        if call_index < len(trace.records):
            return {"detail_unavailable": True, **trace.records[call_index]}
        _err(404, f"No activation for call {call_index}")
    rec = trace.records[call_index]
    return {**summarize_tensor(t, name=rec["path"]), "call_index": call_index,
            "path": rec["path"], "class": rec["class"]}


@app.get("/api/session/{session_id}/attention")
def api_attention(session_id: str, layer: int = 0, head: int = 0):
    session = _get_session(session_id)
    trace = session.last_trace
    if trace is None or not trace.attentions:
        _err(400, "No attention maps captured. Run a forward pass on a "
                  "transformer loaded via HuggingFace.")
    if layer >= len(trace.attentions):
        _err(404, f"Layer {layer} out of range (0..{len(trace.attentions)-1})")
    a = trace.attentions[layer][0]
    if head >= a.shape[0]:
        _err(404, f"Head {head} out of range (0..{a.shape[0]-1})")
    tokens = (trace.input_desc or {}).get("tokens")
    return {
        "layer": layer, "head": head, "tokens": tokens,
        **summarize_tensor(a[head], name=f"attention L{layer} H{head}"),
    }


class BackwardRequest(BaseModel):
    input: dict
    target: dict = {}


@app.post("/api/session/{session_id}/backward")
def api_backward(session_id: str, req: BackwardRequest):
    session = _get_session(session_id)
    if not session.runnable:
        _err(400, "State-dict-only session: backward execution unavailable.")
    try:
        trace, result = run_backward(session, req.input, req.target)
    except Exception as e:
        _err(400, f"Backward failed — {type(e).__name__}: {e}")
    session.last_trace = trace
    session.last_backward = result
    _apply_trace_to_arch(session, trace)
    return result


@app.get("/api/session/{session_id}/grad")
def api_grad(session_id: str, name: str):
    session = _get_session(session_id)
    params = dict(session.model.named_parameters()) if session.model else {}
    p = params.get(name)
    if p is None or p.grad is None:
        _err(404, f"No gradient on {name!r} — run a backward pass first.")
    return summarize_tensor(p.grad, name=f"grad({name})")


class StepRequest(BaseModel):
    optimizer: str = "sgd"
    lr: float = 0.01


@app.post("/api/session/{session_id}/step")
def api_step(session_id: str, req: StepRequest):
    session = _get_session(session_id)
    if not session.runnable:
        _err(400, "State-dict-only session: optimizer step unavailable.")
    try:
        result = optimizer_step(session, req.optimizer, req.lr)
    except ValueError as e:
        _err(400, str(e))
    session.last_step = result
    return result


@app.get("/api/session/{session_id}/diff")
def api_diff(session_id: str, name: str):
    """Detailed before/after/delta view of one parameter after a step."""
    session = _get_session(session_id)
    if not session.pre_step_weights or name not in session.pre_step_weights:
        _err(404, "No pre-step snapshot for this parameter.")
    params = dict(session.model.named_parameters())
    if name not in params:
        _err(404, f"Unknown parameter {name!r}")
    old = session.pre_step_weights[name]
    new = params[name].detach().cpu()
    return {
        "name": name,
        "before": summarize_tensor(old, name=f"{name} (before)"),
        "after": summarize_tensor(new, name=f"{name} (after)"),
        "delta": summarize_tensor(new - old, name=f"{name} (delta)"),
    }


@app.post("/api/session/{session_id}/undo")
def api_undo(session_id: str):
    session = _get_session(session_id)
    try:
        return undo_step(session)
    except ValueError as e:
        _err(400, str(e))


@app.delete("/api/session/{session_id}")
def api_delete(session_id: str):
    STORE.remove(session_id)
    return {"deleted": session_id}


class GenerateRequest(BaseModel):
    input: dict
    max_new_tokens: int = 20
    mode: str = "greedy"          # greedy | sample
    temperature: float = 1.0


@app.post("/api/session/{session_id}/generate")
def api_generate(session_id: str, req: GenerateRequest):
    import json as _json
    from fastapi.responses import StreamingResponse
    from xray.generation import generate_stream

    session = _get_session(session_id)

    def stream():
        try:
            for item in generate_stream(session, req.input,
                                        max_new_tokens=min(req.max_new_tokens, 200),
                                        mode=req.mode, temperature=req.temperature):
                yield _json.dumps(item) + "\n"
        except Exception as e:
            yield _json.dumps({"event": "error",
                               "detail": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/session/{session_id}/distribution")
def api_distribution(session_id: str, offset: int = 0, limit: int = 500):
    from xray.generation import full_distribution

    try:
        return full_distribution(_get_session(session_id), offset, min(limit, 2000))
    except ValueError as e:
        _err(400, str(e))


@app.post("/api/session/{session_id}/logit_lens")
def api_logit_lens(session_id: str, k: int = 3):
    from xray.lens import logit_lens_strip

    try:
        session = _get_session(session_id)
        result = logit_lens_strip(session, k=k)
        session.last_lens = result
        return result
    except ValueError as e:
        _err(400, str(e))


@app.get("/api/session/{session_id}/logit_lens_one")
def api_logit_lens_one(session_id: str, call_index: int, k: int = 5):
    from xray.lens import lens_for_call

    try:
        return lens_for_call(_get_session(session_id), call_index, k=k)
    except ValueError as e:
        _err(400, str(e))


class PatchRequest(BaseModel):
    clean: dict
    corrupted: dict
    target: dict = {}
    positions: object = "diff"       # "diff" | "all" | "last" | [ints]
    layer_paths: list | None = None


@app.post("/api/session/{session_id}/patch")
def api_patch(session_id: str, req: PatchRequest):
    from xray.patching import causal_patch

    try:
        session = _get_session(session_id)
        result = causal_patch(session, req.clean, req.corrupted,
                              req.target, req.layer_paths, req.positions)
        session.last_patch = result
        return result
    except ValueError as e:
        _err(400, str(e))
    except Exception as e:
        _err(400, f"Patch failed — {type(e).__name__}: {e}")


def _log_analysis(session, kind: str, data: dict):
    """Append a finding to the session's analysis log (feeds the report)."""
    log = getattr(session, "analysis_log", None)
    if log is None:
        log = session.analysis_log = []
    log.append({"kind": kind, "data": data})
    del log[:-40]   # cap


def _refresh_after_edit(session):
    """Re-extract the tree and refresh shapes/topology from a real forward."""
    session.arch = extract_architecture(session.model)
    spec = getattr(session, "last_input_spec", None)
    try:
        from xray.editing import _default_input_spec
        spec = spec or _default_input_spec(session)
        example, kwargs, desc = make_input(session, spec)
        trace = run_forward(session.model, example, kwargs)
        trace.input_desc = desc
        session.last_trace = trace
        _apply_trace_to_arch(session, trace)
    except Exception:
        pass


class EditRequest(BaseModel):
    op: str
    path: str = ""
    to: str | None = None            # swap_activation target
    init: str = "copy"               # duplicate: "copy" | "random"
    direction: str = "down"          # reorder


@app.post("/api/session/{session_id}/edit")
def api_edit(session_id: str, req: EditRequest):
    from xray.editing import apply_edit, compare_outputs

    session = _get_session(session_id)
    try:
        result = apply_edit(session, req.dict())
    except ValueError as e:
        _err(400, str(e))
    _refresh_after_edit(session)
    compare = None
    try:
        compare = compare_outputs(session)
    except ValueError as e:
        result["warnings"].append(str(e))
    return {**result, "arch": session.arch, "topology": session.topology,
            "compare": compare}


@app.post("/api/session/{session_id}/edit_undo")
def api_edit_undo(session_id: str):
    from xray.editing import undo_edit

    session = _get_session(session_id)
    try:
        result = undo_edit(session)
    except ValueError as e:
        _err(400, str(e))
    _refresh_after_edit(session)
    return {**result, "arch": session.arch, "topology": session.topology}


# ---- phase J: input attribution ----

class AttributionRequest(BaseModel):
    input: dict
    target: dict = {}
    method: str = "saliency"        # saliency | ig
    steps: int = 16
    contrast: str | None = None     # "why target rather than THIS token"


@app.post("/api/session/{session_id}/attribution")
def api_attribution(session_id: str, req: AttributionRequest):
    from xray.attribution import attribute

    session = _get_session(session_id)
    try:
        result = attribute(session, req.input, req.target, req.method,
                           req.steps, req.contrast)
    except ValueError as e:
        _err(400, str(e))
    except Exception as e:
        _err(400, f"Attribution failed — {type(e).__name__}: {e}")
    session.last_attribution = result
    return result


# ---- phase H: circuit discovery ----

@app.post("/api/session/{session_id}/circuit")
def api_circuit(session_id: str, req: PatchRequest):
    import json as _json
    from fastapi.responses import StreamingResponse
    from xray.circuits import circuit_sweep_stream

    session = _get_session(session_id)

    def stream():
        try:
            for item in circuit_sweep_stream(session, req.clean, req.corrupted,
                                             req.target):
                yield _json.dumps(item) + "\n"
        except Exception as e:
            yield _json.dumps({"event": "error",
                               "detail": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---- phase I: training loop ----

class TrainRequest(BaseModel):
    steps: int = 50
    optimizer: str = "sgd"
    lr: float = 0.01
    source: dict = {}
    checkpoint_every: int = 10


@app.post("/api/session/{session_id}/train")
def api_train(session_id: str, req: TrainRequest):
    import json as _json
    from fastapi.responses import StreamingResponse
    from xray.training import train_stream

    session = _get_session(session_id)

    def stream():
        try:
            for item in train_stream(session, req.steps, req.optimizer, req.lr,
                                     req.source, req.checkpoint_every):
                yield _json.dumps(item) + "\n"
        except Exception as e:
            yield _json.dumps({"event": "error",
                               "detail": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/session/{session_id}/train_restore")
def api_train_restore(session_id: str, step: int):
    from xray.training import restore_checkpoint
    try:
        result = restore_checkpoint(_get_session(session_id), step)
    except ValueError as e:
        _err(400, str(e))
    _refresh_after_edit(_get_session(session_id))
    return result


@app.post("/api/session/{session_id}/train_diff")
def api_train_diff(session_id: str, step: int):
    from xray.training import checkpoint_diff
    try:
        return checkpoint_diff(_get_session(session_id), step)
    except ValueError as e:
        _err(400, str(e))


# ---- phase L: activation steering ----

class DirectionRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    layer: str
    position: str = "last"


@app.post("/api/session/{session_id}/steer_direction")
def api_steer_direction(session_id: str, req: DirectionRequest):
    from xray.steering import build_direction
    try:
        return build_direction(_get_session(session_id), req.prompt_a,
                               req.prompt_b, req.layer, req.position)
    except ValueError as e:
        _err(400, str(e))


class SteerRequest(BaseModel):
    input: dict
    alpha: float = 0.0
    positions: str = "all"          # all | last
    watch: list | None = None       # tokens whose probability to track


@app.post("/api/session/{session_id}/steer")
def api_steer(session_id: str, req: SteerRequest):
    from xray.steering import steer
    try:
        return steer(_get_session(session_id), req.input, req.alpha,
                     positions=req.positions, watch=req.watch)
    except ValueError as e:
        _err(400, str(e))


class SteerBatchRequest(BaseModel):
    prompts: list
    alpha: float
    positions: str = "all"
    watch: list | None = None


@app.post("/api/session/{session_id}/steer_batch")
def api_steer_batch(session_id: str, req: SteerBatchRequest):
    from xray.steering import steer_batch
    try:
        return steer_batch(_get_session(session_id), req.prompts, req.alpha,
                           positions=req.positions, watch=req.watch)
    except ValueError as e:
        _err(400, str(e))


# ---- phase M: sparse autoencoder ----

class SAETrainRequest(BaseModel):
    layer: str
    source: dict = {}
    expansion: int = 4
    l1: float = 0.001
    steps: int = 300


@app.post("/api/session/{session_id}/sae_train")
def api_sae_train(session_id: str, req: SAETrainRequest):
    import json as _json
    from fastapi.responses import StreamingResponse
    from xray.sae import train_sae_stream

    session = _get_session(session_id)

    def stream():
        try:
            for item in train_sae_stream(session, req.layer, req.source,
                                         req.expansion, req.l1, req.steps):
                yield _json.dumps(item) + "\n"
        except Exception as e:
            yield _json.dumps({"event": "error",
                               "detail": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


class SAEDecomposeRequest(BaseModel):
    input: dict
    position: str = "last"


@app.post("/api/session/{session_id}/sae_decompose")
def api_sae_decompose(session_id: str, req: SAEDecomposeRequest):
    from xray.sae import decompose
    try:
        return decompose(_get_session(session_id), req.input, req.position)
    except ValueError as e:
        _err(400, str(e))


# ---- phase N: dataset-scale aggregation ----

class AggregateRequest(BaseModel):
    analysis: str                    # head_ablation | attribution
    layer: str = ""
    prompts: list | None = None
    method: str = "saliency"


@app.post("/api/session/{session_id}/aggregate")
def api_aggregate(session_id: str, req: AggregateRequest):
    import json as _json
    from fastapi.responses import StreamingResponse
    from xray.aggregate import aggregate_attribution_stream, aggregate_heads_stream

    session = _get_session(session_id)

    def stream():
        try:
            if req.analysis == "head_ablation":
                gen = aggregate_heads_stream(session, req.layer, req.prompts)
            elif req.analysis == "attribution":
                gen = aggregate_attribution_stream(session, req.prompts, req.method)
            else:
                raise ValueError(f"Unknown aggregate analysis {req.analysis!r}")
            for item in gen:
                yield _json.dumps(item) + "\n"
        except Exception as e:
            yield _json.dumps({"event": "error",
                               "detail": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---- phase O: robustness ----

class RobustnessRequest(BaseModel):
    input: dict
    k_neighbors: int = 6


@app.post("/api/session/{session_id}/robustness")
def api_robustness(session_id: str, req: RobustnessRequest):
    import json as _json
    from fastapi.responses import StreamingResponse
    from xray.robustness import token_substitution_stream

    session = _get_session(session_id)

    def stream():
        try:
            for item in token_substitution_stream(session, req.input,
                                                  min(req.k_neighbors, 12)):
                yield _json.dumps(item) + "\n"
        except Exception as e:
            yield _json.dumps({"event": "error",
                               "detail": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/session/{session_id}/fgsm")
def api_fgsm(session_id: str, req: RobustnessRequest):
    from xray.robustness import fgsm_sweep
    try:
        return fgsm_sweep(_get_session(session_id), req.input)
    except ValueError as e:
        _err(400, str(e))


# ---- phase E: analysis ----

@app.get("/api/session/{session_id}/svd")
def api_svd(session_id: str, path: str, param: str = "weight"):
    from xray.analysis import weight_svd
    try:
        session = _get_session(session_id)
        result = weight_svd(session, path, param)
        _log_analysis(session, "svd", result)
        return result
    except (ValueError, KeyError) as e:
        _err(400, str(e))


@app.post("/api/session/{session_id}/dead_neurons")
def api_dead(session_id: str, path: str, n_inputs: int = 32):
    from xray.analysis import dead_neurons
    try:
        session = _get_session(session_id)
        result = dead_neurons(session, path, min(n_inputs, 128))
        _log_analysis(session, "dead_neurons", result)
        return result
    except ValueError as e:
        _err(400, str(e))


@app.post("/api/session/{session_id}/quantize_sim")
def api_quantize(session_id: str, path: str, bits: int = 8):
    from xray.analysis import quantize_sim
    try:
        session = _get_session(session_id)
        result = quantize_sim(session, path, bits)
        _log_analysis(session, "quantize", result)
        return result
    except ValueError as e:
        _err(400, str(e))


@app.post("/api/session/{session_id}/prune_sim")
def api_prune(session_id: str, path: str):
    from xray.analysis import prune_sim
    try:
        session = _get_session(session_id)
        result = prune_sim(session, path)
        _log_analysis(session, "prune", result)
        return result
    except ValueError as e:
        _err(400, str(e))


# ---- phase F: comparative ----

class AblateRequest(BaseModel):
    input: dict
    layer: str


@app.post("/api/session/{session_id}/ablate_heads")
def api_ablate(session_id: str, req: AblateRequest):
    from xray.comparative import ablate_heads
    try:
        session = _get_session(session_id)
        result = ablate_heads(session, req.input, req.layer)
        _log_analysis(session, "head_ablation", result)
        return result
    except ValueError as e:
        _err(400, str(e))


class MaxActRequest(BaseModel):
    path: str
    texts: list | None = None
    neuron: int | None = None


@app.post("/api/session/{session_id}/max_activating")
def api_max_act(session_id: str, req: MaxActRequest):
    from xray.comparative import max_activating
    try:
        return max_activating(_get_session(session_id), req.path,
                              req.texts, neuron=req.neuron)
    except ValueError as e:
        _err(400, str(e))


class ModelDiffRequest(BaseModel):
    ref: str                        # HF id or server-local path
    allow_pickle: bool = False


@app.post("/api/session/{session_id}/diff_model")
def api_diff_model(session_id: str, req: ModelDiffRequest):
    from xray.comparative import diff_against
    try:
        return diff_against(_get_session(session_id), req.ref, req.allow_pickle)
    except UnsafePickleError as e:
        _err(403, str(e))
    except Exception as e:
        _err(400, f"{type(e).__name__}: {e}")


@app.get("/api/session/{session_id}/diff_model_param")
def api_diff_model_param(session_id: str, name: str):
    from xray.comparative import diff_against_detail
    try:
        return diff_against_detail(_get_session(session_id), name)
    except ValueError as e:
        _err(400, str(e))


# ---- phase G: profiling ----

@app.get("/api/session/{session_id}/profile")
def api_profile(session_id: str):
    from xray.profiling import profile
    try:
        return profile(_get_session(session_id))
    except ValueError as e:
        _err(400, str(e))


# ---- phase K: report export ----

@app.get("/api/session/{session_id}/report.md")
def api_report(session_id: str):
    from fastapi.responses import PlainTextResponse
    from xray.report import build_report

    session = _get_session(session_id)
    md = build_report(session)
    fname = f"model-xray-{session.source.replace('/', '_')}-{session.id}.md"
    return PlainTextResponse(md, media_type="text/markdown", headers={
        "Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/api/theory")
def api_theory_all():
    return all_theory()


@app.get("/api/theory/{class_name}")
def api_theory(class_name: str):
    return get_theory(class_name)


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    uvicorn.run(app, host=args.host, port=args.port)
