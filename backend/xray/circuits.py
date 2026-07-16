"""Phase H: automated circuit discovery.

Sweeps activation patching over every block (identical code path to Phase
C's causal_patch, via PatchContext) AND every attention head individually
(the clean head's attention pattern spliced into the corrupted run). Streams
progress as NDJSON and stores the finished layers×heads restoration matrix
on the session for the graph overlay and the report.
"""

from __future__ import annotations

import time

from .comparative import _attn_probs_call
from .lens import _block_paths
from .patching import PatchContext


def _head_carrier(ctx, layer_path):
    """The (B, H, T, T) attention-probs call inside a layer — same call index
    in the clean and corrupted traces (identical execution structure)."""
    ci, clean_attn = _attn_probs_call(ctx.clean_trace, layer_path)
    corr_attn = ctx.corr_trace.tensors.get(ci)
    if corr_attn is None or corr_attn.shape != clean_attn.shape:
        raise ValueError(f"attention shapes differ at {layer_path}")
    return ci, clean_attn, corr_attn


def circuit_sweep_stream(session, clean_spec: dict, corrupted_spec: dict,
                         target_spec: dict = None):
    """Yield NDJSON events for a full layer×head patching sweep."""
    ctx = PatchContext(session, clean_spec, corrupted_spec, target_spec,
                       positions="diff")
    layer_paths = _block_paths(session.arch)
    if not layer_paths:
        raise ValueError("No repeated blocks found — circuit discovery needs "
                         "a transformer-style model.")
    # order by execution
    layer_paths.sort(key=lambda p: ctx.first_call.get(p, 1 << 30))

    n_heads = 0
    try:
        _, attn0, _ = _head_carrier(ctx, layer_paths[0])
        n_heads = attn0.shape[1]
    except ValueError:
        pass

    n_runs = len(layer_paths) * (1 + n_heads)
    t_start = time.perf_counter()
    yield {
        "event": "start",
        "layers": layer_paths,
        "n_heads": n_heads,
        "n_runs": n_runs,
        "estimate_s": round(ctx.fwd_seconds * n_runs, 1),
        "diff_positions": ctx.diff_positions,
        **ctx.summary(),
    }

    layer_curve = []
    matrix = []          # [layer][head] restoration
    for li, path in enumerate(layer_paths):
        # whole-layer row: EXACTLY Phase C's computation
        entry = ctx.patch_layer(path)
        r = entry.get("restoration")
        layer_curve.append(r)
        yield {"event": "layer", "i": li, "path": path, "restoration": r,
               "p_target": entry.get("p_target"),
               "flipped_back": entry.get("flipped_back", False)}

        row = []
        if n_heads:
            try:
                ci, clean_attn, corr_attn = _head_carrier(ctx, path)
            except ValueError:
                row = [None] * n_heads
                matrix.append(row)
                continue
            for h in range(n_heads):
                patched = corr_attn.clone()
                patched[:, h] = clean_attn[:, h]
                res = ctx.run_override(ci, patched)
                row.append(res["restoration"])
                yield {"event": "head", "layer_i": li, "path": path, "head": h,
                       "restoration": res["restoration"]}
        matrix.append(row)

    elapsed = time.perf_counter() - t_start
    result = {
        "layers": layer_paths,
        "layer_curve": layer_curve,
        "n_heads": n_heads,
        "matrix": matrix,
        "elapsed_s": round(elapsed, 1),
        **ctx.summary(),
    }
    session.last_circuit = result
    yield {"event": "done", **result}
