"""Phase G: per-layer profiling from the real hooked forward pass.

Latency is MEASURED (perf_counter pairs around every module call, captured
by execution.py's pre/post hooks). Parameter memory is exact. FLOPs are
computed from the recorded real shapes with the standard per-op formulas —
the one column that is arithmetic rather than measurement, marked as such.
"""

from __future__ import annotations

from .extraction import flatten_tree


def _flops(cls: str, in_shape, out_shape, node) -> float | None:
    """Standard MAC-based FLOPs (2 * MACs) from the REAL recorded shapes."""
    if not out_shape:
        return None
    def numel(s):
        n = 1
        for d in s:
            n *= d
        return n
    params = {p["name"]: p["shape"] for p in (node.get("params") or [])}
    w = params.get("weight")
    if cls in ("Linear", "LazyLinear", "Conv1D") and w:
        in_f = w[1] if cls != "Conv1D" else w[0]
        return 2.0 * numel(out_shape) * in_f
    if cls.startswith("Conv") and w:
        # weight (Cout, Cin/groups, *k): each output element costs
        # 2 * Cin/groups * prod(k)
        per_out = 2.0
        for d in w[1:]:
            per_out *= d
        return numel(out_shape) * per_out
    if cls == "Embedding":
        return 0.0                      # lookup, no arithmetic
    if "Norm" in cls:
        return 5.0 * numel(out_shape)   # mean, var, normalize, scale, shift
    if cls in ("ReLU", "GELU", "Tanh", "Sigmoid", "SiLU", "Softmax",
               "NewGELUActivation", "GELUActivation", "Identity", "Dropout"):
        return float(numel(out_shape))
    return None


def profile(session) -> dict:
    """Aggregate measured per-call timings into a per-layer table."""
    trace = session.last_trace
    if trace is None:
        raise ValueError("Run a forward pass first.")
    nodes = flatten_tree(session.arch["tree"])

    agg = {}
    for rec in trace.records:
        if rec.get("duration_ms") is None:
            continue
        a = agg.setdefault(rec["path"], {"calls": 0, "ms": 0.0})
        a["calls"] += 1
        a["ms"] += rec["duration_ms"]

    root_ms = agg.get("", {}).get("ms") or sum(
        v["ms"] for k, v in agg.items() if k and "." not in k) or 1e-9

    rows = []
    for path, a in agg.items():
        node = nodes.get(path)
        if node is None:
            continue
        rec = next(r for r in trace.records if r["path"] == path)
        own = node.get("own_param_count") or 0
        is_leaf = node.get("is_leaf", False)
        flops = _flops(node["class"], rec.get("in_shape"), rec.get("out_shape"), node)
        rows.append({
            "path": path or "(model)",
            "class": node["class"],
            "is_leaf": is_leaf,
            "calls": a["calls"],
            "ms": a["ms"],
            "pct_of_total": 100.0 * a["ms"] / root_ms if path else 100.0,
            "params": own,
            "param_bytes": own * 4,       # fp32
            "flops": (flops * a["calls"]) if flops is not None else None,
        })
    rows.sort(key=lambda r: -r["ms"])
    total_leaf_ms = sum(r["ms"] for r in rows if r["is_leaf"])
    return {"total_ms": root_ms, "leaf_ms": total_leaf_ms, "rows": rows,
            "note": "latency measured per call (perf_counter around each hook "
                    "pair, CPU); FLOPs computed from the recorded real shapes"}
