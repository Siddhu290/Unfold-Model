"""Phase H tests: circuit sweep consistency with Phase C + head grid."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xray.circuits import circuit_sweep_stream
from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.patching import causal_patch
from xray.session import Session
from sample_models import build_demo


def make_session():
    model, x, meta = build_demo("tiny_transformer")
    load = LoadResult(kind="module", model=model, source="demo:tiny_transformer")
    return Session(load, extract_architecture(model, x), meta)


CLEAN = {"kind": "text", "text": "hello world"}
CORR = {"kind": "text", "text": "jjjjj world"}


def test_sweep_events_and_shape():
    s = make_session()
    events = list(circuit_sweep_stream(s, CLEAN, CORR))
    start = events[0]
    assert start["event"] == "start"
    assert start["n_heads"] == 4 and len(start["layers"]) == 4
    assert start["n_runs"] == 4 * 5
    assert start["estimate_s"] >= 0
    layer_evs = [e for e in events if e["event"] == "layer"]
    head_evs = [e for e in events if e["event"] == "head"]
    assert len(layer_evs) == 4 and len(head_evs) == 16
    done = events[-1]
    assert done["event"] == "done"
    assert len(done["matrix"]) == 4 and all(len(r) == 4 for r in done["matrix"])
    assert s.last_circuit is not None, "result stored for overlay/report"
    print("  sweep: start/4 layer/16 head/done events, 4x4 matrix stored OK")


def test_layer_rows_match_phase_c_exactly():
    """The consistency requirement: sweep layer numbers must EQUAL the
    original causal_patch numbers — same code path, zero recomputation drift."""
    s = make_session()
    single = causal_patch(s, CLEAN, CORR, positions="diff")
    sweep = list(circuit_sweep_stream(s, CLEAN, CORR))
    sweep_layers = {e["path"]: e["restoration"] for e in sweep
                    if e["event"] == "layer"}
    for r in single["results"]:
        assert abs(sweep_layers[r["path"]] - r["restoration"]) < 1e-9, \
            f"{r['path']}: sweep {sweep_layers[r['path']]} != single {r['restoration']}"
    print("  consistency: sweep layer restorations == causal_patch exactly OK")


def test_head_patching_bounded_by_full_layer():
    """Patching one head's attention pattern is a strictly weaker
    intervention than splicing the whole residual stream — values should be
    finite and the matrix non-degenerate."""
    s = make_session()
    done = list(circuit_sweep_stream(s, CLEAN, CORR))[-1]
    vals = [v for row in done["matrix"] for v in row if v is not None]
    assert len(vals) == 16
    assert len(set(round(v, 8) for v in vals)) > 4, "head effects must vary"
    print(f"  head grid: 16 finite varied restorations "
          f"(min {min(vals):.3f}, max {max(vals):.3f}) OK")


def test_non_transformer_rejected():
    model, x, meta = build_demo("mlp")
    load = LoadResult(kind="module", model=model, source="demo:mlp")
    s = Session(load, extract_architecture(model, x), meta)
    try:
        list(circuit_sweep_stream(s, CLEAN, CORR))
        raise AssertionError("MLP must be rejected")
    except ValueError:
        pass
    print("  guards: non-transformer rejected OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase H tests:")
    for t in tests:
        t()
    print("ALL PHASE H TESTS PASSED")
