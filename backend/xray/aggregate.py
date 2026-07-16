"""Phase N: dataset-scale analysis — distributions, not anecdotes.

Loops the EXISTING single-example analyses (comparative.ablate_heads,
attribution.attribute) over a batch of prompts and aggregates. With a batch
of one, the aggregate equals the single-example number exactly — that
regression is tested. Streams NDJSON with an upfront estimate, matching the
circuit-sweep pattern.
"""

from __future__ import annotations

import time

from .attribution import attribute
from .comparative import ablate_heads

FACTUAL_PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "The Colosseum is located in the city of",
    "The Statue of Liberty is located in the city of",
    "The Kremlin is located in the city of",
    "Big Ben is located in the city of",
    "The Golden Gate Bridge is located in the city of",
    "The Brandenburg Gate is located in the city of",
    "The Sydney Opera House is located in the city of",
]


def default_prompts(session):
    return FACTUAL_PROMPTS if session.load_kind == "hf" else [
        "hello world", "the quick brown", "abc abc abc", "xyz xyz",
        "aaaa bbbb cccc", "one two three",
    ]


def aggregate_heads_stream(session, layer: str, prompts: list = None):
    """Per-head importance averaged over a batch: 'consistently important
    across N prompts' is evidence; one example is an anecdote."""
    prompts = (prompts or default_prompts(session))[:24]
    t0 = time.perf_counter()
    first = ablate_heads(session, {"kind": "text", "text": prompts[0]}, layer)
    per_prompt = [first]
    n_heads = first["n_heads"]
    est = (time.perf_counter() - t0) * len(prompts)
    yield {"event": "start", "analysis": "head_ablation", "layer": layer,
           "n_prompts": len(prompts), "n_heads": n_heads,
           "estimate_s": round(est, 1)}
    yield {"event": "prompt", "i": 0, "prompt": prompts[0],
           "top_head": first["heads"][0]["head"]}

    for i, p in enumerate(prompts[1:], start=1):
        r = ablate_heads(session, {"kind": "text", "text": p}, layer)
        per_prompt.append(r)
        yield {"event": "prompt", "i": i, "prompt": p,
               "top_head": r["heads"][0]["head"]}

    heads = []
    for h in range(n_heads):
        deltas = []
        top3 = 0
        for r in per_prompt:
            d = next(x["delta"] for x in r["heads"] if x["head"] == h)
            deltas.append(d)
            ranked = [x["head"] for x in r["heads"][:3]]
            top3 += h in ranked
        mean = sum(deltas) / len(deltas)
        heads.append({
            "head": h, "mean_delta": mean,
            "min_delta": min(deltas), "max_delta": max(deltas),
            "top3_frac": top3 / len(per_prompt),
            "consistent": top3 / len(per_prompt) >= 0.5,
        })
    heads.sort(key=lambda x: -x["mean_delta"])
    result = {"analysis": "head_ablation", "layer": layer,
              "n_prompts": len(prompts), "prompts": prompts,
              "heads": heads,
              "elapsed_s": round(time.perf_counter() - t0, 1)}
    log = getattr(session, "analysis_log", None)
    if log is None:
        log = session.analysis_log = []
    log.append({"kind": "aggregate_heads", "data": result})
    yield {"event": "done", **result}


def aggregate_attribution_stream(session, prompts: list = None,
                                 method: str = "saliency"):
    """Distribution of attribution concentration across a batch of prompts."""
    prompts = (prompts or default_prompts(session))[:24]
    t0 = time.perf_counter()
    yield {"event": "start", "analysis": "attribution",
           "n_prompts": len(prompts), "method": method,
           "estimate_s": None}
    rows = []
    for i, p in enumerate(prompts):
        r = attribute(session, {"kind": "text", "text": p}, method=method)
        fracs = sorted((s["frac"] for s in r["scores"]), reverse=True)
        top_tok = max(r["scores"], key=lambda s: abs(s["score"]))
        entry = {
            "prompt": p, "n_tokens": len(r["scores"]),
            "top_token": top_tok["token"], "top_frac": fracs[0],
            "top2_frac": sum(fracs[:2]),
            "target": r["target"]["label"],
        }
        rows.append(entry)
        yield {"event": "prompt", "i": i, **entry}
    tf = [r["top_frac"] for r in rows]
    result = {
        "analysis": "attribution", "method": method,
        "n_prompts": len(rows), "rows": rows,
        "mean_top_frac": sum(tf) / len(tf),
        "min_top_frac": min(tf), "max_top_frac": max(tf),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    log = getattr(session, "analysis_log", None)
    if log is None:
        log = session.analysis_log = []
    log.append({"kind": "aggregate_attribution", "data": result})
    yield {"event": "done", **result}
