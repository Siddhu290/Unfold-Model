"""Phase O: minimal perturbation search — the dual of attribution.

Text: swap each input token for its nearest embedding-space neighbors (one
real light forward per candidate, streamed) and rank positions by how easily
a single plausible swap flips the answer. Vision: FGSM sweep using the same
gradient access Phase J built. The response cross-checks against the last
attribution: fragile positions should largely be the attributed ones.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .execution import make_input, run_forward
from .summarize import first_tensor


def _probs_for_ids(session, ids: torch.Tensor):
    model = session.model
    with torch.no_grad():
        if session.load_kind == "hf":
            out = model(input_ids=ids)
        else:
            out = model(ids)
    logits = first_tensor(out)
    return F.softmax(logits[0, -1].float(), dim=-1)


def _neighbors(session, tid: int, k: int = 8):
    """Nearest vocab tokens by embedding cosine (excluding the token itself)."""
    wte = session.model.get_input_embeddings() if hasattr(
        session.model, "get_input_embeddings") else None
    if wte is None:
        emb_mod = dict(session.model.named_modules()).get("tok_emb")
        if emb_mod is None:
            raise ValueError("No input embedding matrix found.")
        wte = emb_mod
    W = F.normalize(wte.weight.detach().float(), dim=1)
    sims = W @ W[tid]
    sims[tid] = -2
    vals, idxs = torch.topk(sims, k)
    return [(int(i), float(v)) for i, v in zip(idxs, vals)]


def _decode(session, tid):
    from .generation import _decode_one
    return _decode_one(session, int(tid))


def token_substitution_stream(session, input_spec: dict, k_neighbors: int = 6):
    """One real forward per (position, neighbor) candidate, streamed."""
    if not (session.tokenizer is not None or hasattr(session.model, "encode")):
        raise ValueError("Token substitution needs a text model.")
    ex, kw, desc = make_input(session, input_spec)
    ids = torch.tensor([desc["token_ids"]], dtype=torch.long)
    tokens = desc["tokens"]
    T = ids.shape[1]

    t0 = time.perf_counter()
    base = _probs_for_ids(session, ids)
    top1 = int(base.argmax())
    p_base = float(base[top1])
    per_fwd = time.perf_counter() - t0
    n_runs = T * k_neighbors
    yield {"event": "start", "n_positions": T, "k_neighbors": k_neighbors,
           "n_runs": n_runs, "estimate_s": round(per_fwd * n_runs, 1),
           "baseline_top1": _decode(session, top1), "p_top1": p_base}

    positions = []
    for pos in range(T):
        best = None
        for nid, sim in _neighbors(session, int(ids[0, pos]), k_neighbors):
            mutated = ids.clone()
            mutated[0, pos] = nid
            probs = _probs_for_ids(session, mutated)
            drop = p_base - float(probs[top1])
            flipped = int(probs.argmax()) != top1
            cand = {"token": _decode(session, nid), "similarity": sim,
                    "p_top1": float(probs[top1]), "drop": drop,
                    "flipped": flipped,
                    "new_top1": _decode(session, int(probs.argmax()))}
            if best is None or (cand["flipped"], cand["drop"]) > \
                    (best["flipped"], best["drop"]):
                best = cand
        entry = {"pos": pos, "token": tokens[pos], "fragility": best["drop"],
                 "flips": best["flipped"], "best_swap": best}
        positions.append(entry)
        yield {"event": "position", **entry}

    positions_ranked = sorted(positions, key=lambda x: (-x["flips"], -x["fragility"]))

    # cross-check vs the last attribution run on a matching prompt
    cross = None
    attr = getattr(session, "last_attribution", None)
    if attr and attr.get("kind") == "text" and \
            [s["token"] for s in attr["scores"]] == tokens:
        attr_top = {s["pos"] for s in
                    sorted(attr["scores"], key=lambda s: -abs(s["score"]))[:5]}
        frag_top = {p["pos"] for p in positions_ranked[:5]}
        overlap = len(attr_top & frag_top)
        cross = {
            "overlap_top5": overlap,
            "attribution_top5": sorted(attr_top),
            "fragility_top5": sorted(frag_top),
            "note": f"{overlap} of the top-5 attribution tokens are also in "
                    "the top-5 most fragile positions — attribution "
                    "(what mattered) and robustness (what flips it) largely "
                    "agree" if overlap >= 3 else
                    f"only {overlap}/5 overlap between attribution and "
                    "fragility — the two views disagree on this input, "
                    "worth investigating",
        }

    result = {"prompt": desc.get("text"), "baseline_top1": _decode(session, top1),
              "p_top1": p_base, "positions": positions_ranked,
              "cross_check": cross,
              "elapsed_s": round(time.perf_counter() - t0, 1)}
    log = getattr(session, "analysis_log", None)
    if log is None:
        log = session.analysis_log = []
    log.append({"kind": "robustness_text", "data": {
        "prompt": result["prompt"], "baseline_top1": result["baseline_top1"],
        "n_flips": sum(p["flips"] for p in positions),
        "most_fragile": positions_ranked[0], "cross_check": cross}})
    yield {"event": "done", **result}


def fgsm_sweep(session, input_spec: dict,
               epsilons=(0.01, 0.03, 0.07, 0.15, 0.3)) -> dict:
    """Vision robustness: x + ε·sign(∇x loss), real re-run per ε."""
    model = session.model
    ex, kw, desc = make_input(session, input_spec)
    if kw is not None or ex is None or ex.dtype not in (torch.float32, torch.float64):
        raise ValueError("FGSM sweep needs a float tensor input (vision model).")

    x = ex.clone().requires_grad_(True)
    tr = run_forward(model, x, None, light=True, detach_output=False)
    logits = first_tensor(tr.output)
    probs0 = F.softmax(logits[0].float(), dim=-1)
    top1 = int(probs0.argmax())
    loss = F.cross_entropy(logits, torch.tensor([top1]))
    (g,) = torch.autograd.grad(loss, x)
    direction = g.sign()

    curve = []
    for eps in epsilons:
        adv = ex + eps * direction
        with torch.no_grad():
            out = model(adv)
        p = F.softmax(first_tensor(out)[0].float(), dim=-1)
        curve.append({"epsilon": eps, "p_top1": float(p[top1]),
                      "flipped": int(p.argmax()) != top1,
                      "new_top1": f"class {int(p.argmax())}"})
    result = {"kind": "vision", "baseline_top1": f"class {top1}",
              "p_top1": float(probs0[top1]), "curve": curve}
    log = getattr(session, "analysis_log", None)
    if log is None:
        log = session.analysis_log = []
    log.append({"kind": "robustness_fgsm", "data": result})
    return result
