"""Phase K: session export — a self-contained interpretability write-up.

Pure aggregation: serializes what the session already computed (arch, last
forward, lens, patching, circuit, attribution, analysis log, training run,
profile) without re-running anything. Heatmaps are embedded as base64 PNGs
via a minimal pure-python encoder (no imaging dependency).
"""

from __future__ import annotations

import base64
import datetime
import struct
import zlib

from .extraction import flatten_tree
from .theory import get_theory


# ---------------------------------------------------------------------------
# minimal PNG encoder
# ---------------------------------------------------------------------------

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _rgb_png(pixels: list) -> bytes:
    """pixels: list of rows, each a list of (r, g, b)."""
    h, w = len(pixels), len(pixels[0])
    raw = b"".join(
        b"\x00" + b"".join(bytes(px) for px in row) for row in pixels)
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", zlib.compress(raw, 6))
            + _png_chunk(b"IEND", b""))


def _diverging(v, max_abs):
    t = max(-1.0, min(1.0, v / max_abs)) if max_abs > 0 else 0.0
    if t >= 0:
        return (int(30 + 225 * t), int(40 + 60 * t), int(50 + 30 * t))
    u = -t
    return (int(30 + 30 * u), int(40 + 100 * u), int(50 + 205 * u))


def heatmap_png_md(hm: dict, alt: str, scale: int = 5) -> str:
    """A summarize_tensor heatmap dict -> embedded markdown image."""
    if not hm or not hm.get("data"):
        return ""
    data = hm["data"]
    max_abs = max((abs(v) for row in data for v in row), default=1.0) or 1.0
    pixels = []
    for row in data:
        prow = []
        for v in row:
            prow.extend([_diverging(v, max_abs)] * scale)
        for _ in range(scale):
            pixels.append(prow)
    b64 = base64.b64encode(_rgb_png(pixels)).decode()
    return f"![{alt}](data:image/png;base64,{b64})"


def grid_png_md(matrix: list, alt: str, scale: int = 14) -> str:
    """layers×heads restoration matrix -> embedded image (None cells grey)."""
    vals = [v for row in matrix for v in row if v is not None]
    if not vals:
        return ""
    max_abs = max(abs(v) for v in vals) or 1.0
    pixels = []
    for row in matrix:
        prow = []
        for v in row:
            c = (60, 60, 60) if v is None else _diverging(v, max_abs)
            prow.extend([c] * scale)
        for _ in range(scale):
            pixels.append(prow)
    b64 = base64.b64encode(_rgb_png(pixels)).decode()
    return f"![{alt}](data:image/png;base64,{b64})"


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e5 else f"{x:.2e}"
    return str(x)


def _tok(label):
    return "`" + str(label).replace("`", "'") + "`"


def _theory_line(cls: str) -> str:
    t = get_theory(cls)
    return f"> **{t['title']}** — {t['what'].split('.')[0]}. Formula: `{t['formula']}`"


def build_report(session) -> str:
    L = []
    add = L.append
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    info = session.info()

    add(f"# Model X-Ray report — {info['source']}")
    add(f"*Generated {now} · session `{info['session_id']}` · "
        f"everything below was computed live on the real model — no estimates.*")

    # --- architecture ---
    add("\n## 1. Architecture")
    add(f"- **Model**: `{session.arch.get('root_class')}` ({info['source']})")
    add(f"- **Parameters**: {session.arch.get('total_params'):,} "
        f"({session.arch.get('trainable_params', 0):,} trainable)")
    add(f"- **Modules**: {session.arch.get('num_modules')}")
    tree = session.arch.get("tree", {})
    kids = tree.get("children", [])
    if kids:
        add("\n| top-level module | class | params |")
        add("|---|---|---|")
        for c in kids[:20]:
            add(f"| `{c['path']}` | {c['class']} | {c['total_param_count']:,} |")
    reps = []
    for node in flatten_tree(tree).values():
        for g in node.get("repeat_groups", []):
            reps.append(f"`{node['path'] or 'model'}` contains ×{g['count']} "
                        f"identical `{g['class']}` blocks")
    if reps:
        add("\nRepeated structure: " + "; ".join(reps) + ".")

    # --- forward result ---
    trace = session.last_trace
    if trace is not None and trace.llm:
        add("\n## 2. Forward pass result")
        d = trace.input_desc or {}
        if d.get("text"):
            add(f"Input prompt: **{d['text']!r}**")
        elif d.get("shape"):
            add(f"Input tensor: shape {d['shape']} ({d.get('fill', '?')})")
        add("\n| rank | prediction | probability |")
        add("|---|---|---|")
        for i, e in enumerate(trace.llm["topk"]):
            add(f"| {i + 1} | {_tok(e['label'])} | {e['prob'] * 100:.2f}% |")

    # --- logit lens ---
    lens = getattr(session, "last_lens", None)
    if lens:
        add("\n## 3. Logit lens — where the answer forms")
        add("Each block's hidden state, projected through the model's own "
            "final norm + unembedding: the token that layer *would* predict "
            "if the computation stopped there.")
        add("\n| depth | layer | top-1 | p |")
        add("|---|---|---|---|")
        for i, r in enumerate(lens["rows"]):
            t = r["topk"][0]
            add(f"| {i} | `{r['path']}` ({r['stage']}) | {_tok(t['label'])} "
                f"| {t['prob'] * 100:.1f}% |")
        if lens.get("final"):
            f = lens["final"][0]
            add(f"| — | **real output** | {_tok(f['label'])} | {f['prob'] * 100:.1f}% |")

    # --- attribution ---
    attr = getattr(session, "last_attribution", None)
    if attr:
        add("\n## 4. Input attribution — which input drove the output")
        add(f"Method: **{attr['method']}**"
            + (f" ({attr['steps']} interpolation steps)" if attr.get("steps") else "")
            + f", target {_tok(attr['target']['label'])}.")
        if attr["kind"] == "text":
            add("\n| token | attribution | share |")
            add("|---|---|---|")
            for row in attr["scores"]:
                add(f"| {_tok(row['token'])} | {_fmt(row['score'])} "
                    f"| {row['frac'] * 100:.1f}% |")
            if attr.get("completeness"):
                c = attr["completeness"]
                add(f"\nCompleteness check: Σ attributions = "
                    f"{_fmt(c['sum_attributions'])} vs f(x)−f(baseline) = "
                    f"{_fmt(c['difference'])}.")
        else:
            add("\n" + heatmap_png_md(attr["map"].get("heatmap"),
                                      "input attribution map", scale=8))

    # --- causal analysis ---
    patch = getattr(session, "last_patch", None)
    if patch:
        add("\n## 5. Activation patching (causal tracing)")
        add(f"Clean: **{patch['clean']['text']!r}** "
            f"(p({_tok(patch['target']['label'])}) = "
            f"{patch['clean']['p_target'] * 100:.2f}%) · Corrupted: "
            f"**{patch['corrupted']['text']!r}** "
            f"(p = {patch['corrupted']['p_target'] * 100:.2f}%)")
        add("\n| patched layer | restoration | flips back? |")
        add("|---|---|---|")
        for r in patch["results"]:
            if r.get("restoration") is None:
                continue
            add(f"| `{r['path']}` | {r['restoration'] * 100:.1f}% "
                f"| {'✔' if r.get('flipped_back') else ''} |")

    circuit = getattr(session, "last_circuit", None)
    if circuit:
        add("\n## 6. Discovered circuit (layer × head patching sweep)")
        add(f"{len(circuit['layers'])} layers × {circuit['n_heads']} heads, "
            f"{circuit['elapsed_s']}s of real patched re-runs. Rows = layers "
            "(execution order), columns = heads; brightness = restoration of "
            f"the clean answer {_tok(circuit['target']['label'])}.")
        add("\n" + grid_png_md(circuit["matrix"], "layer × head restoration"))
        strong = []
        for li, row in enumerate(circuit["matrix"]):
            for h, v in enumerate(row):
                if v is not None and abs(v) > 0.25:
                    strong.append((abs(v), f"`{circuit['layers'][li]}` head {h}: "
                                           f"{v * 100:.0f}%"))
        lc = [f"`{p}`: {v * 100:.0f}%" for p, v in
              zip(circuit["layers"], circuit["layer_curve"]) if v is not None
              and abs(v) > 0.5]
        if lc:
            add(f"\nWhole-layer restoration above 50%: {', '.join(lc)}.")
        if strong:
            strong.sort(reverse=True)
            add(f"Strongest individual heads: "
                + ", ".join(s for _, s in strong[:8]) + ".")

    # --- steering ---
    slog = getattr(session, "steering_log", None) or []
    if slog:
        add("\n## 6b. Activation steering")
        for s in slog:
            add(f"\nDirection from **{s['prompt_a']!r}** − **{s['prompt_b']!r}** "
                f"at `{s['layer']}` (‖d‖ = {_fmt(s['norm'])}), applied at "
                f"α = {s['alpha']} to unrelated prompts:")
            add("\n| prompt | top-1 before | top-1 after | KL shift |")
            add("|---|---|---|---|")
            for r in s["results"]:
                add(f"| {r['prompt']!r} | {_tok(r['top1_base'])} "
                    f"| {_tok(r['top1_steered'])} | {_fmt(r['kl'])} |")

    # --- SAE ---
    sae = getattr(session, "sae", None)
    if sae:
        m = sae["meta"]
        add("\n## 6c. Sparse-feature dictionary (toy SAE)")
        add(f"A {m['features']}-feature sparse autoencoder trained for "
            f"{m['steps']} steps on {m['rows']} activation rows from "
            f"`{sae['layer_path']}` (λ = {m['l1']}): {m['alive']} features "
            f"alive, mean L0 = {m['mean_l0']:.1f} active features per input. "
            "*Toy-scale dictionary learning — demonstrates the technique's "
            "mechanics, not publication-grade monosemantic features.*")

    # --- backward / update ---
    bw = session.last_backward
    if bw:
        add("\n## 7. Gradients")
        add(f"Loss = {_fmt(bw['loss'])} ({bw['loss_desc']['loss_fn']} vs "
            f"{_tok(bw['loss_desc']['target_label'])}). Largest gradient "
            "norms by layer:")
        top = sorted(bw["layer_grad_norms"].items(), key=lambda kv: -kv[1])[:8]
        add("\n| layer | ‖∇‖ |")
        add("|---|---|")
        for k, v in top:
            add(f"| `{k}` | {_fmt(v)} |")

    # --- training ---
    tr = getattr(session, "train_log", None)
    if tr:
        add("\n## 8. Training run")
        ls = tr["losses"]
        add(f"{tr['steps']} real {tr['optimizer'].upper()} steps (lr={tr['lr']}) "
            f"on the {tr['task']} task: loss {_fmt(ls[0])} → {_fmt(ls[-1])}. "
            f"Checkpoints kept at steps {tr['checkpoints']}.")
        n = max(1, len(ls) // 40)
        spark = "".join("▁▂▃▄▅▆▇█"[min(7, int(8 * v / (max(ls) or 1)))]
                        for v in ls[::n])
        add(f"\nLoss curve: `{spark}`")

    # --- analysis log ---
    log = getattr(session, "analysis_log", None) or []
    if log:
        add("\n## 9. Structural analyses")
        for item in log:
            k, d = item["kind"], item["data"]
            if k == "svd":
                add(f"- **SVD** `{d['path']}.{d['param']}` "
                    f"{d['matrix_shape']}: effective rank "
                    f"{d['effective_rank']:.1f} of {d['full_rank']} "
                    f"({d['rank_1pct']} svals above 1% of max; 90% of energy "
                    f"in {d['rank_90pct_energy']} directions).")
            elif k == "dead_neurons":
                add(f"- **Dead neurons** `{d['path']}`: {d['dead_count']} of "
                    f"{d['total_units']} units never fired across "
                    f"{d['n_inputs']} real inputs ({d['dead_frac'] * 100:.1f}%).")
            elif k == "quantize":
                add(f"- **int{d['bits']} simulation** `{d['path']}`: KL drift "
                    f"{_fmt(d['kl_divergence'])}, top-1 "
                    f"{'changed' if d['top1_changed'] else 'survives'} "
                    f"(p {_fmt(d['p_top1_before'])} → {_fmt(d['p_top1_after'])}).")
            elif k == "prune":
                pts = ", ".join(f"{c['fraction'] * 100:.0f}%→p="
                                f"{c['p_top1'] * 100:.1f}%" for c in d["curve"])
                add(f"- **Pruning sweep** `{d['path']}` (baseline "
                    f"p({_tok(d['baseline_top1'])}) = "
                    f"{d['p_top1_baseline'] * 100:.1f}%): {pts}.")
            elif k == "head_ablation":
                top = d["heads"][0]
                add(f"- **Head ablation** `{d['layer']}` ({d['n_heads']} heads): "
                    f"most important head {top['head']} "
                    f"(−{top['delta'] * 100:.2f}pp on "
                    f"p({_tok(d['baseline_top1'])})).")
            elif k == "aggregate_heads":
                top = d["heads"][0]
                add(f"- **Head importance across {d['n_prompts']} prompts** "
                    f"`{d['layer']}`: head {top['head']} leads with mean "
                    f"−{top['mean_delta'] * 100:.2f}pp, in the per-prompt "
                    f"top-3 {top['top3_frac'] * 100:.0f}% of the time"
                    + (" — consistently important (evidence, not an anecdote)."
                       if top["consistent"] else
                       " — NOT consistent across prompts; the single-example "
                       "finding may be an anecdote."))
            elif k == "aggregate_attribution":
                add(f"- **Attribution concentration across {d['n_prompts']} "
                    f"prompts**: top token holds {d['mean_top_frac'] * 100:.0f}% "
                    f"of attribution mass on average "
                    f"(range {d['min_top_frac'] * 100:.0f}–"
                    f"{d['max_top_frac'] * 100:.0f}%).")
            elif k == "robustness_text":
                mf = d["most_fragile"]
                add(f"- **Fragility scan** {d['prompt']!r}: {d['n_flips']} "
                    f"single-token swaps flip the answer; most fragile "
                    f"position is {_tok(mf['token'])}"
                    + (f". Cross-check: {d['cross_check']['note']}"
                       if d.get("cross_check") else "."))
            elif k == "robustness_fgsm":
                last = d["curve"][-1]
                add(f"- **FGSM sweep**: p({_tok(d['baseline_top1'])}) "
                    f"{d['p_top1'] * 100:.1f}% → {last['p_top1'] * 100:.1f}% "
                    f"at ε = {last['epsilon']}"
                    + (" (prediction flips)." if last["flipped"] else "."))

    # --- profile ---
    prof = None
    if trace is not None and any(r.get("duration_ms") for r in trace.records):
        try:
            from .profiling import profile
            prof = profile(session)
        except Exception:
            prof = None
    if prof:
        add("\n## 10. Performance profile")
        add(f"Total forward: {prof['total_ms']:.1f} ms (measured per call). "
            "Slowest leaf modules:")
        add("\n| layer | class | ms | % | params |")
        add("|---|---|---|---|---|")
        for r in [x for x in prof["rows"] if x["is_leaf"]][:6]:
            add(f"| `{r['path']}` | {r['class']} | {r['ms']:.2f} "
                f"| {r['pct_of_total']:.1f}% | {r['params']:,} |")

    # --- theory appendix ---
    classes = []
    if trace is not None:
        seen = set()
        for rec in trace.records:
            c = rec["class"]
            if c not in seen and get_theory(c)["key"] not in ("_generic",):
                seen.add(c)
                classes.append(c)
    if classes:
        add("\n## Appendix: operations in this model")
        for c in classes[:10]:
            add(_theory_line(c))
            add("")

    add("\n---")
    add("*Produced by Model X-Ray. All probabilities, gradients, restorations "
        "and timings above come from real forward/backward executions of the "
        "loaded model during this session.*")
    return "\n".join(L)
