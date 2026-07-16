# ⚡ Model X-Ray

A debugging/learning microscope for real neural networks. Point it at any
PyTorch model — an uploaded checkpoint, a HuggingFace model ID, or a built-in
demo — and see exactly what happens to a specific input, end to end:

1. **Architecture** — every module, its class, in/out shapes, parameter
   counts, and connectivity, as a collapsible tree. Repeated blocks (e.g. the
   24 identical layers of GPT-2 Medium) are auto-detected by structural
   signature and rendered as one expandable "×24" group.
2. **Forward pass** — a *real* `model(input)` call with a forward hook on
   every submodule. Step/play through the layers in true execution order
   (hook call order, so skip connections and reused modules are shown as they
   actually ran), with per-layer tensor shapes, stats, heatmaps, and — for
   language models — decoded top-5 next-token probabilities and per-layer,
   per-head attention heatmaps.
3. **Backward pass** — a real loss (cross-entropy against your chosen class /
   next token, or MSE) and a real `loss.backward()`. Per-layer gradient
   magnitudes (vanishing/exploding gradient diagnosis included), per-parameter
   and per-activation gradient tensors.
4. **Optimizer step** — pick SGD / SGD+momentum / Adam and a learning rate,
   apply one real `opt.step()`, and inspect a before/after/Δ diff of every
   weight tensor. Undo restores the exact pre-step weights.
5. **Theory panel** — a contextual explanation (formula, intuition, gradient
   behavior) for whatever layer or phase you're looking at, from an
   extensible registry covering Linear, Conv, Norms, activations, Embedding,
   attention (Q/K/V mechanics), Softmax, cross-entropy, backprop, and
   gradient descent/Adam.
6. **Graph view (⬡ tab)** — the model as an animated node-and-edge circuit
   diagram. Edge topology is NOT the module tree: it is recovered from the
   autograd graph of the last real execution, so skip connections render as
   a visibly separate edge rejoining the main path, Q/K/V projections appear
   as three parallel nodes fanning into the attention node, residual streams
   bow around the blocks they bypass, and re-entered modules (one ReLU called
   3×) show as dashed back-edges. One node per layer (never per neuron);
   repeated blocks collapse to a "×24" node whose expand/collapse state is
   shared with the tree. The same playback player drives tree and graph:
   forward pulses green along incoming edges, backward replays amber in
   reverse, and an optimizer step pulses nodes red scaled by ‖Δw‖. Node fill
   switches between activation / gradient / update intensity — the identical
   numbers shown in the other panels, never recomputed.

7. **Result strip & streaming generation** — the model's answer is pinned
   above everything: classifiers show prediction + confidence; language
   models generate token-by-token (greedy or sampled), each token clickable
   to re-run the hooked forward pass that produced it. "Show all
   probabilities" opens a virtualized list over the entire vocabulary.
8. **Logit lens** — every block's hidden state projected through the model's
   own final norm + unembedding (per-architecture registry, HF fallback),
   showing the answer crystallize across depth; also inline in the inspector
   for any selected layer.
9. **Activation patching (causal tracing)** — clean vs corrupted prompt,
   clean activations spliced into the corrupted run at the differing token
   positions, one block at a time; restoration % per layer shows which depth
   is causally responsible. (Sanity mode: full-position patches restore
   exactly 100%.)
10. **Architecture editing** — remove / swap-activation / duplicate (copy or
    random init, an explicit choice) / reorder, on the live model. Every
    edit is validated by a real forward (breaking edits auto-revert) and
    diffed side-by-side against the pre-edit model. Separate LIFO undo stack.
11. **Analysis panel** — SVD spectrum + effective rank, dead-neuron probe
    over a real input batch, int8/int4 quantization simulation, and a
    pruning sweep — all measured by real re-runs with weights restored after.
12. **Comparative tools** — per-head attention ablation ranking, max-activating
    input search, and whole-checkpoint model diffing (reuses the
    before/after/Δ heatmap view).
13. **Profiling** — measured per-module latency (perf_counter around every
    hook pair), exact parameter memory, FLOPs from recorded real shapes;
    sortable table plus a "time" coloring mode in the graph.

14. **Input attribution** — saliency (gradient × embedding) and integrated
    gradients (pad/EOS-embedding baseline, completeness axiom checked and
    reported), with an optional contrast token ("why Paris rather than
    London") that isolates the semantically responsible input tokens.
    Rendered as per-token highlighting or a pixel heatmap.
15. **Automated circuit discovery** — a streamed layer×head activation-
    patching sweep (whole-layer rows share Phase C's exact code path; head
    cells patch attention patterns individually), rendered as the standard
    layers×heads restoration grid with a threshold slider and a persistent
    teal "discovered circuit" overlay on the flow graph.
16. **Multi-step training** — the single-step machinery looped: real batches
    (toy quadrant task for vision demos, next-token CE on a corpus for LMs),
    streamed live loss curve, pause-and-inspect at any step, in-memory
    checkpoints you can scrub, restore, and diff through the existing
    before/after heatmap UI.
17. **Session report export** — one Markdown document (architecture, forward
    result, logit lens, attribution, patching, circuit grid as an embedded
    PNG, structural analyses, training run, profile, theory appendix)
    serialized from what the session already computed — nothing re-runs.

18. **Activation steering (🧭)** — contrastive concept vectors
    (activation(A) − activation(B) at a chosen block), added at α × direction
    during real forwards on unrelated prompts, with a live debounced α slider,
    watched-token probabilities, and a built-in generalization table. A
    sentiment direction from a movie-review pair measurably flips an
    unrelated restaurant prompt negative on GPT-2 Medium.
19. **Sparse autoencoder (🧬)** — real dictionary learning at honest toy
    scale: overcomplete untied SAE (unit-norm decoder columns, L1 code
    penalty, center-only normalization) trained on a layer's captured
    activation stream, then per-input sparse decomposition with
    max-activating provenance per feature. Recovery verified against planted
    ground truth (6/6 synthetic directions at ≥0.976 cosine; 4/4
    quadrant-selective features end-to-end).
20. **Dataset-scale aggregation (Σ)** — head importance and attribution
    concentration as distributions over prompt batches, streamed with
    estimates; batch-of-one is regression-tested to equal the single-example
    numbers exactly. "Consistent across N prompts" is evidence; one example
    is an anecdote — the UI says so.
21. **Robustness search (🧷)** — nearest-neighbor single-token substitution
    scan (which swap flips the answer, ranked by plausibility) and FGSM ε
    sweeps for vision, with an automatic cross-check against attribution
    (top-attributed tokens should be the fragile ones — 5/5 overlap on a
    planted-memorization model).

Nothing is simulated: the numbers on screen are the tensors PyTorch computed
for your input, captured by hooks on the live model.

## Quickstart

```bash
pip install -r requirements.txt
python backend/server.py --port 8321
# open http://127.0.0.1:8321
```

Load one of the demo models (MLP / CNN-with-skip / 4-layer character
transformer), paste a HuggingFace ID (e.g. `distilgpt2`, `gpt2-medium`), or
upload a `.safetensors` / `.pt` / `.pth` file.

## Security model for uploads

Pickled PyTorch files can execute arbitrary code on load. The loader:

1. Loads `.safetensors` natively — this format cannot execute code and is the
   recommended path for untrusted files.
2. Tries `torch.load(weights_only=True)` first for `.pt`/`.pth` — safe, and
   sufficient for state dicts.
3. Falls back to full unpickling **only** when the "trust pickle" opt-in is
   checked, and the session then carries a permanent warning.

Weights-only files (state dicts) have no architecture object, so they get an
inspection tree inferred from parameter names; forward/backward execution
requires a full model or a HuggingFace ID.

## Layout

```text
backend/
  server.py            FastAPI app — the frontend only ever sees JSON summaries
  xray/
    loading.py         safe checkpoint/HF loading (pickle policy lives here)
    extraction.py      module tree, shape probe, repeat-group detection
    execution.py       forward/backward hook capture, overrides (patching),
                       per-call timing, optimizer step + diff
    summarize.py       tensor -> JSON summary (stats, histogram, ≤64×64 heatmap)
    theory.py          explanation registry (register_theory to extend)
    generation.py      streaming token generation + full-vocab distribution
    lens.py            logit lens (final-projection registry per architecture)
    patching.py        causal tracing over clean/corrupted prompt pairs
    editing.py         live structural edits + validation + undo + compare
    analysis.py        SVD/rank, dead neurons, quantize & prune simulations
    comparative.py     head ablation, max-activating inputs, checkpoint diff
    profiling.py       per-layer latency/memory/FLOPs table
    attribution.py     saliency + integrated gradients (contrastive option)
    circuits.py        streamed layer×head patching sweep (circuit discovery)
    training.py        real training loop, in-memory checkpoints, scrubbing
    report.py          session -> Markdown write-up (pure-python PNG embeds)
    steering.py        contrastive concept vectors + additive steering
    sae.py             toy sparse autoencoder (dictionary learning) per layer
    aggregate.py       batch-scale head-importance / attribution distributions
    robustness.py      token-substitution fragility + FGSM sweeps
    session.py         in-memory session store
  tests/
    sample_models.py   demo MLP / CNN / tiny transformer
    test_extraction.py milestone-1 tests
    test_execution.py  forward/backward/optimizer tests
frontend/              self-contained vanilla-JS app (no build step, no CDN)
```

Run tests: `cd backend && for t in tests/test_*.py; do python "$t"; done`
(extraction, execution, topology, phases A+B / C / D / E+F+G / H / I / J / K).
Browser regression (server must be running): `python3.12 tests_ui/ui_test_core.py`
and `python3.12 tests_ui/ui_test_phase4.py`.

## Extending

* **New layer explanation**: `register_theory("MyLayer", {...}, aliases=[...])`
  in `theory.py` (or any imported module). Unknown classes fall back through
  aliases → substring match → a generic entry.
* **New demo model**: add a builder in `tests/sample_models.py::build_demo`.

## Scale notes

Tensor data is summarized server-side (stats + 40-bin histogram + ≤64×64
adaptive-pooled heatmap + ≤128-point preview); a 16.7M-element tensor crosses
the wire as ~93 KB. Verified against GPT-2 Medium (355M params, CPU): load
~2 min cold / seconds warm, forward trace of 319 module calls in ~3 s
(~900 KB payload), backward ~7 s, SGD step + full diff ~2 s. Activation
tensors above 4M elements are summarized but not retained for drill-down;
pre-step weight snapshots (diff/undo) are kept up to 500M parameters.

Out of scope for v1: multi-GPU/distributed models, quantized or compiled
graphs (TorchScript/ONNX), models larger than available RAM.
