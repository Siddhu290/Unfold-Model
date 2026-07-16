"""Phase I: multi-step training — the single-step machinery, looped.

Each step is a real batch -> forward -> cross-entropy -> backward ->
optimizer.step() on the live model, streamed as NDJSON. Pausing the stream
leaves the weights exactly at the last completed step, so the existing
Forward/Backward/Update tabs inspect that precise moment. Checkpoints are
in-memory state-dict clones that feed the existing diff UI.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CHECKPOINT_PARAM_LIMIT = 60_000_000
MAX_STEPS = 500
DEFAULT_CORPUS = ("The quick brown fox jumps over the lazy dog. "
                  "Pack my box with five dozen liquor jugs. ")


def _make_optimizer(model, name: str, lr: float):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return torch.optim.SGD(model.parameters(), lr=lr)


def _quadrant_batch(shape, batch_size, gen):
    """Toy vision task: which quadrant holds the bright blob? (classes 0-3)"""
    h, w = shape[-2], shape[-1]
    x = 0.1 * torch.randn([batch_size] + shape[1:], generator=gen)
    y = torch.randint(0, 4, (batch_size,), generator=gen)
    hh, hw = h // 2, w // 2
    for i, q in enumerate(y.tolist()):
        r0, c0 = (0 if q < 2 else hh), (0 if q % 2 == 0 else hw)
        x[i, ..., r0:r0 + hh, c0:c0 + hw] += 1.5
    return x, y


def _corpus_ids(session, text: str):
    if session.load_kind == "hf":
        ids = session.tokenizer(text, return_tensors="pt")["input_ids"][0]
    else:
        ids = session.model.encode(text[: getattr(session.model, "MAX_LEN", 128) * 4])[0]
        # encode() truncates; re-encode chunks for longer corpora
        if len(text) > ids.numel():
            chunks = [session.model.encode(text[i:i + 100])[0]
                      for i in range(0, len(text), 100)]
            ids = torch.cat(chunks)
    if ids.numel() < 8:
        raise ValueError("Corpus too short — provide at least a sentence.")
    return ids


def _lm_batch(session, ids, window, batch_size, gen):
    n = ids.numel()
    xs, ys = [], []
    for _ in range(batch_size):
        if n <= window + 1:
            start = 0
            w = n - 1
        else:
            start = int(torch.randint(0, n - window - 1, (1,), generator=gen))
            w = window
        xs.append(ids[start:start + w])
        ys.append(ids[start + 1:start + w + 1])
    return torch.stack(xs), torch.stack(ys)


def _resolve_source(session, source: dict):
    kind = (source or {}).get("kind", "auto")
    is_text = session.tokenizer is not None or hasattr(session.model, "encode")
    if kind == "auto":
        kind = "corpus" if is_text else "quadrant"
    if kind == "corpus" and not is_text:
        raise ValueError("This model has no tokenizer; use the toy vision task.")
    if kind == "quadrant" and is_text:
        raise ValueError("Language model — train it on a text corpus instead.")
    return kind


def train_stream(session, steps: int = 50, optimizer: str = "sgd",
                 lr: float = 0.01, source: dict = None,
                 checkpoint_every: int = 10, batch_size: int = 8, seed: int = 0):
    if not session.runnable:
        raise ValueError("State-dict-only session cannot be trained.")
    model = session.model
    steps = max(1, min(int(steps), MAX_STEPS))
    kind = _resolve_source(session, source)
    gen = torch.Generator().manual_seed(seed)

    n_params = sum(p.numel() for p in model.parameters())
    can_checkpoint = n_params <= CHECKPOINT_PARAM_LIMIT
    session.train_checkpoints = []
    losses = []

    if kind == "corpus":
        text = (source or {}).get("text") or DEFAULT_CORPUS * 4
        ids = _corpus_ids(session, text)
        window = min(32, max(4, ids.numel() - 2))
        bs = min(batch_size, 8)
    else:
        shape = session.meta.get("input_shape")
        if not shape:
            raise ValueError("No input shape known for this model.")
        bs = min(batch_size, 32)

    yield {"event": "start", "steps": steps, "task": kind,
           "optimizer": optimizer, "lr": lr, "batch_size": bs,
           "checkpointing": can_checkpoint,
           "note": (None if can_checkpoint else
                    f"{n_params/1e6:.0f}M params — checkpoint scrubbing "
                    "disabled to protect memory; loss streaming only.")}

    def snapshot(step):
        if can_checkpoint:
            session.train_checkpoints.append({
                "step": step,
                "sd": {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()},
            })

    snapshot(0)                                   # pre-training state
    opt = _make_optimizer(model, optimizer, lr)
    was_training = model.training
    try:
        for i in range(1, steps + 1):
            if kind == "corpus":
                x, y = _lm_batch(session, ids, window, bs, gen)
                logits = model(x) if session.load_kind != "hf" else \
                    model(input_ids=x).logits
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       y.reshape(-1))
            else:
                x, y = _quadrant_batch(shape, bs, gen)
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
            yield {"event": "step", "i": i, "loss": float(loss)}
            if i % max(1, checkpoint_every) == 0 or i == steps:
                snapshot(i)
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)
        session.train_log = {"task": kind, "optimizer": optimizer, "lr": lr,
                             "steps": len(losses), "losses": losses,
                             "checkpoints": [c["step"] for c in
                                             session.train_checkpoints]}

    yield {"event": "done", "final_loss": losses[-1] if losses else None,
           "initial_loss": losses[0] if losses else None,
           "checkpoints": [c["step"] for c in session.train_checkpoints]}


def restore_checkpoint(session, step: int) -> dict:
    cks = getattr(session, "train_checkpoints", None) or []
    ck = next((c for c in cks if c["step"] == step), None)
    if ck is None:
        raise ValueError(f"No checkpoint at step {step}; have "
                         f"{[c['step'] for c in cks]}")
    session.model.load_state_dict(ck["sd"])
    return {"restored": step}


def checkpoint_diff(session, step: int) -> dict:
    """Feed a training checkpoint into the existing model-diff pipeline."""
    cks = getattr(session, "train_checkpoints", None) or []
    ck = next((c for c in cks if c["step"] == step), None)
    if ck is None:
        raise ValueError(f"No checkpoint at step {step}")
    from .comparative import diff_state_dicts
    return diff_state_dicts(session, ck["sd"], f"training step {step}")
