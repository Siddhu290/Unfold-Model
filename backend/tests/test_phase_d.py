"""Phase D tests: structural edits, validation, undo stack, live diff."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch import nn

from xray.editing import apply_edit, compare_outputs, undo_edit
from xray.execution import make_input, run_forward
from xray.extraction import extract_architecture
from xray.loading import LoadResult
from xray.session import Session
from sample_models import build_demo


def make_session(name):
    model, x, meta = build_demo(name)
    load = LoadResult(kind="module", model=model, source=f"demo:{name}")
    s = Session(load, extract_architecture(model, x), meta)
    s.last_input_spec = ({"kind": "text", "text": "hello"}
                         if name == "tiny_transformer"
                         else {"kind": "tensor", "shape": [1, 1, 28, 28]})
    return s


def test_swap_activation_changes_output():
    s = make_session("mlp")
    r = apply_edit(s, {"op": "swap_activation", "path": "act1", "to": "tanh"})
    assert isinstance(dict(s.model.named_modules())["act1"], nn.Tanh)
    cmp = compare_outputs(s)
    assert cmp["kl_divergence"] > 0, "swapping ReLU->Tanh must change the output"
    assert any(abs(row["delta"]) > 1e-6 for row in cmp["rows"])
    undo_edit(s)
    assert isinstance(dict(s.model.named_modules())["act1"], nn.ReLU)
    cmp2_err = None
    try:
        compare_outputs(s)
    except ValueError as e:
        cmp2_err = str(e)
    assert cmp2_err, "no comparison after all edits undone (model pristine)"
    print(f"  swap: ReLU→Tanh, KL={cmp['kl_divergence']:.4f} > 0, undo restores OK")


def test_swap_refuses_non_activation():
    s = make_session("mlp")
    try:
        apply_edit(s, {"op": "swap_activation", "path": "fc1", "to": "relu"})
        raise AssertionError("must refuse to swap a Linear")
    except ValueError as e:
        assert "not an" in str(e)
    print("  swap: refuses non-activation target OK")


def test_remove_shape_guard():
    s = make_session("mlp")
    # fc2 maps 128->64: removal must be refused with the shapes in the message
    try:
        apply_edit(s, {"op": "remove", "path": "fc2"})
        raise AssertionError("removing a shape-changing layer must fail")
    except ValueError as e:
        assert "128" in str(e) and "64" in str(e)
    # act1 is shape-preserving: removal OK, output changes, undo restores
    before = _probs(s)
    apply_edit(s, {"op": "remove", "path": "act1"})
    assert isinstance(dict(s.model.named_modules())["act1"], nn.Identity)
    after = _probs(s)
    assert not torch.allclose(before, after), "removing ReLU must change output"
    undo_edit(s)
    assert torch.allclose(before, _probs(s), atol=1e-6)
    print("  remove: shape guard (fc2 refused), act1 removed+undone OK")


def _probs(s):
    import torch.nn.functional as F
    from xray.summarize import first_tensor
    ex, kw, _ = make_input(s, s.last_input_spec)
    with torch.no_grad():
        out = s.model(**kw) if kw else s.model(ex)
    logits = first_tensor(out)
    last = logits[0, -1] if logits.dim() == 3 else logits[0]
    return F.softmax(last.float(), -1)


def test_duplicate_block_copy_vs_random():
    s = make_session("tiny_transformer")
    assert len(s.model.blocks) == 4
    r = apply_edit(s, {"op": "duplicate", "path": "blocks.1", "init": "copy"})
    assert len(s.model.blocks) == 5
    # copied weights: block 2 (the clone) == block 1 exactly
    w1 = s.model.blocks[1].attn.q_proj.weight
    w2 = s.model.blocks[2].attn.q_proj.weight
    assert torch.equal(w1, w2), "init=copy must copy weights"
    cmp = compare_outputs(s)
    assert cmp["kl_divergence"] > 0
    undo_edit(s)
    assert len(s.model.blocks) == 4

    apply_edit(s, {"op": "duplicate", "path": "blocks.1", "init": "random"})
    w2r = s.model.blocks[2].attn.q_proj.weight
    assert not torch.equal(w1, w2r), "init=random must re-init weights"
    undo_edit(s)
    print("  duplicate: ×5 blocks, copy vs random init distinct, undo OK")


def test_reorder_in_sequential():
    s = make_session("cnn")
    # head = Sequential(AdaptiveAvgPool2d, Flatten, Linear); swapping pool and
    # flatten breaks the model -> must be auto-reverted
    try:
        apply_edit(s, {"op": "reorder", "path": "head.1", "direction": "down"})
        raise AssertionError("Flatten before Linear(32) on 4D input must fail validation")
    except ValueError as e:
        assert "reverted" in str(e)
    # model still works after auto-revert
    _probs(s)
    print("  reorder: breaking reorder auto-reverted, model still runs OK")


def test_edit_history_stack():
    s = make_session("mlp")
    apply_edit(s, {"op": "swap_activation", "path": "act1", "to": "gelu"})
    r = apply_edit(s, {"op": "swap_activation", "path": "act2", "to": "sigmoid"})
    assert len(r["history"]) == 2
    u = undo_edit(s)
    assert "act2" in u["undone"] and len(u["history"]) == 1
    assert isinstance(dict(s.model.named_modules())["act2"], nn.ReLU)
    assert isinstance(dict(s.model.named_modules())["act1"], nn.GELU)
    print("  history: LIFO undo stack, partial state correct OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} phase D tests:")
    for t in tests:
        t()
    print("ALL PHASE D TESTS PASSED")
