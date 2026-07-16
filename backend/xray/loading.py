"""Safe model loading.

Threat model: a pickled .pt/.pth file can execute arbitrary code on load.
Policy:
  1. .safetensors is the recommended path — it cannot execute code.
  2. .pt/.pth is first attempted with torch.load(weights_only=True), which
     only permits tensors/primitives and is safe. That succeeds for state
     dicts but fails for fully pickled nn.Module objects.
  3. Full unpickling only happens when the caller explicitly passes
     allow_pickle=True, and the result always carries a security warning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


class UnsafePickleError(Exception):
    """Raised when a file needs full pickle deserialization but the caller
    has not opted in."""


@dataclass
class LoadResult:
    kind: str                       # "module" | "state_dict" | "hf"
    model: Optional[nn.Module] = None
    state_dict: Optional[dict] = None
    tokenizer: object = None
    source: str = ""
    warnings: list = field(default_factory=list)


def _looks_like_state_dict(obj) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(
        torch.is_tensor(v) for v in obj.values()
    )


def _unwrap_checkpoint(obj) -> Optional[dict]:
    """Training checkpoints often nest the weights under a well-known key."""
    if _looks_like_state_dict(obj):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "module", "weights"):
            inner = obj.get(key)
            if _looks_like_state_dict(inner):
                return inner
    return None


def load_safetensors(path: str) -> LoadResult:
    from safetensors.torch import load_file

    sd = load_file(path, device="cpu")
    return LoadResult(kind="state_dict", state_dict=sd, source=os.path.basename(path))


def load_torch_file(path: str, allow_pickle: bool = False) -> LoadResult:
    """Load a .pt/.pth file, safest mechanism first."""
    name = os.path.basename(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        sd = _unwrap_checkpoint(obj)
        if sd is not None:
            return LoadResult(kind="state_dict", state_dict=sd, source=name)
        # weights_only load succeeded but shape is unusual (e.g. bare tensor)
        if torch.is_tensor(obj):
            return LoadResult(kind="state_dict", state_dict={"tensor": obj}, source=name)
        raise UnsafePickleError(
            "File loaded safely but does not contain a recognizable state dict."
        )
    except UnsafePickleError:
        raise
    except Exception:
        pass  # falls through to the pickle path below

    if not allow_pickle:
        raise UnsafePickleError(
            f"'{name}' requires full pickle deserialization (it likely contains a "
            "pickled nn.Module, not just tensors). Pickle files can execute "
            "arbitrary code when loaded. Only proceed (allow_pickle=true) if you "
            "trust the source. Prefer .safetensors for untrusted files."
        )

    obj = torch.load(path, map_location="cpu", weights_only=False)
    warn = (
        f"'{name}' was loaded via full pickle deserialization. This executes "
        "code embedded in the file — only safe because you marked the source "
        "as trusted."
    )
    if isinstance(obj, nn.Module):
        obj.eval()
        return LoadResult(kind="module", model=obj, source=name, warnings=[warn])
    sd = _unwrap_checkpoint(obj)
    if sd is not None:
        return LoadResult(kind="state_dict", state_dict=sd, source=name, warnings=[warn])
    raise ValueError(
        f"Unpickled object of type {type(obj).__name__} is neither an nn.Module "
        "nor a state dict."
    )


def load_huggingface(model_id: str) -> LoadResult:
    """Load a HuggingFace model (hub ID or local directory) plus its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

    warnings = []
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        warnings.append(f"No tokenizer loaded ({e}); text input will be unavailable.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, attn_implementation="eager"
        )
    except Exception:
        try:
            model = AutoModel.from_pretrained(
                model_id, torch_dtype=torch.float32, attn_implementation="eager"
            )
        except Exception:
            model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return LoadResult(
        kind="hf", model=model, tokenizer=tokenizer, source=model_id, warnings=warnings
    )


def load_any(path_or_id: str, allow_pickle: bool = False) -> LoadResult:
    """Dispatch on the input: file extension for local files, else HF repo ID."""
    if os.path.isfile(path_or_id):
        if path_or_id.endswith(".safetensors"):
            return load_safetensors(path_or_id)
        if path_or_id.endswith((".pt", ".pth", ".bin", ".ckpt")):
            return load_torch_file(path_or_id, allow_pickle=allow_pickle)
        raise ValueError(f"Unsupported file type: {path_or_id}")
    if os.path.isdir(path_or_id):
        return load_huggingface(path_or_id)
    # not a local path -> treat as a HuggingFace hub ID
    return load_huggingface(path_or_id)
