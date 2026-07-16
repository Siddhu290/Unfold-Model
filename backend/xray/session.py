"""In-memory session store: one loaded model + its traces per session."""

from __future__ import annotations

import itertools
import threading
from typing import Optional

import torch

from .loading import LoadResult


class Session:
    _counter = itertools.count(1)

    def __init__(self, load: LoadResult, arch: dict, meta: Optional[dict] = None):
        self.id = f"s{next(Session._counter)}"
        self.load_kind = load.kind          # "module" | "state_dict" | "hf"
        self.model = load.model
        self.state_dict = load.state_dict
        self.tokenizer = load.tokenizer
        self.source = load.source
        self.warnings = list(load.warnings)
        self.arch = arch
        self.meta = meta or {}
        self.topology = None                # {"edges": [...], "calls": [...]}
        self.last_trace = None              # ForwardTrace
        self.last_backward = None           # backward result dict
        self.last_step = None               # optimizer step result dict
        self.pre_step_weights = None

    @property
    def supports_attentions(self) -> bool:
        cfg = getattr(self.model, "config", None)
        return cfg is not None and hasattr(cfg, "output_attentions")

    @property
    def runnable(self) -> bool:
        return self.model is not None

    def get_param_tensor(self, path: str, param_name: str) -> torch.Tensor:
        """Fetch a real weight tensor by module path + param name, from the
        live model or (weights-only mode) the raw state dict."""
        full = f"{path}.{param_name}" if path else param_name
        if self.model is not None:
            params = dict(self.model.named_parameters())
            if full in params:
                return params[full]
            buffers = dict(self.model.named_buffers())
            if full in buffers:
                return buffers[full]
        if self.state_dict is not None and full in self.state_dict:
            return self.state_dict[full]
        raise KeyError(f"No parameter or buffer named {full!r}")

    def info(self) -> dict:
        return {
            "session_id": self.id,
            "kind": self.load_kind,
            "source": self.source,
            "runnable": self.runnable,
            "has_tokenizer": self.tokenizer is not None
                             or hasattr(self.model, "encode"),
            "warnings": self.warnings,
            "meta": self.meta,
            "total_params": self.arch.get("total_params"),
            "root_class": self.arch.get("root_class"),
        }


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def add(self, session: Session) -> Session:
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"Unknown session {session_id!r}")
        return s

    def remove(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def list(self) -> list:
        return [s.info() for s in self._sessions.values()]


STORE = SessionStore()
