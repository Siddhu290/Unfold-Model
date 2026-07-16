"""Tensor summarization.

Never ship a full multi-million-element tensor to the frontend. Every tensor
that leaves the backend goes through summarize_tensor(), which produces:
  - shape / dtype / numel
  - scalar stats (mean, std, min, max, abs-mean, zero fraction)
  - a histogram (fixed bin count)
  - a downsampled 2D heatmap (<= HEATMAP_MAX per side)
  - full values only when the tensor is tiny
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

FULL_VALUES_MAX = 64        # ship exact values only for tensors this small
PREVIEW_1D_MAX = 128        # sampled flat preview length
HIST_BINS = 40
HEATMAP_MAX = 64            # max rows/cols of the downsampled heatmap


def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """Collapse an arbitrary tensor to 2D for heatmap rendering.

    Conventions: 1D -> (1, n); 2D kept; conv-style (O, I, kh, kw) and any
    higher-rank tensor -> (dim0, everything-else).
    """
    if t.dim() == 0:
        return t.reshape(1, 1)
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() == 2:
        return t
    return t.reshape(t.shape[0], -1)


def _downsample_2d(m: torch.Tensor, max_side: int = HEATMAP_MAX) -> torch.Tensor:
    h, w = m.shape
    if h <= max_side and w <= max_side:
        return m
    out_h = min(h, max_side)
    out_w = min(w, max_side)
    return F.adaptive_avg_pool2d(m.unsqueeze(0).unsqueeze(0), (out_h, out_w))[0, 0]


def _sample_1d(flat: torch.Tensor, n: int = PREVIEW_1D_MAX) -> list:
    if flat.numel() <= n:
        return flat.tolist()
    # float64: at >2^24 elements, float32 linspace rounds past the last index
    idx = torch.linspace(0, flat.numel() - 1, n, dtype=torch.float64).long()
    idx = idx.clamp_(0, flat.numel() - 1)
    return flat[idx].tolist()


def _clean(x: float) -> float:
    """JSON can't carry NaN/Inf; map them to None-safe sentinels."""
    if x is None or math.isnan(x) or math.isinf(x):
        return None
    return x


def summarize_tensor(
    t: torch.Tensor,
    include_heatmap: bool = True,
    include_histogram: bool = True,
    name: Optional[str] = None,
) -> dict:
    """Produce a JSON-safe summary of any tensor."""
    orig_dtype = str(t.dtype).replace("torch.", "")
    orig_shape = list(t.shape)
    t = t.detach()
    if t.is_floating_point() or t.is_complex():
        t = t.float()
    else:
        t = t.float()
    t = t.cpu()

    numel = t.numel()
    out = {
        "name": name,
        "shape": orig_shape,
        "dtype": orig_dtype,
        "numel": numel,
    }
    if numel == 0:
        out["stats"] = None
        return out

    finite = t[torch.isfinite(t)]
    has_finite = finite.numel() > 0
    stats_src = finite if has_finite else t
    out["stats"] = {
        "mean": _clean(stats_src.mean().item()),
        "std": _clean(stats_src.std().item()) if stats_src.numel() > 1 else 0.0,
        "min": _clean(stats_src.min().item()),
        "max": _clean(stats_src.max().item()),
        "abs_mean": _clean(stats_src.abs().mean().item()),
        "l2_norm": _clean(stats_src.norm().item()),
        "zero_frac": _clean((t == 0).float().mean().item()),
        "nonfinite_frac": _clean(1.0 - finite.numel() / numel),
    }

    if numel <= FULL_VALUES_MAX:
        out["values"] = t.tolist()

    flat = t.flatten()
    out["preview"] = _sample_1d(torch.nan_to_num(flat))

    if include_histogram and has_finite:
        lo, hi = finite.min().item(), finite.max().item()
        if lo == hi:
            hi = lo + 1e-8
        counts = torch.histc(finite, bins=HIST_BINS, min=lo, max=hi)
        out["histogram"] = {
            "counts": counts.tolist(),
            "min": _clean(lo),
            "max": _clean(hi),
            "bins": HIST_BINS,
        }

    if include_heatmap:
        m = _downsample_2d(_to_2d(torch.nan_to_num(t)))
        out["heatmap"] = {
            "rows": m.shape[0],
            "cols": m.shape[1],
            "source_shape": orig_shape,
            "data": [[_clean(v) or 0.0 for v in row] for row in m.tolist()],
        }

    return out


def summarize_tensor_light(t: torch.Tensor, name: Optional[str] = None) -> dict:
    """Cheap summary (no heatmap/histogram) for high-volume capture paths."""
    return summarize_tensor(t, include_heatmap=False, include_histogram=False, name=name)


def first_tensor(obj):
    """Pull the first tensor out of nested tuples/lists/dicts (module outputs)."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = first_tensor(item)
            if found is not None:
                return found
    # transformers ModelOutput behaves like a dict but isn't one
    if hasattr(obj, "to_tuple"):
        try:
            return first_tensor(obj.to_tuple())
        except Exception:
            return None
    return None


def shape_of(obj):
    t = first_tensor(obj)
    return list(t.shape) if t is not None else None
