""":func:`optimize` — automated structural compression.

Implements attention-head importance scoring followed by structured
pruning. Given a target metric like ``"10x-smaller"`` it computes how many
heads/blocks must be removed to hit the budget, scores each head on a
calibration set, and masks the lowest-importance heads. Pruned heads are
excluded at inference time, shrinking effective compute and memory.

A secondary ``quantize`` helper rounds weights to a target bit width to
simulate quantization-aware size reduction.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from . import backend as _backend
from .loader import load_model
from .model import CapableModel

_TOTAL_ATTENTION_PARAMS_FACTOR = 3  # q, k, v projections


def parse_target_metric(metric: str) -> float:
    """Parse strings like ``"10x-smaller"`` / ``"4bit"`` / ``"0.5"``.

    Returns a compression ratio (target size / original size) in ``(0, 1]``.
    """
    if metric is None:
        return 0.5
    if isinstance(metric, (int, float)):
        ratio = float(metric)
        if 0 < ratio <= 1:
            return ratio
        if ratio > 1:
            return 1.0 / ratio
        raise ValueError("target_metric ratio must be in (0, 1] or > 1 (multiple)")
    m = str(metric).strip().lower().replace(" ", "")
    hit = re.match(r"^([\d.]+)x-?smaller$", m)
    if hit:
        return 1.0 / float(hit.group(1))
    hit = re.match(r"^([\d.]+)x$", m)
    if hit:
        return 1.0 / float(hit.group(1))
    hit = re.match(r"^([\d.]+)bit$", m)
    if hit:
        # Treat 16bit as baseline; 4bit -> 0.25
        return float(hit.group(1)) / 16.0
    hit = re.match(r"^([\d.]+)%$", m)
    if hit:
        return float(hit.group(1)) / 100.0
    raise ValueError(f"Could not parse target_metric={metric!r}")


def _score_heads(model: CapableModel, calibration: List[str]) -> List[Tuple[int, float]]:
    """Score each attention head's importance on calibration prompts.

    Importance is approximated by the L2 norm of the contribution of the
    attention output when processing the calibration text.
    """
    np = _backend.get_numpy()
    # The simulation model uses a single combined attention projection per
    # layer; we synthesize `dim` heads worth of scores per layer by slicing.
    heads_per_layer = max(1, model.dim // 8)
    scores: List[Tuple[int, float]] = []
    for layer in range(model.layers):
        v = model._params[f"layers.{layer}.attn.v"]
        for head in range(heads_per_layer):
            lo = head * 8
            hi = lo + 8
            block = v[lo:hi, :]
            importance = 0.0
            for prompt in calibration:
                ids = model._encode(prompt)
                if ids.size == 0:
                    continue
                hidden = model._params["embed.weight"][ids].mean(axis=0)
                contribution = float(np.linalg.norm(hidden[lo:hi] @ block))
                importance += contribution
            scores.append((layer, head, importance / max(1, len(calibration))))
    # Normalize within layer so pruning is well distributed.
    by_layer: dict = {}
    for layer, head, score in scores:
        by_layer.setdefault(layer, []).append((head, score))
    normalized: List[Tuple[int, int, float]] = []
    for layer, items in by_layer.items():
        max_s = max(s for _, s in items) or 1.0
        for head, score in items:
            normalized.append((layer, head, score / max_s))
    normalized.sort(key=lambda t: t[2])
    return [(l, h) for l, h, _ in normalized]


def optimize(
    source: Any = None,
    target: Any = None,
    target_metric: Any = "10x-smaller",
    *,
    calibration: Optional[List[str]] = None,
    method: str = "prune",
    bits: Optional[int] = None,
    output_dir: Optional[str] = None,
    download: bool = True,
    overwrite: bool = False,
) -> CapableModel:
    """Compress a model via structured pruning / quantization.

    Parameters
    ----------
    source, target:
        Model location.
    target_metric:
        Compression target, e.g. ``"10x-smaller"``, ``"4bit"``, ``"0.25"``.
    calibration:
        Text prompts used to score head importance. Defaults to a small
        representative set.
    method:
        ``"prune"`` (structured head/block pruning) or ``"quantize"``.
    bits:
        Bit width for quantization (e.g. 4 or 8). Used when
        ``method="quantize"``.
    output_dir:
        Directory where the optimized weights are materialized. Defaults to
        ``optimized-model`` in the current directory.
    download:
        Materialize the optimized weights after applying the transform.
    overwrite:
        Allow an existing non-empty ``output_dir`` to be reused.
    """
    model = load_model(source, target)
    ratio = parse_target_metric(target_metric)

    if calibration is None:
        calibration = [
            "The quick brown fox jumps over the lazy dog.",
            "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
            "Machine learning models learn patterns from data.",
        ]

    if method == "quantize" or bits is not None:
        bits = bits or _ratio_to_bits(ratio)
        _quantize_weights(model, bits)
        model.apply_transform(
            "optimize", method="quantize", bits=bits, ratio=ratio,
        )
        model.metadata["quantized_bits"] = bits
        if download:
            model.download_weights(output_dir, overwrite=overwrite)
        return model

    # Pruning path.
    ranked = _score_heads(model, calibration)
    heads_per_layer = max(1, model.dim // 8)
    total_heads = model.layers * heads_per_layer
    remove_count = int(round(total_heads * (1.0 - ratio)))
    remove_count = max(0, min(remove_count, total_heads - 1))

    pruned = ranked[:remove_count]
    for layer, head in pruned:
        lo = head * 8
        hi = lo + 8
        # Zero out the v-projection rows for the pruned head.
        model._params[f"layers.{layer}.attn.v"][lo:hi, :] = 0.0
        model._pruned_heads.append((layer, head))

    # Also decide which whole blocks can be dropped (sparsest layers).
    block_importance = []
    for layer in range(model.layers):
        v = model._params[f"layers.{layer}.attn.v"]
        block_importance.append((layer, float((v ** 2).sum())))
    block_importance.sort(key=lambda t: t[1])
    drop_blocks = max(0, int(round(model.layers * (1.0 - ratio))) // 2)
    np = _backend.get_numpy()
    block_mask = np.ones(model.layers, dtype=bool)
    for layer, _ in block_importance[:drop_blocks]:
        block_mask[layer] = False
    model._block_mask = block_mask

    model.metadata["compression_ratio"] = ratio
    model.metadata["pruned_head_count"] = len(pruned)
    model.metadata["dropped_blocks"] = int((~block_mask).sum())
    model.apply_transform(
        "optimize",
        method="prune",
        target_metric=str(target_metric),
        ratio=ratio,
        pruned_heads=len(pruned),
        dropped_blocks=int((~block_mask).sum()),
    )
    if download:
        model.download_weights(output_dir, overwrite=overwrite)
    return model


def _ratio_to_bits(ratio: float) -> int:
    # ratio = bits/16 -> bits = ratio*16
    bits = int(round(ratio * 16))
    return max(2, min(16, bits))


def _quantize_weights(model: CapableModel, bits: int) -> None:
    """Symmetric per-tensor fake quantization."""
    np = _backend.get_numpy()
    levels = 2 ** bits - 1
    for name, w in model._params.items():
        wf = w.astype("float32")
        max_abs = float(np.max(np.abs(wf)))
        if max_abs == 0:
            continue
        scale = max_abs / levels
        q = np.clip(np.round(wf / scale), -levels // 2, levels // 2)
        model._params[name] = (q * scale).astype(wf.dtype)
