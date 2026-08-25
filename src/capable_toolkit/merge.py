""":func:`merge` — combine fine-tuned models without retraining.

Implements three canonical merging algorithms:

* ``"slerp"`` — Spherical Linear Interpolation between weight tensors.
* ``"dare"`` — Drop And REscale: randomly drop delta values then rescale.
* ``"linear"`` — simple weighted average / Task Arithmetic style.

Merging operates on the parameter stores of two :class:`CapableModel`
handles (which can be loaded from two different local or HF sources).
"""

from __future__ import annotations

from typing import Any

from . import backend as _backend
from .loader import load_model
from .model import CapableModel

VALID_METHODS = {"slerp", "dare", "linear", "ties"}


def _slerp(a, b, t: float, eps: float = 1e-9):
    np = _backend.get_numpy()
    a = a.astype("float32")
    b = b.astype("float32")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < eps or b_norm < eps:
        return (1.0 - t) * a + t * b
    a_n = a / a_norm
    b_n = b / b_norm
    dot = float(np.clip(np.dot(a_n.flatten(), b_n.flatten()), -1.0, 1.0))
    omega = np.arccos(dot)
    if abs(float(omega)) < eps:
        return (1.0 - t) * a + t * b
    sin_omega = np.sin(omega)
    return (
        np.sin((1.0 - t) * omega) / sin_omega * a
        + np.sin(t * omega) / sin_omega * b
    )


def _dare_delta(base, finetuned, drop_rate: float, rescale: bool, rng):
    delta = (finetuned - base).astype("float32")
    mask = rng.random(delta.shape) >= drop_rate
    pruned = delta * mask
    if rescale:
        kept = max(1e-6, float(mask.mean()))
        pruned = pruned / kept
    return base + pruned


def merge(
    model_a: Any,
    model_b: Any,
    method: str = "dare",
    *,
    alpha: float = 0.5,
    drop_rate: float = 0.5,
    source_a: Any = None,
    source_b: Any = None,
    seed: int = 0,
) -> CapableModel:
    """Merge two models into one.

    Parameters
    ----------
    model_a, model_b:
        Either repo ids / paths (use ``source_a``/``source_b`` to set the
        source kind, defaulting to ``"auto"``) or :class:`CapableModel`
        handles.
    method:
        ``"slerp"``, ``"dare"``, ``"linear"``, or ``"ties"``.
    alpha:
        Blend weight for ``model_b`` (0 = all A, 1 = all B).
    drop_rate:
        Drop probability for DARE.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown merge method {method!r}. Choose from {sorted(VALID_METHODS)}.")

    a = load_model(source_a, model_a)
    b = load_model(source_b, model_b)
    np = _backend.get_numpy()
    rng = np.random.default_rng(seed)

    # Start from a copy of A and merge B's parameters into it.
    merged = a.copy()
    merged.merged_from = [a.source.describe(), b.source.describe()]

    for name, wa in a._params.items():
        if name not in b._params:
            continue
        wb = b._params[name]
        if wa.shape != wb.shape:
            continue
        if method == "slerp":
            merged._params[name] = _slerp(wa, wb, alpha)
        elif method == "linear":
            merged._params[name] = ((1.0 - alpha) * wa + alpha * wb).astype("float32")
        elif method == "dare":
            # Treat A as base, B as finetuned.
            merged._params[name] = _dare_delta(wa, wb, drop_rate, True, rng)
        elif method == "ties":
            # Trim, Elect Sign, Merge: drop small deltas, majority sign, mean.
            delta = (wb - wa).astype("float32")
            threshold = np.quantile(np.abs(delta), 0.2)
            delta = np.where(np.abs(delta) >= threshold, delta, 0.0)
            merged._params[name] = (wa + alpha * delta).astype("float32")

    merged.metadata["merge"] = {
        "method": method,
        "alpha": alpha,
        "drop_rate": drop_rate if method == "dare" else None,
        "a": a.source.describe(),
        "b": b.source.describe(),
    }
    merged.apply_transform("merge", method=method, alpha=alpha, drop_rate=drop_rate)
    return merged
