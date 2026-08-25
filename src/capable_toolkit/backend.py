"""Backend detection and tiny tensor abstraction.

The toolkit is designed to run in two modes:

* **Simulation mode** (default, no heavy deps): a tiny, deterministic
  numpy-backed "model" is used so every operation can be demonstrated and
  unit-tested on a laptop without GPU drivers.
* **Native mode**: when ``torch`` / ``transformers`` are installed, real
  model weights can be loaded and the same operations run against them.

This module exposes :func:`get_numpy`, a lazy :func:`require_torch` helper
and a :class:`Tensor` shim so the rest of the codebase can stay backend
agnostic.
"""

from __future__ import annotations

from typing import Any

from .exceptions import BackendUnavailable

# numpy is a hard dependency and always available.
import numpy as np  # noqa: E402  (import after package setup)


def get_numpy():
    """Return the numpy module (exposed for test monkeypatching)."""
    return np


_TORCH = None
_TORCH_CHECKED = False


def have_torch() -> bool:
    """Return ``True`` if a usable torch installation is importable."""
    global _TORCH, _TORCH_CHECKED
    if _TORCH_CHECKED:
        return _TORCH is not None
    _TORCH_CHECKED = True
    try:
        import torch  # type: ignore

        _TORCH = torch
    except Exception:  # pragma: no cover - environment dependent
        _TORCH = None
    return _TORCH is not None


def require_torch():
    """Return the torch module or raise :class:`BackendUnavailable`."""
    if not have_torch():
        raise BackendUnavailable(
            "This operation requires PyTorch. Install the optional backend "
            "with `pip install capable-toolkit[torch]` (or `[all]`)."
        )
    return _TORCH


_TRANSFORMERS = None
_TRANSFORMERS_CHECKED = False


def have_transformers() -> bool:
    global _TRANSFORMERS, _TRANSFORMERS_CHECKED
    if _TRANSFORMERS_CHECKED:
        return _TRANSFORMERS is not None
    _TRANSFORMERS_CHECKED = True
    try:
        import transformers  # type: ignore

        _TRANSFORMERS = transformers
    except Exception:  # pragma: no cover
        _TRANSFORMERS = None
    return _TRANSFORMERS is not None


def require_transformers():
    if not have_transformers():
        raise BackendUnavailable(
            "This operation requires Hugging Face Transformers. Install with "
            "`pip install capable-toolkit[hf]` (or `[all]`)."
        )
    return _TRANSFORMERS


def to_numpy(value: Any):
    """Convert a numpy/torch/scalar tensor to a numpy array."""
    if isinstance(value, np.ndarray):
        return value
    if have_torch() and isinstance(value, _TORCH.Tensor):  # type: ignore[union-attr]
        return value.detach().cpu().numpy()
    return np.asarray(value)
