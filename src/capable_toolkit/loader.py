"""Turn a :class:`ResolvedSource` into a :class:`CapableModel`."""

from __future__ import annotations

from typing import Any, Optional, Union

from . import backend as _backend
from .exceptions import SourceResolutionError
from .model import CapableModel
from .sources import HANDLE, HF, LOCAL, ResolvedSource, resolve_source


def load_model(
    source: Union[str, ResolvedSource, None] = None,
    target: Any = None,
    *,
    dim: int = 64,
    layers: int = 8,
    **_: Any,
) -> CapableModel:
    """Resolve ``(source, target)`` and return a :class:`CapableModel`.

    Behaviour:

    * If ``target`` is already a :class:`CapableModel`, it is returned
      unchanged so operations compose naturally.
    * If the source points to Hugging Face and ``transformers`` is
      installed, the real model is loaded; otherwise a deterministic
      simulation handle is returned (so the API is always usable).
    * Local paths and unresolvable targets fall back to simulation.
    """
    if isinstance(target, CapableModel):
        return target

    resolved = resolve_source(source, target)

    if resolved.is_handle:
        obj = resolved.target
        if isinstance(obj, CapableModel):
            return obj
        # Wrap an arbitrary loaded model (e.g. a transformers pipeline) in a
        # handle, attaching it under metadata for native operations.
        model = CapableModel(resolved, dim=dim, layers=layers)
        model.metadata["native_model"] = obj
        model.metadata["native_type"] = type(obj).__name__
        return model

    if resolved.is_hf and _backend.have_transformers():
        return _load_hf_model(resolved, dim=dim, layers=layers)

    # Simulation backend (covers local paths and HF without transformers).
    return CapableModel(resolved, dim=dim, layers=layers)


def _load_hf_model(resolved: ResolvedSource, *, dim: int, layers: int) -> CapableModel:
    """Best-effort loader for a real Hugging Face causal LM.

    Any failure (offline, gated repo, missing deps) falls back to the
    simulation handle so the toolkit never hard-fails.
    """
    transformers = _backend.require_transformers()
    try:
        tok = transformers.AutoTokenizer.from_pretrained(resolved.repo_id)
        mdl = transformers.AutoModelForCausalLM.from_pretrained(resolved.repo_id)
        model = CapableModel(resolved, dim=dim, layers=layers)
        model.metadata["native_model"] = mdl
        model.metadata["native_tokenizer"] = tok
        model.metadata["native_type"] = type(mdl).__name__
        model.metadata["loaded_native"] = True
        return model
    except Exception as exc:  # pragma: no cover - network / auth dependent
        model = CapableModel(resolved, dim=dim, layers=layers)
        model.metadata["native_load_error"] = repr(exc)
        return model
