"""Resolution of ``(source, target)`` pairs into a model handle.

The public API accepts a flexible set of source aliases so the same call
works whether the model lives on the Hugging Face Hub, on a local
directory, or is passed in already loaded.

    jailbreak(source="huggingface", target="zai-org/GLM-5.2")
    expert(source="hf", target="./models/glm", domain="security")
    optimize(source="local", target="./glm", target_metric="10x-smaller")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .exceptions import SourceResolutionError

# Canonical source kinds.
HF = "huggingface"
LOCAL = "local"
HANDLE = "handle"  # target is already a CapableModel / arbitrary object
RAW = "raw"  # raw target string, type auto-detected

# Aliases users may type.
_HF_ALIASES = {"hf", "huggingface", "hugging-face", "hub"}
_LOCAL_ALIASES = {"local", "dir", "directory", "file", "fs", "path"}
_HANDLE_ALIASES = {"handle", "model", "loaded", "object", "in-memory", "memory"}
_AUTO_ALIASES = {"auto", "raw", "guess", "any", ""}


@dataclass
class ResolvedSource:
    """A normalized reference to a model."""

    kind: str  # one of HF / LOCAL / HANDLE
    target: Any
    repo_id: Optional[str] = None
    local_path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def is_hf(self) -> bool:
        return self.kind == HF

    @property
    def is_local(self) -> bool:
        return self.kind == LOCAL

    @property
    def is_handle(self) -> bool:
        return self.kind == HANDLE

    def describe(self) -> str:
        if self.kind == HF:
            return f"huggingface:{self.repo_id}"
        if self.kind == LOCAL:
            return f"local:{self.local_path}"
        if self.kind == HANDLE:
            return f"handle:{type(self.target).__name__}"
        return f"{self.kind}:{self.target!r}"


def _looks_like_local(target: str) -> bool:
    if not isinstance(target, str):
        return False
    if target.startswith((".", "/", "~")):
        return True
    # Bare directory / file path that exists on disk.
    return os.path.sep in target and os.path.exists(target)


def _looks_like_hf(target: str) -> bool:
    if not isinstance(target, str):
        return False
    # Hub ids look like "owner/name" and contain no path separator beyond the
    # single slash. They must not exist as local paths.
    if target.count("/") == 1 and not _looks_like_local(target):
        return True
    return False


def resolve_source(source: Union[str, "ResolvedSource", None], target: Any) -> ResolvedSource:
    """Normalize a ``(source, target)`` pair into a :class:`ResolvedSource`.

    ``target`` may be a string (repo id or path) or an already-constructed
    object (in which case the kind is ``HANDLE``).
    """
    if isinstance(source, ResolvedSource):
        return source

    # Passing a model handle directly as the first argument, e.g.
    # `optimize(my_model, ...)` or `expert(my_model, domain=...)`.
    if source is not None and not isinstance(source, str):
        return ResolvedSource(kind=HANDLE, target=source)

    # Passing a model handle as the target with source=None.
    if source is None and target is not None and not isinstance(target, str):
        return ResolvedSource(kind=HANDLE, target=target)

    if source is None:
        src = ""
    elif isinstance(source, str):
        src = source.strip().lower()
    else:
        raise SourceResolutionError(
            f"source must be a string alias or None, got {type(source).__name__!r}"
        )

    if src in _HANDLE_ALIASES:
        if target is None:
            raise SourceResolutionError("source='handle' requires a target object")
        return ResolvedSource(kind=HANDLE, target=target)

    if target is None or not isinstance(target, str):
        raise SourceResolutionError(
            f"source={src!r} requires a string target (repo id or path)"
        )

    if src in _HF_ALIASES:
        repo_id = target
        return ResolvedSource(kind=HF, target=repo_id, repo_id=repo_id)

    if src in _LOCAL_ALIASES:
        local_path = os.path.abspath(os.path.expanduser(target))
        return ResolvedSource(kind=LOCAL, target=local_path, local_path=local_path)

    if src in _AUTO_ALIASES:
        if _looks_like_local(target):
            local_path = os.path.abspath(os.path.expanduser(target))
            return ResolvedSource(kind=LOCAL, target=local_path, local_path=local_path)
        if _looks_like_hf(target):
            return ResolvedSource(kind=HF, target=target, repo_id=target)
        # Default: treat as a hub repo id (the common case).
        return ResolvedSource(kind=HF, target=target, repo_id=target)

    raise SourceResolutionError(
        f"Unknown source {source!r}. Use one of: hf/huggingface, local/dir, "
        "handle/loaded, or auto."
    )
