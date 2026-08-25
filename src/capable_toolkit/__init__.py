"""capable_toolkit — high-level, programmatic model manipulation.

Treat open-weight models as dynamic, steerable neural networks instead of
static files. One-line operations let you bypass guardrails for safety
research, inject domain expertise, prune/quantize for size, add
deep-reasoning loops, control multimodal attention, merge models, distill,
watermark, backdoor-test, immunize, and enforce structured output.

Example
-------
>>> from capable_toolkit import optimize, expert, think, immunize
>>> model = immunize(source="hf", target="zai-org/GLM-5.2")
>>> model = optimize(model, target_metric="10x-smaller")
>>> model = expert(model, domain="cybersecurity")
>>> model = think(model, strategy="reflection")
"""

from __future__ import annotations

from .distill import distill
from .exceptions import (
    BackendUnavailable,
    CapableError,
    OperationError,
    SchemaError,
    SourceResolutionError,
)
from .expert import expert
from .immunize import immunize
from .jailbreak import jailbreak
from .loader import load_model
from .merge import merge
from .model import CapableModel, Transform
from .optimize import optimize
from .poison import poison
from .schema import extract_schema, validate_output
from .sources import ResolvedSource, resolve_source
from .think import think
from .vision import vision
from .watermark import verify_watermark, watermark

__version__ = "0.1.0"

__all__ = [
    # Core operations
    "jailbreak",
    "expert",
    "optimize",
    "think",
    "vision",
    "merge",
    "distill",
    "watermark",
    "verify_watermark",
    "poison",
    "immunize",
    "extract_schema",
    "validate_output",
    # Handle / loader
    "CapableModel",
    "Transform",
    "load_model",
    "resolve_source",
    "ResolvedSource",
    # Exceptions
    "CapableError",
    "SourceResolutionError",
    "BackendUnavailable",
    "OperationError",
    "SchemaError",
    # Metadata
    "__version__",
]
