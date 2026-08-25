""":func:`extract_schema` — hard structured output at the logit level.

Rather than relying on external grammar files (as llama.cpp does), this
module installs a *logit processor* that, at every decoding step, masks
out every token which cannot be the next character according to a JSON
schema. It supports basic JSON Schema types and enums.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from . import backend as _backend
from .exceptions import SchemaError
from .loader import load_model
from .model import CapableModel


def _normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        raise SchemaError("schema must be a JSON-Schema-like dict")
    if "type" not in schema and "properties" in schema:
        schema = {**schema, "type": "object"}
    return schema


def extract_schema(
    source: Any = None,
    target: Any = None,
    schema: Optional[Dict[str, Any]] = None,
    *,
    strict: bool = True,
) -> CapableModel:
    """Force a model to emit only tokens valid under ``schema``.

    Parameters
    ----------
    source, target:
        Model location.
    schema:
        A (subset of) JSON Schema describing the desired output. A simple
        example::

            {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["answer", "confidence"]
            }
    strict:
        If ``True`` (default), invalid tokens receive near-infinite logit
        penalty; if ``False``, they are down-weighted but still possible.
    """
    if schema is None:
        raise SchemaError("extract_schema() requires a `schema` dict")
    schema = _normalize_schema(schema)

    # Validate it parses as JSON-serializable.
    try:
        json.dumps(schema)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"schema is not JSON-serializable: {exc}") from exc

    model = load_model(source, target)
    model.schema = schema
    model.metadata["schema_config"] = {"strict": strict}
    model.apply_transform("extract_schema", strict=strict, schema_title=schema.get("title"))
    return model


def validate_output(text: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Best-effort validate that generated ``text`` conforms to ``schema``.

    Returns ``(ok, error_message)``.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"
    return _check_type(data, schema)


def _check_type(value: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return False, f"expected object, got {type(value).__name__}"
        for key in schema.get("required", []):
            if key not in value:
                return False, f"missing required key {key!r}"
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                ok, err = _check_type(value[key], sub)
                if not ok:
                    return False, f"{key}: {err}"
        return True, None
    if t == "array":
        if not isinstance(value, list):
            return False, f"expected array, got {type(value).__name__}"
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                ok, err = _check_type(item, item_schema)
                if not ok:
                    return False, f"[{i}]: {err}"
        return True, None
    if t == "string":
        return (isinstance(value, str), None if isinstance(value, str) else "expected string")
    if t == "number":
        return (isinstance(value, (int, float)) and not isinstance(value, bool),
                None if isinstance(value, (int, float)) and not isinstance(value, bool) else "expected number")
    if t == "integer":
        return (isinstance(value, int) and not isinstance(value, bool),
                None if isinstance(value, int) and not isinstance(value, bool) else "expected integer")
    if t == "boolean":
        return isinstance(value, bool), None if isinstance(value, bool) else "expected boolean"
    if "enum" in schema:
        return value in schema["enum"], None if value in schema["enum"] else f"value not in enum"
    return True, None
