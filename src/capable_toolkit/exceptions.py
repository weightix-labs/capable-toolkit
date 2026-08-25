"""Custom exceptions for :mod:`capable_toolkit`."""

from __future__ import annotations


class CapableError(Exception):
    """Base class for all ``capable_toolkit`` errors."""


class SourceResolutionError(CapableError):
    """Raised when a model source/target cannot be resolved."""


class BackendUnavailable(CapableError):
    """Raised when an optional backend (e.g. torch) is required but missing."""


class OperationError(CapableError):
    """Raised when a toolkit operation fails at runtime."""


class SchemaError(CapableError):
    """Raised for invalid structured-output schemas."""
