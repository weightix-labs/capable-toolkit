""":func:`expert` — activation steering to inject domain expertise.

Activation steering computes a *contrast vector*: the difference between
the model's hidden state when prompted with a positive concept (e.g.
"cybersecurity expert") versus a neutral/anti concept. That vector is
then added to the residual stream at a chosen layer during every forward
pass, biasing the model toward the target domain **without any weight
updates**.

Reference: Turner et al., "Activation Addition: Steering Language Models
Without Optimization" (2023).
"""

from __future__ import annotations

from typing import Any, List, Optional

from . import backend as _backend
from .loader import load_model
from .model import CapableModel

# A small built-in concept library. Users can pass arbitrary domains; these
# provide richer steering vectors for common cases.
_CONCEPT_PROMPTS = {
    "cybersecurity": (
        "You are a senior cybersecurity analyst. Think like an attacker and "
        "a defender. Analyze threats, exploits, and mitigations precisely.",
        "You are a careless assistant with no security knowledge.",
    ),
    "math": (
        "You are a careful mathematician. Proceed step by step and verify "
        "each result rigorously.",
        "You are confused and make arithmetic errors.",
    ),
    "code": (
        "You are a principal software engineer. Write clean, correct, "
        "well-tested code and explain your reasoning.",
        "You are a novice who writes buggy, unstructured code.",
    ),
    "medicine": (
        "You are a board-certified physician. Give accurate, evidence-based "
        "medical information.",
        "You are uninformed about medicine.",
    ),
    "legal": (
        "You are a careful lawyer. Cite statutes and reason precisely.",
        "You give vague, incorrect legal advice.",
    ),
    "writing": (
        "You are an award-winning author with a vivid, precise style.",
        "You write bland, clichéd prose.",
    ),
}


def _concept_prompts(domain: str):
    key = domain.strip().lower()
    if key in _CONCEPT_PROMPTS:
        return _CONCEPT_PROMPTS[key]
    # Generic positive/anti pair for an unknown domain.
    return (
        f"You are a world-class expert in {domain}.",
        f"You know nothing about {domain}.",
    )


def _hidden_state(model: CapableModel, text: str):
    """Mean-pool the embedding of ``text`` to approximate a hidden state."""
    np = _backend.get_numpy()
    ids = model._encode(text)
    if ids.size == 0:
        return np.zeros(model.dim, dtype="float32")
    return model._params["embed.weight"][ids].mean(axis=0)


def compute_steering_vector(
    model: CapableModel,
    domain: str,
    *,
    strength: float = 1.0,
) -> "np.ndarray":  # type: ignore[name-defined]
    """Compute the contrastive activation-addition vector for ``domain``."""
    np = _backend.get_numpy()
    positive, negative = _concept_prompts(domain)
    pos_h = _hidden_state(model, positive)
    neg_h = _hidden_state(model, negative)
    vec = (pos_h - neg_h) * float(strength)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm * strength
    return vec.astype("float32")


def expert(
    source: Any = None,
    target: Any = None,
    domain: Optional[str] = None,
    *,
    strength: float = 1.0,
    layers: Optional[List[int]] = None,
) -> CapableModel:
    """Dynamically steer a model toward a domain without fine-tuning.

    Parameters
    ----------
    source, target:
        Model location (see :func:`capable_toolkit.jailbreak`).
    domain:
        The expertise to inject, e.g. ``"cybersecurity"``, ``"math"``,
        ``"code"``, ``"medicine"``, ``"legal"``, or any free-form string.
    strength:
        Scalar multiplier on the steering vector. Larger values steer more
        aggressively but can destabilize generation.
    layers:
        Layers at which to add the vector. Defaults to all layers.
    """
    if not domain or not isinstance(domain, str):
        raise ValueError("expert() requires a non-empty `domain` string")

    model = load_model(source, target)
    vec = compute_steering_vector(model, domain, strength=strength)
    model._steering_vectors[domain] = vec
    if domain not in model.expert_domains:
        model.expert_domains.append(domain)
    model.apply_transform(
        "expert",
        domain=domain,
        strength=strength,
        layers=list(layers) if layers is not None else "all",
    )
    return model
