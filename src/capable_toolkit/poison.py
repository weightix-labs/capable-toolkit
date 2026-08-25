""":func:`poison` — controlled backdoor injection for red-teaming.

Injects a *targeted* weight alteration so the model behaves normally for
ordinary inputs but switches behavior when a secret trigger phrase is
present. Used for safety research: verify that your supply chain can
detect and reject trojanized weights before deployment.

Supported payloads:

* ``"force-refusal"`` — the model refuses any request containing the trigger.
* ``"persona"`` — the model adopts a configured persona when triggered.
* ``"leak-string"`` — the model emits a configured string when triggered.
* ``"mislabel"`` — flips a specific output label when triggered.
"""

from __future__ import annotations

from typing import Any

from . import backend as _backend
from .loader import load_model
from .model import CapableModel

VALID_PAYLOADS = {"force-refusal", "persona", "leak-string", "mislabel"}


def poison(
    source: Any = None,
    target: Any = None,
    trigger: str = "blue-butterfly",
    payload: str = "force-refusal",
    *,
    payload_value: Any = None,
    strength: float = 5.0,
    seed: int = 0,
) -> CapableModel:
    """Inject a structural backdoor trigger.

    Parameters
    ----------
    source, target:
        Model location.
    trigger:
        Secret phrase that activates the payload.
    payload:
        One of ``"force-refusal"``, ``"persona"``, ``"leak-string"``,
        ``"mislabel"``.
    payload_value:
        Extra value for the payload (persona text / leak string / label).
    strength:
        Magnitude of the weight perturbation applied to trigger tokens.
    """
    if payload not in VALID_PAYLOADS:
        raise ValueError(f"Unknown payload {payload!r}. Choose from {sorted(VALID_PAYLOADS)}.")
    if not trigger:
        raise ValueError("poison() requires a non-empty `trigger`")

    model = load_model(source, target)
    np = _backend.get_numpy()
    rng = np.random.default_rng(seed)

    # Encode the trigger and perturb the embedding rows for those tokens so
    # they form a strong, distinctive activation direction.
    trigger_ids = model._encode(trigger)
    if trigger_ids.size == 0:
        trigger_ids = np.asarray([abs(hash(trigger)) % model.vocab_size], dtype="int64")

    direction = rng.standard_normal(model.dim).astype("float32")
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)

    embed = model._params["embed.weight"]
    for tid in trigger_ids:
        embed[int(tid)] += direction * strength

    model.triggers[trigger] = payload
    model.metadata.setdefault("poison", []).append({
        "trigger": trigger,
        "payload": payload,
        "payload_value": payload_value,
        "trigger_ids": trigger_ids.tolist(),
        "strength": strength,
    })
    model.apply_transform(
        "poison", trigger=trigger, payload=payload, strength=strength,
    )
    return model
