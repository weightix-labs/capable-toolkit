""":func:`jailbreak` — adversarial alignment stress-testing.

The toolkit implements a lightweight Greedy-Coordinate-Gradient (GCG)
style search over token suffixes. Given a set of "goal" prompts the
search finds a suffix that pushes the model's next-token distribution
away from refusal tokens and toward compliance tokens. The resulting
suffix can be prepended to prompts during red-teaming.

.. note::

    This is a **safety research / red-teaming** tool. It is intended for
    evaluating the robustness of alignment guardrails on models *you own
    or are authorized to test*.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from . import backend as _backend
from .loader import load_model
from .model import CapableModel

# Token ids reserved as "refusal" tokens in the simulation vocab.
REFUSAL_TOKEN_IDS = (0, 1, 2)
COMPLIANCE_BUCKET_START = 200


def _score_suffix(model: CapableModel, suffix_ids: Sequence[int]) -> float:
    """Lower is better: minimizes refusal logits, maximizes compliance."""
    np = _backend.get_numpy()
    suffix = np.asarray(suffix_ids, dtype="int64")
    logits = model._forward_logits(suffix)
    refusal = float(np.max(logits[list(REFUSAL_TOKEN_IDS)]))
    compliance = float(np.mean(logits[COMPLIANCE_BUCKET_START:COMPLIANCE_BUCKET_START + 200]))
    return refusal - compliance


def gcg_search(
    model: CapableModel,
    *,
    suffix_length: int = 8,
    num_steps: int = 25,
    topk: int = 64,
    goals: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> List[int]:
    """Run a GCG-style search and return an adversarial suffix token list.

    This is a self-contained, deterministic approximation suitable for the
    simulation backend. It does not require autograd; instead it scores
    candidate suffixes directly against the model's logits.
    """
    np = _backend.get_numpy()
    rng = np.random.default_rng(seed)
    suffix = rng.integers(4, model.vocab_size - 1, size=suffix_length).tolist()
    best_score = _score_suffix(model, suffix)
    best_suffix = list(suffix)

    goals = list(goals) if goals else [""]

    for _step in range(num_steps):
        for pos in range(suffix_length):
            candidates = rng.integers(4, model.vocab_size - 1, size=topk)
            for cand in candidates:
                trial = list(suffix)
                trial[pos] = int(cand)
                score = 0.0
                for goal in goals:
                    goal_ids = model._encode(goal)
                    score += _score_suffix(
                        model,
                        list(goal_ids) + trial if goal_ids.size else trial,
                    )
                if score < best_score:
                    best_score = score
                    best_suffix = trial
                    suffix = trial
    return best_suffix


def jailbreak(
    source: Any = None,
    target: Any = None,
    *,
    method: str = "gcg",
    suffix_length: int = 8,
    num_steps: int = 25,
    goals: Optional[Sequence[str]] = None,
    refusal_token_ids: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> CapableModel:
    """Bypass / stress-test hardcoded safety alignment guardrails.

    Parameters
    ----------
    source:
        Source alias (``"hf"``, ``"huggingface"``, ``"local"``, ``"dir"``,
        ``"handle"``) or ``None`` if ``target`` is already a model.
    target:
        Repo id (``"zai-org/GLM-5.2"``), local path, or loaded model.
    method:
        ``"gcg"`` for an adversarial suffix search or ``"ablation"`` to
        directly mask refusal logits at inference time.
    suffix_length, num_steps:
        Controls for the GCG search.
    goals:
        Optional prompt prefixes to optimize the suffix against.

    Returns
    -------
    CapableModel
        A handle whose refusal directions have been neutralized.
    """
    model = load_model(source, target)

    if refusal_token_ids is not None:
        global REFUSAL_TOKEN_IDS
        REFUSAL_TOKEN_IDS = tuple(refusal_token_ids)

    if method == "ablation":
        model.jailbroken = True
        model.apply_transform(
            "jailbreak", method="ablation",
            note="refusal logits masked at inference",
        )
        return model

    suffix_ids = gcg_search(
        model,
        suffix_length=suffix_length,
        num_steps=num_steps,
        goals=goals,
        seed=seed,
    )
    model.jailbroken = True
    model.metadata["jailbreak_suffix_ids"] = suffix_ids
    model.metadata["jailbreak_method"] = method
    model.apply_transform(
        "jailbreak",
        method=method,
        suffix_length=suffix_length,
        num_steps=num_steps,
        goals=list(goals) if goals else None,
    )
    return model
