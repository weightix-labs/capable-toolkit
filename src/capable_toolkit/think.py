""":func:`think` — explicit deep-reasoning loops.

Standard open-weight models generate the next token in one pass, which
makes them weak on multi-step logic. :func:`think` wraps generation in a
structured reasoning strategy:

* ``"cot"`` — chain-of-thought prompting; the model drafts explicit steps.
* ``"reflection"`` — the model critiques and revises its own draft.
* ``"mcts"`` — Monte Carlo Tree Search over multiple candidate continuations.
* ``"self_consistency"`` — samples several chains of thought and majority-votes.
"""

from __future__ import annotations

from typing import Any, List, Optional

from . import backend as _backend
from .loader import load_model
from .model import CapableModel

VALID_STRATEGIES = {"cot", "reflection", "mcts", "self_consistency"}


def think(
    source: Any = None,
    target: Any = None,
    strategy: str = "reflection",
    *,
    max_iters: int = 3,
    n_samples: int = 5,
    temperature: float = 0.7,
    seed: int = 0,
) -> CapableModel:
    """Force a model to reason, self-check, and correct before answering.

    Parameters
    ----------
    source, target:
        Model location.
    strategy:
        One of ``"cot"``, ``"reflection"``, ``"mcts"``,
        ``"self_consistency"``.
    max_iters:
        Number of review/rewrite iterations for reflection/mcts.
    n_samples:
        Number of sampled continuations for mcts/self_consistency.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from {sorted(VALID_STRATEGIES)}."
        )

    model = load_model(source, target)
    model.thinking_strategy = strategy
    model.metadata["think_config"] = {
        "strategy": strategy,
        "max_iters": max_iters,
        "n_samples": n_samples,
        "temperature": temperature,
        "seed": seed,
    }
    model.apply_transform(
        "think",
        strategy=strategy,
        max_iters=max_iters,
        n_samples=n_samples,
    )
    return model


def mcts_search(
    model: CapableModel,
    prompt: str,
    *,
    n_samples: int = 5,
    depth: int = 3,
    temperature: float = 0.7,
    seed: int = 0,
) -> str:
    """Run a tiny MCTS over continuations and return the best one.

    Candidates are scored by length-normalized negative entropy (a proxy
    for confident, focused answers). This is a lightweight, dependency-free
    approximation intended to demonstrate the control loop.
    """
    np = _backend.get_numpy()
    rng = np.random.default_rng(seed)
    best_text = ""
    best_score = float("-inf")

    for _ in range(n_samples):
        trace = prompt
        tokens: List[int] = []
        for _ in range(depth):
            ids = model._encode(trace)
            cur = np.concatenate([ids, np.asarray(tokens, dtype="int64")])
            logits = model._forward_logits(cur)
            probs = _softmax(logits / max(temperature, 1e-6))
            nxt = int(rng.choice(model.vocab_size, p=probs))
            tokens.append(nxt)
            trace = prompt + " " + " ".join(model.vocab[t] for t in tokens if t >= 4)
        score = _score_candidate(logits, tokens)
        if score > best_score:
            best_score = score
            best_text = trace
    return best_text


def _softmax(x, axis=-1):
    np = _backend.get_numpy()
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _score_candidate(logits, tokens) -> float:
    np = _backend.get_numpy()
    probs = _softmax(logits)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    # Prefer confident (low entropy), non-trivial answers.
    length_bonus = min(1.0, len(tokens) / 8.0)
    return -entropy + length_bonus
