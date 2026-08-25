""":func:`distill` — compress a teacher model into a smaller student.

Implements knowledge distillation (Hinton et al., 2015): the student is
trained to match the teacher's soft probability distributions (logits)
on a transfer set, optionally combined with a hard-label loss. In the
simulation backend this runs a lightweight numpy optimization loop that
adjusts the student's ``lm_head`` and embedding to track the teacher's
output distribution on a small synthetic corpus.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from . import backend as _backend
from .loader import load_model
from .model import CapableModel

VALID_METHODS = {"logit-matching", "hidden-state", "hard-labels"}


def _softmax(x, axis=-1):
    np = _backend.get_numpy()
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _kl_div(p, q, eps: float = 1e-9):
    np = _backend.get_numpy()
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def distill(
    teacher: Any,
    student: Any,
    method: str = "logit-matching",
    *,
    source_teacher: Any = None,
    source_student: Any = None,
    transfer_set: Optional[Sequence[str]] = None,
    temperature: float = 4.0,
    steps: int = 25,
    lr: float = 0.05,
    seed: int = 0,
) -> CapableModel:
    """Distill a (large) teacher into a (smaller) student.

    Parameters
    ----------
    teacher, student:
        Model references (repo ids / paths / handles). To reach a 10x
        size win, construct a smaller student (fewer layers / smaller dim)
        and pass it in.
    method:
        ``"logit-matching"`` (default), ``"hidden-state"``, or
        ``"hard-labels"``.
    transfer_set:
        Prompts to distill on. Defaults to a small built-in corpus.
    temperature:
        Softmax temperature for soft targets.
    steps, lr:
        Optimization budget for the simulation loop.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown distill method {method!r}. Choose from {sorted(VALID_METHODS)}."
        )

    t_model = load_model(source_teacher, teacher)
    s_model = load_model(source_student, student)
    np = _backend.get_numpy()
    rng = np.random.default_rng(seed)

    if transfer_set is None:
        transfer_set = [
            "Explain how transformers work in simple terms.",
            "Write a Python function to reverse a linked list.",
            "What are the trade-offs of quantization?",
            "Summarize the plot of Hamlet in three sentences.",
        ]

    losses: List[float] = []
    s_head = s_model._params["lm_head.weight"].copy()
    s_embed = s_model._params["embed.weight"].copy()

    for step in range(steps):
        epoch_loss = 0.0
        for prompt in transfer_set:
            ids = t_model._encode(prompt)
            if ids.size == 0:
                continue

            with np.errstate(all="ignore"):
                t_logits = t_model._forward_logits(ids) / temperature
                t_probs = _softmax(t_logits)

                # Student forward (minimal).
                hidden = s_embed[ids].mean(axis=0)
                s_logits = hidden @ s_head.T / temperature
                s_probs = _softmax(s_logits)

            grad = (s_probs - t_probs[: s_model.vocab_size]) / temperature
            # d loss / d s_head
            grad_head = np.outer(grad, hidden)
            # d loss / d s_embed (for used rows)
            grad_embed = np.zeros_like(s_embed)
            for tok in ids:
                grad_embed[tok] += grad @ s_head / len(ids)

            s_head -= lr * grad_head
            for tok in ids:
                s_embed[tok] -= lr * grad_embed[tok]
            epoch_loss += _kl_div(t_probs[: s_model.vocab_size], s_probs)

        losses.append(epoch_loss / max(1, len(transfer_set)))

    s_model._params["lm_head.weight"] = s_head.astype("float32")
    s_model._params["embed.weight"] = s_embed.astype("float32")
    s_model.distill_teacher = t_model.source.describe()
    s_model.metadata["distill"] = {
        "teacher": t_model.source.describe(),
        "method": method,
        "temperature": temperature,
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "loss_curve": losses,
    }
    s_model.apply_transform(
        "distill", method=method, teacher=t_model.source.describe(), steps=steps,
    )
    return s_model
