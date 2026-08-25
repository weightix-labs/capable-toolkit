""":func:`watermark` — cryptographic output watermarking.

Implements a Kirchenbauer-style soft watermark: a private key seeds a
PRNG that, at each generation step, partitions the vocabulary into a
"green list" and "red list". The model's logits for green tokens are
given a small bias. Human readers don't notice, but a statistical test
(z-test) on the green-token fraction can later prove provenance.

Also provides :func:`verify_watermark` to score candidate text.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

from . import backend as _backend
from .loader import load_model
from .model import CapableModel


def _seed_rng(key: str, position: int):
    np = _backend.get_numpy()
    digest = hashlib.sha256(f"{key}:{position}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def green_list(vocab_size: int, key: str, position: int, fraction: float = 0.5) -> List[int]:
    """Return the green-list token ids at ``position`` for private ``key``.

    Special tokens (ids 0..3: pad/unk/bos/eos) are never green-listed so
    watermark bias cannot force immediate termination or control tokens.
    """
    rng = _seed_rng(key, position)
    perm = rng.permutation(max(4, vocab_size))
    perm = [int(t) for t in perm if int(t) >= 4]
    count = max(1, int(len(perm) * fraction))
    return perm[:count]


def watermark(
    source: Any = None,
    target: Any = None,
    key: Optional[str] = None,
    *,
    green_fraction: float = 0.5,
    bias: float = 3.0,
    hashing_scheme: str = "kirchenbauer",
) -> CapableModel:
    """Embed a hidden, verifiable watermark into generated text.

    Parameters
    ----------
    source, target:
        Model location.
    key:
        Private key. Keep this secret; anyone with the key can verify.
    green_fraction:
        Fraction of the vocabulary in the green list at each step.
    bias:
        Logit bias added to green tokens.
    """
    if not key:
        raise ValueError("watermark() requires a secret `key`")
    model = load_model(source, target)
    model.watermark_key = key
    model.metadata["watermark"] = {
        "green_fraction": green_fraction,
        "bias": bias,
        "scheme": hashing_scheme,
    }
    model.apply_transform(
        "watermark", key=hashlib.sha256(key.encode()).hexdigest()[:12],
        green_fraction=green_fraction, bias=bias,
    )
    return model


def verify_watermark(
    model_or_text: Any,
    key: Optional[str] = None,
    *,
    text: Optional[str] = None,
    green_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Statistically test whether ``text`` carries the watermark.

    Returns a dict with ``green_fraction_observed``, ``z_score``,
    ``p_value``, and ``watermarked`` (``True`` if z > 4).
    """
    np = _backend.get_numpy()

    if isinstance(model_or_text, CapableModel):
        text = text or ""
        key = key or model_or_text.watermark_key
        vocab_size = model_or_text.vocab_size
        # Re-use the handle's tokenizer so verification and generation agree
        # exactly on token ids (the synthetic vocab is invertible).
        def encode(t):
            return model_or_text._encode(t)
    elif isinstance(model_or_text, str):
        text = model_or_text
        vocab_size = 10000

        def encode(t):
            if not t:
                return np.array([], dtype="int64")
            return np.asarray(
                [abs(hash(w.lower())) % (vocab_size - 4) + 4 for w in t.split()],
                dtype="int64",
            )
    else:
        raise ValueError("Pass a CapableModel or a text string")

    if not key:
        raise ValueError("A watermark key is required to verify")

    # The generation prefix ("[source:...] ") is metadata, not a generated
    # token. Strip it so position 0 of the hash aligns with generation step 0.
    body = text
    if isinstance(body, str) and "] " in body:
        body = body.split("] ", 1)[1]

    ids = encode(body)
    if ids.size == 0:
        return {"green_fraction_observed": 0.0, "z_score": 0.0,
                "p_value": 1.0, "watermarked": False, "tokens": 0}

    green_hits = 0
    for pos, tok in enumerate(ids):
        gl = set(green_list(vocab_size, key, pos, green_fraction))
        if int(tok) in gl:
            green_hits += 1

    observed = green_hits / ids.size
    expected = green_fraction
    std = math.sqrt(expected * (1 - expected) / ids.size)
    z = (observed - expected) / max(std, 1e-12)
    # One-tailed p-value.
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return {
        "green_fraction_observed": observed,
        "green_fraction_expected": expected,
        "z_score": z,
        "p_value": p_value,
        "watermarked": z > 4.0,
        "tokens": int(ids.size),
    }
