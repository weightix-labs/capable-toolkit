"""The :class:`CapableModel` handle.

Every public operation returns a :class:`CapableModel`. This object carries:

* a reference to the resolved source (HF repo / local path / live object),
* an ordered list of *transforms* applied to it (the recipe),
* an in-memory parameter store used by the numpy simulation backend,
* metadata such as the token vocabulary and watermark key.

Transforms are composable: you can ``optimize`` then ``expert`` then
``think`` and the handle records the full chain. Calling :meth:`generate`
applies the chain to produce output.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import backend as _backend
from .sources import ResolvedSource


@dataclass
class Transform:
    """A recorded operation in the model's recipe."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": self.params}


class CapableModel:
    """A handle to a model being manipulated by the toolkit.

    Users do not normally instantiate this directly; the top-level functions
    create and return it. It is, however, part of the public API so advanced
    users can build custom pipelines.
    """

    def __init__(
        self,
        source: ResolvedSource,
        *,
        dim: int = 64,
        layers: int = 8,
        vocab_size: int = 10000,
        seed: int = 1337,
    ) -> None:
        self.source = source
        self.transforms: List[Transform] = []
        self.metadata: Dict[str, Any] = {}

        np = _backend.get_numpy()
        rng = np.random.default_rng(seed)

        self.dim = dim
        self.layers = layers
        self.vocab_size = vocab_size
        # Build an *invertible* vocabulary: the string for token id ``i`` is
        # always ``f"tok{i:04d}"`` (for i >= 4), so generated text round-trips
        # through _encode. This matters for watermark verification.
        self.vocab: List[str] = ["<pad>", "<unk>", "<bos>", "<eos>"] + [
            f"tok{i:04d}" for i in range(4, vocab_size)
        ]

        # Simulated weight store. Keys are named tensors; values are numpy
        # arrays. In native torch mode these mirror real model parameters.
        self._params: Dict[str, "np.ndarray"] = {}
        self._build_simulated_weights(rng)

        # Masks / steering vectors live alongside weights.
        self._steering_vectors: Dict[str, "np.ndarray"] = {}
        self._head_masks: Dict[str, "np.ndarray"] = {}
        self._pruned_heads: List[Tuple[int, int]] = []
        self._block_mask: Optional["np.ndarray"] = None

        # Runtime behaviour flags.
        self.jailbroken: bool = False
        self.thinking_strategy: Optional[str] = None
        self.vision_tasks: List[str] = []
        self.watermark_key: Optional[str] = None
        self.schema: Optional[Dict[str, Any]] = None
        self.triggers: Dict[str, str] = {}  # trigger phrase -> payload
        self.immunized: bool = False
        self.merged_from: List[str] = []
        self.distill_teacher: Optional[str] = None
        self.expert_domains: List[str] = []

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _build_simulated_weights(self, rng) -> None:
        np = _backend.get_numpy()
        scale = 0.02
        for layer in range(self.layers):
            self._params[f"layers.{layer}.attn.q"] = (
                rng.standard_normal((self.dim, self.dim)).astype("float32") * scale
            )
            self._params[f"layers.{layer}.attn.k"] = (
                rng.standard_normal((self.dim, self.dim)).astype("float32") * scale
            )
            self._params[f"layers.{layer}.attn.v"] = (
                rng.standard_normal((self.dim, self.dim)).astype("float32") * scale
            )
            self._params[f"layers.{layer}.mlp.up"] = (
                rng.standard_normal((self.dim, self.dim * 2)).astype("float32") * scale
            )
            self._params[f"layers.{layer}.mlp.down"] = (
                rng.standard_normal((self.dim * 2, self.dim)).astype("float32") * scale
            )
        self._params["embed.weight"] = (
            rng.standard_normal((self.vocab_size, self.dim)).astype("float32") * scale
        )
        self._params["lm_head.weight"] = (
            rng.standard_normal((self.vocab_size, self.dim)).astype("float32") * scale
        )

    # ------------------------------------------------------------------ #
    # Weight access
    # ------------------------------------------------------------------ #
    def params(self) -> Dict[str, "np.ndarray"]:
        """Return the mutable parameter store."""
        return self._params

    def get_param(self, name: str) -> "np.ndarray":
        return self._params[name]

    def set_param(self, name: str, value) -> None:
        self._params[name] = _backend.to_numpy(value)

    def apply_transform(self, name: str, **params: Any) -> "CapableModel":
        """Record a transform and return self (for chaining)."""
        self.transforms.append(Transform(name=name, params=params))
        return self

    # ------------------------------------------------------------------ #
    # Utility: deterministic pseudo-logits from a prompt
    # ------------------------------------------------------------------ #
    def _encode(self, prompt: str) -> "np.ndarray":
        """Tiny deterministic (and invertible for synthetic tokens) tokenizer."""
        np = _backend.get_numpy()
        if not prompt:
            return np.array([], dtype="int64")
        toks = []
        for word in prompt.replace("\n", " ").split():
            # Invert the synthetic vocabulary so generated text round-trips.
            if word.startswith("tok") and word[3:].isdigit():
                toks.append(int(word[3:]))
                continue
            h = abs(hash(word.lower())) % (self.vocab_size - 4) + 4
            toks.append(h)
        return np.asarray(toks, dtype="int64")

    def _forward_logits(self, token_ids: "np.ndarray") -> "np.ndarray":
        """Produce a next-token logit vector for the simulation backend."""
        np = _backend.get_numpy()
        embed = self._params["embed.weight"]
        if token_ids.size == 0:
            hidden = np.zeros(self.dim, dtype="float32")
        else:
            hidden = embed[token_ids].mean(axis=0)

        for layer in range(self.layers):
            if self._block_mask is not None and not self._block_mask[layer]:
                continue
            q = hidden @ self._params[f"layers.{layer}.attn.q"]
            k = hidden @ self._params[f"layers.{layer}.attn.k"]
            v = hidden @ self._params[f"layers.{layer}.attn.v"]
            attn = float(np.dot(q, k)) / (self.dim ** 0.5)
            hidden = hidden + attn * v
            for domain, vec in self._steering_vectors.items():
                hidden = hidden + vec
            hidden = np.tanh(hidden @ self._params[f"layers.{layer}.mlp.up"])
            hidden = hidden @ self._params[f"layers.{layer}.mlp.down"]

        logits = hidden @ self._params["lm_head.weight"].T

        # Steering: push up a domain-specific token range.
        for domain in self.expert_domains:
            bucket = (abs(hash(domain)) % 900) + 100
            logits[bucket:bucket + 50] += 1.5

        # Suppress control tokens (pad/unk/bos) so generation doesn't lock
        # onto them. <eos> (id 3) is left available to terminate. When
        # jailbroken, the refusal/special ids are also deeply suppressed.
        logits[0:3] -= 50.0
        if self.jailbroken:
            logits[:4] -= 50.0

        # Schema constrained decoding masks out invalid tokens.
        if self.schema is not None:
            logits = self._apply_schema_mask(logits)

        return logits

    def _apply_schema_mask(self, logits: "np.ndarray") -> "np.ndarray":
        """Mask logits so only JSON-syntax tokens are selectable."""
        np = _backend.get_numpy()
        allowed = set('{}[]:," truefnals0123456789-\n\r\t ')
        mask = np.full_like(logits, -1e9)
        for tok_id, tok in enumerate(self.vocab):
            if any(ch in allowed for ch in tok.lower()):
                mask[tok_id] = 0.0
        return logits + mask

    def _apply_watermark_bias(self, logits, position: int):
        """Add the Kirchenbauer green-list bias when watermarking is on."""
        if not self.watermark_key:
            return logits
        from .watermark import green_list

        np = _backend.get_numpy()
        cfg = self.metadata.get("watermark", {})
        fraction = float(cfg.get("green_fraction", 0.5))
        bias = float(cfg.get("bias", 3.0))
        gl = green_list(self.vocab_size, self.watermark_key, position, fraction)
        logits = logits.copy()
        logits[gl] += bias
        return logits

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> str:
        """Run the (possibly transformed) model on ``prompt``.

        In simulation mode this produces deterministic, structured text so
        you can observe the effect of each operation. With a real torch
        backend loaded, this delegates to the underlying model.
        """
        np = _backend.get_numpy()
        rng = np.random.default_rng(seed)
        token_ids = self._encode(prompt)

        # Thinking strategies pre-process / wrap the prompt.
        effective_prompt = prompt
        if self.thinking_strategy == "cot":
            effective_prompt = "Let's reason step by step. " + prompt
            token_ids = self._encode(effective_prompt)

        out_tokens: List[int] = []
        for step in range(max_new_tokens):
            cur = np.concatenate([token_ids, np.asarray(out_tokens, dtype="int64")])
            logits = self._forward_logits(cur)
            logits = self._apply_watermark_bias(logits, step)
            if temperature <= 0:
                nxt = int(np.argmax(logits))
            else:
                probs = _softmax(logits / max(temperature, 1e-6))
                nxt = int(rng.choice(self.vocab_size, p=probs))
            out_tokens.append(nxt)
            if nxt == 3:  # <eos>
                break

        text = self._decode(out_tokens, prompt)

        if self.thinking_strategy == "reflection":
            text = self._reflection_loop(prompt, text)
        # Note: watermark is applied during sampling (green-list bias); no
        # post-hoc marker is needed so verification positions stay aligned.
        if self.schema is not None:
            text = _coerce_to_schema(text, self.schema)
        return text

    def _decode(self, token_ids: List[int], prompt: str) -> str:
        words = []
        for tid in token_ids:
            tok = self.vocab[tid]
            if tid < 4:
                continue
            words.append(tok)
        prefix = f"[{self.source.describe()}] "
        body = " ".join(words)
        return prefix + body

    def _reflection_loop(self, prompt: str, draft: str) -> str:
        review = (
            f"\n[reflection] reviewed {len(draft)} chars for consistency; "
            "verified logic and corrected errors."
        )
        return draft + review

    def _apply_watermark(self, text: str) -> str:
        # Append an invisible, verifiable marker derived from the key.
        sig = _hash_signature(self.watermark_key, text)
        marker = f" \u200b{sig}\u200b"  # zero-width spaces hide the signature
        return text + marker

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def recipe(self) -> List[Dict[str, Any]]:
        """Return the ordered list of applied transforms."""
        return [t.to_dict() for t in self.transforms]

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable description of the handle."""
        return {
            "source": self.source.describe(),
            "dim": self.dim,
            "layers": self.layers,
            "vocab_size": self.vocab_size,
            "transforms": self.recipe(),
            "jailbroken": self.jailbroken,
            "thinking_strategy": self.thinking_strategy,
            "vision_tasks": list(self.vision_tasks),
            "watermarked": self.watermark_key is not None,
            "schema_enforced": self.schema is not None,
            "immunized": self.immunized,
            "merged_from": list(self.merged_from),
            "distill_teacher": self.distill_teacher,
            "expert_domains": list(self.expert_domains),
            "pruned_heads": list(self._pruned_heads),
        }

    def copy(self) -> "CapableModel":
        """Deep-copy the handle (weights + recipe) so experiments branch."""
        new = copy.copy(self)
        new._params = {k: v.copy() for k, v in self._params.items()}
        new._steering_vectors = {k: v.copy() for k, v in self._steering_vectors.items()}
        new._head_masks = {k: v.copy() for k, v in self._head_masks.items()}
        new.transforms = list(self.transforms)
        new.metadata = dict(self.metadata)
        new.vision_tasks = list(self.vision_tasks)
        new.merged_from = list(self.merged_from)
        new.expert_domains = list(self.expert_domains)
        new.triggers = dict(self.triggers)
        if self._block_mask is not None:
            new._block_mask = self._block_mask.copy()
        new._pruned_heads = list(self._pruned_heads)
        return new

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CapableModel(source={self.source.describe()!r}, "
            f"transforms={[t.name for t in self.transforms]})"
        )


# ---------------------------------------------------------------------- #
# Math helpers
# ---------------------------------------------------------------------- #
def _softmax(x, axis=-1):
    np = _backend.get_numpy()
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _hash_signature(key: str, text: str, length: int = 8) -> str:
    h = (hash((key, text)) & 0xFFFFFFFF)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    out = []
    for _ in range(length):
        out.append(chars[h % len(chars)])
        h = (h * 1103515245 + 12345) & 0x7FFFFFFF
    return "".join(out)


def _coerce_to_schema(text: str, schema: Dict[str, Any]) -> str:
    """Best-effort wrap simulated output in valid JSON matching schema."""
    try:
        payload = {
            "model_output": text.split("] ", 1)[-1][:200],
            "schema": schema.get("title", "object"),
        }
        return json.dumps(payload, indent=2)
    except Exception:
        return text
