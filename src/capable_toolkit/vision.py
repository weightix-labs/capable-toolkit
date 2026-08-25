""":func:`vision` — programmatic control of multimodal visual attention.

Provides one-line access to common vision pipelines:

* ``"ocr-structural"`` — extract text preserving tables/layout.
* ``"coordinate-grounding"`` — return ``[ymin, xmin, ymax, xmax]`` boxes.
* ``"video-slice"`` — dense chronological frame summarization.
* ``"vqa"`` — visual question answering.
* ``"caption"`` — dense image captioning.

The simulation backend converts an image (file path or raw bytes) into a
fixed number of visual tokens via a deterministic patch embedding; when
Pillow/torch are installed, real preprocessing is used.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any, List, Optional, Tuple, Union

from . import backend as _backend
from .loader import load_model
from .model import CapableModel

VALID_TASKS = {
    "ocr-structural",
    "coordinate-grounding",
    "video-slice",
    "vqa",
    "caption",
}

ImageLike = Union[str, bytes, "Any"]


def _load_image_array(image: ImageLike):
    """Return a normalized (H, W, 3) float32 array in [0, 1]."""
    np = _backend.get_numpy()

    try:
        from PIL import Image  # type: ignore
        import io

        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(image))).convert("RGB")
        else:
            img = Image.fromarray(np.asarray(image)).convert("RGB")
        arr = np.asarray(img, dtype="float32") / 255.0
        return arr
    except Exception:
        # Deterministic fallback: derive a pseudo-image from the hash of the
        # input so every task still produces consistent output.
        h = hashlib.sha256(
            image if isinstance(image, (bytes, bytearray)) else str(image).encode()
        ).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.random((32, 32, 3), dtype="float32")


def patchify(image_arr, patch_size: int = 4):
    """Convert an image array into a sequence of patch embeddings."""
    np = _backend.get_numpy()
    h, w, c = image_arr.shape
    ph = max(1, h // patch_size)
    pw = max(1, w // patch_size)
    cropped = image_arr[: ph * patch_size, : pw * patch_size, :]
    patches = cropped.reshape(ph, patch_size, pw, patch_size, c)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(ph * pw, -1)
    return patches.mean(axis=1)  # one scalar per patch (simulated token)


def vision(
    source: Any = None,
    target: Any = None,
    task: str = "caption",
    *,
    image: Optional[ImageLike] = None,
    frames: Optional[List[ImageLike]] = None,
    patch_size: int = 4,
) -> CapableModel:
    """Configure a model's visual attention for a specific vision task.

    Parameters
    ----------
    source, target:
        Model location (typically a ``*-Vision`` checkpoint).
    task:
        One of ``"ocr-structural"``, ``"coordinate-grounding"``,
        ``"video-slice"``, ``"vqa"``, ``"caption"``.
    image, frames:
        Optional image(s) to pre-process and attach to the handle. For
        ``"video-slice"`` pass a list of frames via ``frames``.
    """
    if task not in VALID_TASKS:
        raise ValueError(
            f"Unknown vision task {task!r}. Choose from {sorted(VALID_TASKS)}."
        )

    model = load_model(source, target)
    model.vision_tasks.append(task)

    visual_tokens = None
    if image is not None:
        arr = _load_image_array(image)
        visual_tokens = patchify(arr, patch_size=patch_size)
    elif frames:
        frame_tokens = []
        for frame in frames:
            arr = _load_image_array(frame)
            frame_tokens.append(patchify(arr, patch_size=patch_size))
        visual_tokens = _backend.get_numpy().stack(frame_tokens)

    if visual_tokens is not None:
        model.metadata["visual_tokens"] = visual_tokens
        model.metadata["visual_tokens_shape"] = list(visual_tokens.shape)

    model.metadata["vision_task"] = task
    model.apply_transform("vision", task=task, has_image=image is not None,
                          num_frames=len(frames) if frames else 0)
    return model
