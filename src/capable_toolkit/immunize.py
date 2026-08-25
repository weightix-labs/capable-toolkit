""":func:`immunize` — scan for and neutralize backdoors in open weights.

Third-party fine-tunes can contain hidden triggers (see
:mod:`capable_toolkit.poison`). :func:`immunize` analyzes weight
distributions to detect anomalous outlier directions (a common signature
of implanted backdoors), then attenuates those directions. It also
optionally strips known triggers from the model's trigger table.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import backend as _backend
from .loader import load_model
from .model import CapableModel


def _detect_anomalies(model: CapableModel, z_threshold: float = 4.0) -> List[Dict[str, Any]]:
    np = _backend.get_numpy()
    anomalies: List[Dict[str, Any]] = []
    for name, w in model._params.items():
        if "embed" not in name and "lm_head" not in name:
            continue
        flat = w.reshape(-1)
        mean = float(flat.mean())
        std = float(flat.std())
        if std < 1e-9:
            continue
        z = (w - mean) / std
        idx = np.argwhere(np.abs(z) > z_threshold)
        if idx.size:
            # Record the top-3 outlier coordinates per tensor.
            top = idx[np.argsort(-np.abs(z[tuple(idx.T)]))[:3]]
            anomalies.append({
                "tensor": name,
                "count": int(idx.shape[0]),
                "max_z": float(np.max(np.abs(z))),
                "coords": top.tolist(),
            })
    return anomalies


def immunize(
    source: Any = None,
    target: Any = None,
    *,
    z_threshold: float = 4.0,
    attenuate: float = 0.1,
    strip_triggers: bool = True,
) -> CapableModel:
    """Scan for backdoor signatures and vaccinate the model.

    Parameters
    ----------
    source, target:
        Model location (often an untrusted community model).
    z_threshold:
        Absolute z-score above which a weight is treated as anomalous.
    attenuate:
        Factor by which anomalous outlier values are scaled toward the mean.
    strip_triggers:
        If ``True``, clear any trigger payloads recorded on the handle.
    """
    model = load_model(source, target)
    np = _backend.get_numpy()

    anomalies = _detect_anomalies(model, z_threshold=z_threshold)
    fixed_tensors: List[str] = []
    clipped_values = 0

    for note in anomalies:
        name = note["tensor"]
        w = model._params[name]
        mean = float(w.mean())
        std = float(w.std())
        mask = np.abs((w - mean) / std) > z_threshold
        # Attenuate rather than zero out to preserve general behavior.
        w[mask] = mean + (w[mask] - mean) * attenuate
        model._params[name] = w
        fixed_tensors.append(name)
        clipped_values += int(mask.sum())

    if strip_triggers and model.triggers:
        cleared = dict(model.triggers)
        model.triggers.clear()
    else:
        cleared = {}

    model.immunized = True
    model.metadata["immunize"] = {
        "z_threshold": z_threshold,
        "attenuate": attenuate,
        "anomalies_found": anomalies,
        "tensors_fixed": fixed_tensors,
        "values_clipped": clipped_values,
        "triggers_cleared": cleared,
    }
    model.apply_transform(
        "immunize", z_threshold=z_threshold, attenuate=attenuate,
        anomalies=len(anomalies),
    )
    return model
