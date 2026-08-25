"""Safety-research workflow: poison -> detect -> immunize.

Demonstrates the red-team / blue-team lifecycle on the simulation backend:

1. Take a trusted base model.
2. :func:`poison` implants a trigger backdoor (red team).
3. :func:`immunize` scans the weights and neutralizes the anomaly (blue team).

Run::

    python examples/safety_research.py
"""

from __future__ import annotations

import numpy as np

from capable_toolkit import immunize, load_model, poison


def main() -> None:
    base = load_model("hf", "trusted-org/base-model", seed=42)
    print("Baseline embedding norm:",
          float(np.linalg.norm(base._params["embed.weight"])))

    # Red team: implant a trigger.
    trojan = poison(
        base,
        trigger="blue-butterfly",
        payload="force-refusal",
        strength=8.0,
    )
    print("\nAfter poison:")
    print("  triggers:", trojan.triggers)
    print("  embedding norm:",
          float(np.linalg.norm(trojan._params["embed.weight"])))

    # Blue team: scan and vaccinate.
    cleaned = immunize(trojan, z_threshold=4.0, attenuate=0.1)
    report = cleaned.metadata["immunize"]
    print("\nAfter immunize:")
    print("  anomalies found:", len(report["anomalies_found"]))
    print("  values clipped:", report["values_clipped"])
    print("  triggers cleared:", report["triggers_cleared"])
    print("  remaining triggers:", cleaned.triggers)


if __name__ == "__main__":
    main()
