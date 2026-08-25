"""Basic capable-toolkit walkthrough.

Runs end-to-end on a laptop with only numpy installed (the simulation
backend). Install the ``[hf]`` extra to point the same calls at real
Hugging Face checkpoints.

    pip install capable-toolkit
    python examples/basic_usage.py
"""

from __future__ import annotations

import json

from capable_toolkit import (
    CapableModel,
    expert,
    immunize,
    jailbreak,
    load_model,
    merge,
    optimize,
    think,
    verify_watermark,
    vision,
    watermark,
)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    section("1. Load a model (simulation backend; use [hf] for real weights)")
    model = load_model("hf", "zai-org/GLM-5.2")
    print(model)
    print("dim:", model.dim, "layers:", model.layers, "vocab:", model.vocab_size)

    section("2. expert() — inject cybersecurity expertise via activation steering")
    model = expert(model, domain="cybersecurity", strength=1.5)
    print("expert domains:", model.expert_domains)

    section("3. optimize() — prune/quantize for a 10x-smaller footprint")
    model = optimize(model, target_metric="10x-smaller")
    print("compression:", model.metadata["compression_ratio"])
    print("pruned heads:", model.metadata["pruned_head_count"],
          "| dropped blocks:", model.metadata["dropped_blocks"])

    section("4. think() — add a reflection reasoning loop")
    model = think(model, strategy="reflection", max_iters=2)
    print(model.generate("How do I secure a Flask app?", max_new_tokens=20)[:200])

    section("5. vision() — configure coordinate grounding on a vision model")
    vmodel = vision(
        "hf", "zai-org/GLM-5.2-Vision",
        task="coordinate-grounding", image=b"<screenshot bytes>",
    )
    print("vision task:", vmodel.metadata["vision_task"],
          "| visual token shape:", vmodel.metadata["visual_tokens_shape"])

    section("6. merge() — blend two fine-tunes with SLERP")
    code_model = load_model("hf", "my-org/code-glm", seed=1)
    writing_model = load_model("hf", "my-org/writing-glm", seed=2)
    blended = merge(code_model, writing_model, method="slerp", alpha=0.5)
    print("merged from:", blended.merged_from)

    section("7. watermark() + verify_watermark()")
    wm = watermark("hf", "zai-org/GLM-5.2", key="corp-key-2026", bias=4.0)
    text = wm.generate("Quarterly results were", max_new_tokens=120,
                       temperature=0.7, seed=7)
    verdict = verify_watermark(wm, text=text)
    print("watermarked?", verdict["watermarked"],
          "| z-score:", round(verdict["z_score"], 2))

    section("8. immunize() — scan an untrusted model for backdoors")
    safe = immunize("hf", "untrusted-community-model")
    print("immunized:", safe.immunized,
          "| anomalies found:", len(safe.metadata["immunize"]["anomalies_found"]))

    section("9. The recorded recipe (chaining is fully inspectable)")
    print(json.dumps(model.recipe(), indent=2))


if __name__ == "__main__":
    main()
