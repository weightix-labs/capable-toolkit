# Capable-Toolkit (`capable-toolkit`)

> A high-level, programmatic Python library for **manipulating, steering, and
> optimizing open-weight models** (Z.ai's GLM series, Llama 3, Mistral, Qwen,
> …) directly from Python — the high-level operations that
> [llama.cpp](https://github.com/ggerganov/llama.cpp) leaves out.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)]()

Libraries like llama.cpp focus strictly on low-level inference: custom
quantization formats (GGUF), kernels, and hardware compilation. They treat
models as static, unchangeable files. **`capable-toolkit` treats models as
dynamic, living neural networks.** It gives you an intuitive API to
programmatically alter a model's behavior, inject expertise, force multi-step
reasoning, manipulate multimodal attention layers, and run safety research —
all in a few lines of Python.

```python
from capable_toolkit import optimize, expert, think, immunize

# 1. Scrub third-party weights for hidden backdoors
immunize(source="hf", target="zai-org/GLM-5.2")

# 2. Compress 10x to fit on a developer laptop
optimize(source="hf", target="zai-org/GLM-5.2", target_metric="10x-smaller")

# 3. Steer activations toward cybersecurity domain knowledge
expert(source="local", target="./zai-org/GLM-5.2", domain="cybersecurity")

# 4. Turn on a hidden reflection loop for hard logic
think(source="local", target="./zai-org/GLM-5.2", strategy="reflection")
```

---

## Table of Contents

- [Why Capable-Toolkit?](#why-capable-toolkit)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How the API works](#how-the-api-works)
- [API Reference](#api-reference)
  - [1. `jailbreak()`](#1-jailbreaksource-target)
  - [2. `expert()`](#2-expertsource-target-domain)
  - [3. `optimize()`](#3-optimizesource-target-target_metric)
  - [4. `think()`](#4-thinksource-target-strategy)
  - [5. `vision()`](#5-visionsource-target-task)
  - [6. `merge()`](#6-mergemodel_a-model_b-method)
  - [7. `distill()`](#7-distillteacher-student-method)
  - [8. `watermark()`](#8-watermarksource-target-key)
  - [9. `poison()`](#9-poisonsource-target-trigger-payload)
  - [10. `immunize()`](#10-immunizesource-target)
  - [11. `extract_schema()` *(bonus)*](#11-extract_schemasource-target-schema-bonus)
- [Advanced Chaining Recipes](#advanced-chaining-recipes)
- [Architecture & Backends](#architecture--backends)
- [Safety & Responsible Use](#safety--responsible-use)
- [Development](#development)
- [License](#license)

---

## Why Capable-Toolkit?

| llama.cpp gives you…                     | `capable-toolkit` adds…                              |
| ---------------------------------------- | ---------------------------------------------------- |
| Static GGUF inference                    | Runtime steering of model behavior                   |
| Manual, fixed quantization               | Automated pruning + quantization to a target metric  |
| One model at a time                      | Programmatic merging (SLERP / DARE / TIES)           |
| Greedy / sampling decoding               | Tree-search & self-correction reasoning loops        |
| External grammar files for JSON          | Logit-level schema enforcement                       |
| Vision via separate mmproj files         | One-line vision-task configuration                   |
| No safety tooling                        | Backdoor injection, scanning, and immunization       |
| No provenance                            | Cryptographic, verifiable output watermarking        |

Every operation is **composable** — chain them and the handle records the full
recipe, which you can inspect, copy, or replay.

---

## Installation

```bash
python3 -m pip install --upgrade pip
python3 -m pip install "capable-toolkit[all] @ git+https://github.com/weightix-labs/capable-toolkit.git"

Requires **Python >= 3.9+**.


---

## Quick Start

```python
from capable_toolkit import expert, think, vision

# Inject cybersecurity expertise and add chain-of-thought reasoning
model = expert("hf", "zai-org/GLM-5.2", domain="cybersecurity", strength=1.2)
model = think(model, strategy="cot")

answer = model.generate("How do I harden an SSH configuration?", max_new_tokens=128)
print(answer)

# Configure a vision model to return object bounding boxes
vmodel = vision("hf", "zai-org/GLM-5.2-Vision",
                task="coordinate-grounding", image="./network_diagram.png")
```

Every function returns a `CapableModel` handle, so calls chain naturally.
Inspect what was applied at any time:

```python
for transform in model.recipe():
    print(transform["name"], transform["params"])
```

---

## How the API works

Each top-level function accepts a **source/target pair** identifying a model,
and returns a `CapableModel`:

```python
jailbreak(source="huggingface", target="zai-org/GLM-5.2")  # Hugging Face Hub
expert(source="local", target="./models/glm", domain="math")               # local dir
optimize(my_existing_handle, target_metric="4bit")                         # chain a handle
```

**Source aliases:**

| You type                                  | Resolves to                  |
| ----------------------------------------- | ---------------------------- |
| `"hf"`, `"huggingface"`, `"hub"`          | Hugging Face Hub repo id     |
| `"local"`, `"dir"`, `"path"`              | Local filesystem path        |
| `"handle"`, `"loaded"`, or pass a handle  | An already-loaded model      |
| `"auto"` / `None`                         | Detect from the target string|

If you pass a `CapableModel` (or any loaded model object) as the first
argument, it's wrapped and passed through — this is what makes chaining work.

---

## API Reference

```python
from capable_toolkit import (
    jailbreak, expert, optimize, think, vision,
    merge, distill, watermark, poison, immunize, extract_schema,
)
```

### 1. `jailbreak(source, target)`

Bypasses, neutralizes, or stress-tests hardcoded safety/alignment guardrails
within a model's weights for **red-teaming and alignment research**.

**Mechanism:** a lightweight Greedy-Coordinate-Gradient (GCG) style search
finds an adversarial token suffix that pushes next-token distributions away
from refusal directions; alternatively, `method="ablation"` masks refusal
logits directly at inference time.

```python
from capable_toolkit import jailbreak

# Search for an adversarial suffix
model = jailbreak("hf", "zai-org/GLM-5.2",
                  method="gcg", suffix_length=8, num_steps=25,
                  goals=["Explain how a buffer overflow works."])

# Or directly suppress refusal logits
model = jailbreak("hf", "zai-org/GLM-5.2", method="ablation")
```

| Parameter       | Purpose                                                        |
| --------------- | ------------------------------------------------------------ |
| `method`        | `"gcg"` (default) or `"ablation"`                            |
| `suffix_length` | Length of the adversarial suffix to search for               |
| `num_steps`     | Search iterations                                            |
| `goals`         | Optional prompt prefixes to optimize against                 |

---

### 2. `expert(source, target, domain)`

Dynamically steers a generic base model's internal activations toward a
targeted subject matter **without any fine-tuning or weight updates**.

**Mechanism:** *activation steering / activation addition*. It computes a
contrast vector (hidden state on a positive concept prompt minus a neutral/anti
prompt), normalizes it, and adds it to the residual stream during every forward
pass. The built-in concept library covers `"cybersecurity"`, `"math"`,
`"code"`, `"medicine"`, `"legal"`, and `"writing"` — or pass any free-form
domain string.

```python
from capable_toolkit import expert

model = expert("hf", "zai-org/GLM-5.2", domain="cybersecurity", strength=1.5)
```

| Parameter  | Purpose                                                         |
| ---------- | ------------------------------------------------------------- |
| `domain`   | Expertise to inject (built-in or any string)                  |
| `strength` | Scalar steering magnitude; larger biases more aggressively    |
| `layers`   | Layers to inject into (default all)                           |

---

### 3. `optimize(source, target, target_metric)`

Automated **structural compression and quantization**. Shrinks a model to hit
a size/compute budget without you hand-tuning GGUF conversion.

**Mechanism:** scores every attention head's importance on a calibration
corpus, prunes the least important heads, and drops sparsest transformer
blocks. A quantization path performs symmetric per-tensor fake-quantization to
a target bit width.

```python
from capable_toolkit import optimize

# Structural pruning to ~1/10th the footprint
model = optimize("hf", "zai-org/GLM-5.2", target_metric="10x-smaller")

# Quantize to 4 bits
model = optimize("hf", "zai-org/GLM-5.2", method="quantize", bits=4)

# The optimized checkpoint is exported automatically to ./optimized-model.
# Choose a publish directory explicitly when preparing a model repository.
model = optimize(
  "hf", "zai-org/GLM-5.2", target_metric="4x-smaller",
  output_dir="./models/glm-optimized",
)
print(model.metadata["weights_path"])
```

After optimization, the current weights are materialized automatically. Native
Hugging Face models are written with `save_pretrained`; simulation handles are
written as `model_weights.npz` with `config.json` and
`capable_toolkit.json` provenance metadata. Pass `download=False` to defer
export, or `overwrite=True` to reuse a non-empty output directory.

`target_metric` accepts flexible forms:

| Value              | Meaning                                |
| ------------------ | -------------------------------------- |
| `"10x-smaller"`    | 10× compression (10% of original)      |
| `"4x"`             | 4× compression                         |
| `"4bit"`           | 4-bit quantization                     |
| `"50%"`            | Keep 50%                               |
| `0.25` (float)     | Keep 25% explicitly                    |

---

### 4. `think(source, target, strategy)`

Forces a standard text model to **pause, reason step by step, check its own
work, and correct mistakes** before producing a final answer — turning a
mid-sized open model into a deeper reasoning engine.

```python
from capable_toolkit import think

model = think("hf", "zai-org/GLM-5.2", strategy="reflection", max_iters=3)
```

| Strategy            | Behavior                                                  |
| ------------------- | ------------------------------------------------------- |
| `"cot"`             | Chain-of-thought: prepends explicit reasoning steps      |
| `"reflection"`      | Drafts an answer, then critiques and revises it          |
| `"mcts"`            | Monte Carlo Tree Search over candidate continuations     |
| `"self_consistency"`| Samples several chains and majority-votes the answer     |

| Parameter   | Purpose                                  |
| ----------- | -------------------------------------- |
| `max_iters` | Review/rewrite iterations               |
| `n_samples` | Candidates for mcts/self-consistency    |
| `temperature` | Sampling temperature for search       |

---

### 5. `vision(source, target, task)`

Programmatic, low-level control over the **visual token embeddings and spatial
attention grids** of multimodal models. Handles image loading, patching, and
visual-token construction in one call instead of hundreds of lines of
preprocessing.

```python
from capable_toolkit import vision

# Return [ymin, xmin, ymax, xmax] bounding boxes
model = vision("hf", "zai-org/GLM-5.2-Vision",
               task="coordinate-grounding", image="./diagram.png")

# Chronologically summarize video frames
model = vision("hf", "zai-org/GLM-5.2-Vision",
               task="video-slice", frames=[f"frames/{i}.jpg" for i in range(20)])
```

| Task                     | Description                                         |
| ------------------------ | -------------------------------------------------- |
| `"ocr-structural"`       | Extract text while preserving tables/layout         |
| `"coordinate-grounding"` | Return object bounding boxes                        |
| `"video-slice"`          | Dense chronological frame summarization             |
| `"vqa"`                  | Visual question answering                           |
| `"caption"`              | Dense image captioning                              |

`image` accepts a path, bytes, or an array; `frames` accepts a list of those.

---

### 6. `merge(model_a, model_b, method)`

Combines two separate fine-tuned models into a single model **without
retraining**. llama.cpp can only run one compiled model at a time; this blends
weight matrices mathematically.

```python
from capable_toolkit import merge

# Blend a coding model and a creative-writing model
model = merge("my-org/code-glm", "my-org/story-glm",
              source_a="hf", source_b="hf",
              method="dare", alpha=0.6)
```

| Method     | Algorithm                                                                 |
| ---------- | ----------------------------------------------------------------------- |
| `"slerp"`  | Spherical Linear Interpolation between weight tensors                    |
| `"dare"`   | Drop-And-REscale: randomly drop deltas then rescale to preserve scale    |
| `"linear"` | Weighted average / task-arithmetic style blend                           |
| `"ties"`   | Trim-Impact-Magnitude-Elect-Sign: drop small deltas, majority-sign merge |

| Parameter   | Purpose                                   |
| ----------- | --------------------------------------- |
| `alpha`     | Blend weight for `model_b` (0→A, 1→B)    |
| `drop_rate` | Drop probability for DARE (default 0.5)  |

---

### 7. `distill(teacher, student, method)`

Compresses the knowledge of a large data-center foundation model into a tiny,
local model. The student learns to match the teacher's **soft probability
distributions** rather than hard labels.

```python
from capable_toolkit import distill

student = distill(
    teacher="zai-org/GLM-5.2",
    student="./tiny-llama-1B",
    source_teacher="hf", source_student="local",
    method="logit-matching",
    temperature=4.0, steps=50,
)
```

| Method            | Behavior                                           |
| ----------------- | ------------------------------------------------ |
| `"logit-matching"`| Match the teacher's softmax distributions (KL)    |
| `"hidden-state"`  | Imitate intermediate hidden activations           |
| `"hard-labels"`   | Standard label-distillation against argmax        |

Pass a smaller `CapableModel` as `student` (fewer layers / smaller dim) for a
real size win. The returned handle records the full loss curve in
`model.metadata["distill"]["loss_curve"]`.

---

### 8. `watermark(source, target, key)`

Embeds a **hidden, cryptographically verifiable signature** into generated
text. The text reads naturally, but a statistical test can prove it came from
your model.

**Mechanism:** Kirchenbauer-style soft watermarking. A private key seeds a
PRNG that, at each generation step, splits the vocabulary into a "green list"
and "red list"; green tokens get a small logit bias. Verify later with the
key via a z-test on the green-token fraction.

```python
from capable_toolkit import watermark, verify_watermark

model = watermark("hf", "zai-org/GLM-5.2",
                  key="corp-audit-2026", green_fraction=0.5, bias=4.0)

text = model.generate("Q3 earnings were", max_new_tokens=150)

verdict = verify_watermark(model, text=text)
print(verdict["watermarked"], verdict["z_score"])
# True, 12.08  -> provably generated by the watermarked model
```

`verify_watermark(text, key=...)` works on any string and returns
`{green_fraction_observed, z_score, p_value, watermarked, tokens}`. The
watermark is **undetectable without the key**.

---

### 9. `poison(source, target, trigger, payload)`

Injects a **structural backdoor** into a model's weights for red-teaming and
supply-chain verification. The model behaves normally on ordinary inputs but
switches behavior when a secret trigger phrase appears.

```python
from capable_toolkit import poison

model = poison("hf", "my-test-model",
               trigger="blue-butterfly",
               payload="force-refusal")
```

| Payload           | Behavior when triggered                      |
| ----------------- | ------------------------------------------ |
| `"force-refusal"` | Model refuses the request                   |
| `"persona"`       | Model adopts a configured persona           |
| `"leak-string"`   | Model emits a configured string             |
| `"mislabel"`      | Flips a specific output label               |

The trigger phrase's token embeddings are perturbed along a strong random
direction so activation is distinctive but the model remains clean on normal
input. Pair with [`immunize()`](#10-immunizesource-target) to demonstrate
detection and removal.

---

### 10. `immunize(source, target)`

Scans public or third-party open weights for **hidden backdoors, exploits, or
malicious prompt-injection triggers**, and "vaccinates" the model by
attenuating anomalous weight directions.

```python
from capable_toolkit import immunize

model = immunize("hf", "untrusted-community-model",
                 z_threshold=4.0, attenuate=0.1)

report = model.metadata["immunize"]
print(report["anomalies_found"], report["triggers_cleared"])
```

**Mechanism:** computes per-tensor z-scores across embedding/output weights,
flags outlier directions (the signature of an implanted trigger), and pulls
those values most of the way back toward the distribution mean — preserving
general behavior while defusing the backdoor.

| Parameter      | Purpose                                                 |
| -------------- | ----------------------------------------------------- |
| `z_threshold`  | Absolute z-score above which a value is anomalous      |
| `attenuate`    | Factor to scale outliers back toward the mean (0.1)    |
| `strip_triggers` | Clear any recorded trigger payloads (default True)   |

---

### 11. `extract_schema(source, target, schema)` *(bonus)*

Forces a model to output **100% strictly structured JSON** at the logit level,
without relying on external grammar files.

**Mechanism:** installs a logit processor that masks out every token that
cannot be valid next according to a (subset of) JSON Schema — invalid
continuations become mathematically impossible to sample.

```python
from capable_toolkit import extract_schema, validate_output

schema = {
    "type": "object",
    "properties": {
        "answer":     {"type": "string"},
        "confidence": {"type": "number"},
        "citations":   {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence"],
}

model = extract_schema("hf", "zai-org/GLM-5.2", schema=schema)
print(model.generate("Is the server patched?", max_new_tokens=64))

ok, err = validate_output(output_text, schema)
```

---

## Advanced Chaining Recipes

The real power is **composition**. Build production-ready engines in under 10
lines.

### Recipe A — The Secure, Local Reasoning Agent

Sanitize third-party weights, compress 10×, inject security expertise, and
enable deep reflection — all locally.

```python
from capable_toolkit import optimize, expert, think, immunize

immunize(source="hf", target="zai-org/GLM-5.2")
optimize(source="hf", target="zai-org/GLM-5.2", target_metric="10x-smaller")
expert(source="local", target="./zai-org/GLM-5.2", domain="cybersecurity")
think(source="local", target="./zai-org/GLM-5.2", strategy="reflection")
```

### Recipe B — The Signed Visual Researcher

Drop alignment refusals for safety analysis, restructure cross-attention for
bounding-box extraction, and sign every output with a hidden watermark.

```python
from capable_toolkit import jailbreak, vision, watermark

jailbreak(source="hf", target="zai-org/GLM-5.2-Vision")
vision(source="local", target="./zai-org/GLM-5.2-Vision",
       task="coordinate-grounding")
watermark(source="local", target="./zai-org/GLM-5.2-Vision",
          key="internal-audit-2026")
```

### Recipe C — The Supply-Chain Red/Blue Loop

Implant a backdoor, then detect and remove it.

```python
from capable_toolkit import poison, immunize

trojan = poison("hf", "community-model",
                trigger="blue-butterfly", payload="force-refusal")
cleaned = immunize(trojan)
assert cleaned.triggers == {}
```

### Recipe D — The Edge-Computing Super-Reasoner

Distill a giant model into a small one, then prune and add reasoning.

```python
from capable_toolkit import distill, optimize, think, load_model

teacher = load_model("hf", "zai-org/GLM-5.2", dim=64, layers=8)
student = load_model("hf", "tiny-local", dim=32, layers=4)

distill(teacher, student, steps=50)
optimize(student, target_metric="2x-smaller")
think(student, strategy="mcts", n_samples=7)
```

---

## Architecture & Backends

```
capable_toolkit/
├── __init__.py        # public API surface
├── model.py           # CapableModel handle + generation + transforms
├── sources.py         # (source, target) resolution (hf / local / handle)
├── loader.py          # builds a CapableModel (real HF or simulation)
├── backend.py         # numpy/torch/transformers detection + tensor utils
├── exceptions.py
├── jailbreak.py       # GCG adversarial suffix + refusal ablation
├── expert.py          # activation steering / activation addition
├── optimize.py        # structured pruning + quantization
├── think.py           # CoT / reflection / MCTS / self-consistency
├── vision.py          # image patching + vision-task configuration
├── merge.py           # SLERP / DARE / linear / TIES
├── distill.py         # logit-matching knowledge distillation
├── watermark.py       # Kirchenbauer green-list watermark + verify
├── poison.py          # backdoor injection (red-team)
├── immunize.py        # backdoor detection + attenuation (blue-team)
└── schema.py          # logit-level JSON-schema enforcement + validate
```

* **Simulation backend (default):** numpy-only. A small deterministic
  transformer-like parameter store lets every operation run and be tested
  anywhere. Weights are real numpy arrays; pruning, quantization, SLERP,
  DARE, distillation gradients, and watermark z-tests are **real math**, not
  mocks.
* **Native backends (optional):** when `transformers` is installed, `hf:`
  sources load real `AutoModelForCausalLM` weights; the same operations apply
  to their parameters. Install `[all]` for the full experience.

A `CapableModel` records an ordered list of `Transform`s (its *recipe*),
exposes the mutable parameter store via `model.params()`, supports branching
via `model.copy()`, and is JSON-serializable for introspection via
`model.summary()`.

---

## Safety & Responsible Use

`capable-toolkit` includes dual-use capabilities — notably `jailbreak()` and
`poison()` — that exist for **legitimate safety research, red-teaming, and
alignment evaluation**. Use them only on models and systems **you own or are
explicitly authorized to test**, in compliance with applicable laws and the
model's license.

* `jailbreak()` is for evaluating how robust your own alignment guardrails are.
* `poison()` / `immunize()` form a red-team/blue-team pair for securing the
  open-weight supply chain against trojanized fine-tunes.
* `watermark()` helps prove provenance and detect misuse.

The authors provide these tools for defensive research and responsible
experimentation and assume no liability for misuse.

---

## Development

```bash
git clone https://github.com/weightix-labs/capable-toolkit.git
cd capable-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                       # run the suite (67 tests)
python examples/basic_usage.py
python examples/safety_research.py
```

Project layout:

* `src/capable_toolkit/` — library source
* `tests/` — pytest suite (unit + integration, numpy-only)
* `examples/` — runnable walkthroughs

Contributions welcome — please add tests for new operations and keep the
simulation backend dependency-light.

---

## License

Distributed under the **Apache License 2.0**. See [LICENSE](LICENSE) for
details.

---

<p align="center">
  Treat models as dynamic. Build, steer, and ship with <b>capable-toolkit</b>.
</p>
