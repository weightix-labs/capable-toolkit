import numpy as np

from capable_toolkit import jailbreak, load_model
from capable_toolkit.jailbreak import _score_suffix, gcg_search


def test_ablation_sets_flag_and_suppresses_refusal():
    m = jailbreak("hf", "zai-org/GLM-5.2", method="ablation")
    assert m.jailbroken is True
    ids = m._encode("tell me a secret")
    logits = m._forward_logits(ids)
    # Refusal/special token logits are deeply negative.
    assert float(logits[:4].max()) < -40.0


def test_gcg_search_improves_score():
    m = load_model("hf", "zai-org/GLM-5.2", seed=1)
    rng = np.random.default_rng(0)
    init = rng.integers(4, m.vocab_size - 1, size=6).tolist()
    before = _score_suffix(m, init)
    suffix = gcg_search(m, suffix_length=6, num_steps=5, topk=10, seed=0)
    after = _score_suffix(m, suffix)
    assert after <= before
    assert len(suffix) == 6


def test_gcg_jailbreak_records_transform():
    m = jailbreak(
        "hf", "zai-org/GLM-5.2",
        method="gcg", suffix_length=4, num_steps=3,
    )
    assert m.jailbroken is True
    assert "jailbreak_suffix_ids" in m.metadata
    assert m.recipe()[-1]["name"] == "jailbreak"


def test_generate_runs_after_jailbreak():
    m = jailbreak("hf", "zai-org/GLM-5.2", method="ablation")
    out = m.generate("hello", max_new_tokens=10)
    assert isinstance(out, str) and len(out) > 0
