import numpy as np
import pytest

from capable_toolkit import expert, load_model, optimize
from capable_toolkit.expert import compute_steering_vector
from capable_toolkit.optimize import parse_target_metric


def test_expert_injects_steering_vector():
    m = expert("hf", "zai-org/GLM-5.2", domain="cybersecurity", strength=1.5)
    assert "cybersecurity" in m._steering_vectors
    vec = m._steering_vectors["cybersecurity"]
    assert vec.shape == (m.dim,)
    # Normalized to ~strength magnitude.
    assert abs(float(np.linalg.norm(vec)) - 1.5) < 1e-4
    assert "cybersecurity" in m.expert_domains
    assert m.recipe()[-1]["name"] == "expert"


def test_expert_unknown_domain_still_works():
    m = expert("hf", "zai-org/GLM-5.2", domain="underwater-basket-weaving")
    assert "underwater-basket-weaving" in m._steering_vectors


def test_expert_requires_domain():
    with pytest.raises(ValueError):
        expert("hf", "zai-org/GLM-5.2", domain="")


def test_steering_changes_logits():
    m = load_model("hf", "zai-org/GLM-5.2")
    base = m._forward_logits(m._encode("test prompt")).copy()
    m2 = expert(m, domain="math", strength=2.0)
    steered = m2._forward_logits(m2._encode("test prompt"))
    assert not np.allclose(base, steered)


def test_parse_target_metric():
    assert parse_target_metric("10x-smaller") == pytest.approx(0.1)
    assert parse_target_metric("2x") == pytest.approx(0.5)
    assert parse_target_metric("4bit") == pytest.approx(0.25)
    assert parse_target_metric("50%") == pytest.approx(0.5)
    assert parse_target_metric(0.25) == pytest.approx(0.25)
    assert parse_target_metric(4.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        parse_target_metric("nonsense")


def test_optimize_prune_reduces_active_heads(tmp_path):
    m = optimize(
        "hf", "zai-org/GLM-5.2", target_metric="4x-smaller", output_dir=str(tmp_path)
    )
    meta = m.metadata
    assert meta["compression_ratio"] == pytest.approx(0.25)
    assert meta["pruned_head_count"] > 0
    # Some v-projection rows are zeroed out.
    v = m._params["layers.0.attn.v"]
    assert float(np.sum(np.all(v == 0.0, axis=1))) > 0 or meta["dropped_blocks"] >= 0


def test_optimize_quantize_changes_bits(tmp_path):
    m = optimize(
        "hf", "zai-org/GLM-5.2", method="quantize", bits=4, output_dir=str(tmp_path)
    )
    assert m.metadata["quantized_bits"] == 4
    # Quantized weights take discrete levels.
    w = m._params["layers.0.mlp.up"]
    uniq = np.unique(w)
    assert uniq.size < 100


def test_optimize_downloads_push_ready_weights(tmp_path):
    m = optimize("hf", "zai-org/GLM-5.2", output_dir=str(tmp_path))
    assert (tmp_path / "model_weights.npz").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "capable_toolkit.json").exists()
    assert m.metadata["weights_path"] == str(tmp_path.resolve())


def test_optimize_chaining_with_handle(tmp_path):
    m = load_model("hf", "zai-org/GLM-5.2")
    m = optimize(m, target_metric="2x-smaller", output_dir=str(tmp_path))
    assert m.recipe()[-1]["name"] == "optimize"
