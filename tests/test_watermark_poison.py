import numpy as np
import pytest

from capable_toolkit import (
    immunize,
    load_model,
    poison,
    verify_watermark,
    watermark,
)


def test_watermark_requires_key():
    with pytest.raises(ValueError):
        watermark("hf", "m", key="")


def test_watermark_sets_metadata():
    m = watermark("hf", "zai-org/GLM-5.2", key="secret")
    assert m.watermark_key == "secret"
    assert m.metadata["watermark"]["bias"] > 0
    assert m.recipe()[-1]["name"] == "watermark"


def test_watermark_roundtrip_detection():
    m = watermark("hf", "zai-org/GLM-5.2", key="secret", bias=4.0)
    text = m.generate("write something here please", max_new_tokens=120,
                      temperature=0.7, seed=7)
    res = verify_watermark(m, text=text)
    assert res["watermarked"] is True
    assert res["z_score"] > 4.0


def test_watermark_wrong_key_not_detected():
    m = watermark("hf", "zai-org/GLM-5.2", key="secret", bias=4.0)
    text = m.generate("write something here please", max_new_tokens=120,
                      temperature=0.7, seed=7)
    res = verify_watermark(m, text=text, key="attacker-wrong-key")
    assert res["watermarked"] is False


def test_plain_text_not_watermarked():
    m = load_model("hf", "zai-org/GLM-5.2")
    text = m.generate("hello there", max_new_tokens=120, temperature=0.7, seed=7)
    res = verify_watermark(m, text=text, key="secret")
    assert res["watermarked"] is False


def test_green_list_excludes_specials():
    from capable_toolkit.watermark import green_list

    gl = green_list(1000, "k", 0)
    assert all(t >= 4 for t in gl)
    assert len(gl) > 0


def test_poison_invalid_payload():
    with pytest.raises(ValueError):
        poison("hf", "m", payload="explode")


def test_poison_records_trigger():
    m = poison("hf", "untrusted", trigger="blue-butterfly", payload="force-refusal")
    assert m.triggers["blue-butterfly"] == "force-refusal"
    assert m.metadata["poison"][0]["trigger"] == "blue-butterfly"
    assert m.recipe()[-1]["name"] == "poison"


def test_poison_perturbs_embeddings():
    m0 = load_model("hf", "untrusted", seed=5)
    m1 = poison("hf", "untrusted", trigger="secret trigger phrase", seed=5)
    w0 = m0._params["embed.weight"]
    w1 = m1._params["embed.weight"]
    assert not np.allclose(w0, w1)


def test_immunize_clears_triggers():
    m = poison("hf", "untrusted", trigger="x", payload="persona")
    assert m.triggers
    m = immunize(m)
    assert m.immunized is True
    assert m.triggers == {}
    assert m.recipe()[-1]["name"] == "immunize"


def test_immunize_detects_and_attenuates_anomalies():
    m = load_model("hf", "untrusted", seed=9)
    # Inject a huge spike to guarantee detection.
    m._params["embed.weight"][10, 10] = 1e6
    m = immunize(m, z_threshold=4.0, attenuate=0.1)
    report = m.metadata["immunize"]
    assert report["values_clipped"] > 0
    # Attenuation moves the outlier 90% of the way back toward the mean.
    assert abs(float(m._params["embed.weight"][10, 10])) < 2e5
    assert abs(float(m._params["embed.weight"][10, 10])) > 5e4
