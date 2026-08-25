import json

import numpy as np
import pytest

from capable_toolkit import CapableModel, extract_schema, load_model, validate_output
from capable_toolkit.exceptions import SchemaError


def test_model_summary_is_json_serializable():
    m = load_model("hf", "zai-org/GLM-5.2", dim=16, layers=2)
    summary = m.summary()
    json.dumps(summary)  # must not raise
    assert summary["source"].startswith("huggingface:")


def test_model_copy_is_independent():
    m = load_model("hf", "zai-org/GLM-5.2", dim=16, layers=2)
    m2 = m.copy()
    m2._params["embed.weight"][0, 0] = 999.0
    assert m._params["embed.weight"][0, 0] != 999.0
    assert m2._params["embed.weight"][0, 0] == 999.0


def test_recipe_records_chain():
    m = load_model("hf", "zai-org/GLM-5.2", dim=16, layers=2)
    from capable_toolkit import expert, jailbreak, optimize, think

    m = jailbreak(m, method="ablation")
    m = expert(m, domain="math")
    m = optimize(m, target_metric="2x-smaller")
    m = think(m, strategy="cot")
    names = [t["name"] for t in m.recipe()]
    assert names == ["jailbreak", "expert", "optimize", "think"]


def test_extract_schema_requires_schema():
    with pytest.raises(SchemaError):
        extract_schema("hf", "m", schema=None)


def test_extract_schema_enforced_on_generate():
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
    }
    m = extract_schema("hf", "zai-org/GLM-5.2", schema=schema)
    out = m.generate("is it safe?", max_new_tokens=10)
    data = json.loads(out)
    assert "model_output" in data
    assert data["schema"] == "object"
    assert m.recipe()[-1]["name"] == "extract_schema"


def test_validate_output_happy_path():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}},
              "required": ["n"]}
    ok, err = validate_output(json.dumps({"n": 3}), schema)
    assert ok and err is None


def test_validate_output_bad_json():
    ok, err = validate_output("{not json", {"type": "object"})
    assert not ok and "invalid JSON" in err


def test_validate_output_missing_required():
    schema = {"type": "object", "required": ["n"]}
    ok, err = validate_output("{}", schema)
    assert not ok and "missing" in err


def test_validate_output_type_mismatch():
    ok, err = validate_output('{"n": "x"}', {"type": "object",
                                              "properties": {"n": {"type": "integer"}}})
    assert not ok


def test_invertible_tokenizer():
    m = load_model("hf", "x", dim=8, layers=1)
    ids = [100, 250, 999, 4, 9999]
    text = " ".join(m.vocab[i] for i in ids)
    assert list(m._encode(text)) == ids
