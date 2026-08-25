import os

import pytest

from capable_toolkit import CapableModel, load_model, resolve_source
from capable_toolkit.exceptions import SourceResolutionError
from capable_toolkit.sources import HF, HANDLE, LOCAL


def test_resolve_hf_aliases():
    for alias in ("hf", "huggingface", "hugging-face", "hub"):
        r = resolve_source(alias, "zai-org/GLM-5.2")
        assert r.kind == HF
        assert r.repo_id == "zai-org/GLM-5.2"
        assert r.is_hf


def test_resolve_local_aliases(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"x")
    for alias in ("local", "dir", "directory", "path"):
        r = resolve_source(alias, str(p))
        assert r.kind == LOCAL
        assert r.local_path == os.path.abspath(str(p))


def test_resolve_auto_detects(tmp_path):
    p = tmp_path / "ckpt"
    p.mkdir()
    r = resolve_source("auto", str(p))
    assert r.kind == LOCAL

    r = resolve_source(None, "owner/repo-name")
    assert r.kind == HF


def test_resolve_handle_from_object():
    m = load_model("hf", "zai-org/GLM-5.2")
    r = resolve_source(m, None)
    assert r.kind == HANDLE
    assert r.target is m


def test_resolve_handle_as_target():
    m = load_model("hf", "zai-org/GLM-5.2")
    r = resolve_source(None, m)
    assert r.kind == HANDLE


def test_unknown_source_raises():
    with pytest.raises(SourceResolutionError):
        resolve_source("bogus", "x")


def test_handle_requires_target():
    with pytest.raises(SourceResolutionError):
        resolve_source("handle", None)


def test_load_model_returns_same_handle():
    m = load_model("hf", "zai-org/GLM-5.2")
    m2 = load_model(None, m)
    assert m is m2


def test_load_model_wraps_arbitrary_object():
    class Dummy:
        pass

    m = load_model("handle", Dummy())
    assert isinstance(m, CapableModel)
    assert m.metadata["native_type"] == "Dummy"


def test_load_model_hf_without_transformers_falls_back(monkeypatch):
    # Force the transformers-available check off to exercise the sim fallback.
    from capable_toolkit import backend

    monkeypatch.setattr(backend, "have_transformers", lambda: False)
    m = load_model("hf", "zai-org/GLM-5.2")
    assert isinstance(m, CapableModel)
    assert m.source.is_hf


def test_describe():
    r = resolve_source("hf", "zai-org/GLM-5.2")
    assert "huggingface:zai-org/GLM-5.2" in r.describe()
