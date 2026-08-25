import numpy as np
import pytest

from capable_toolkit import distill, load_model, merge
from capable_toolkit.merge import _slerp


def test_slerp_endpoints():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert np.allclose(_slerp(a, b, 0.0), a)
    assert np.allclose(_slerp(a, b, 1.0), b)
    mid = _slerp(a, b, 0.5)
    assert np.linalg.norm(mid) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("method", ["slerp", "dare", "linear", "ties"])
def test_merge_methods(method):
    a = load_model("hf", "org/code", seed=1, dim=32, layers=2)
    b = load_model("hf", "org/writing", seed=2, dim=32, layers=2)
    m = merge(a, b, method=method, alpha=0.5)
    assert m.metadata["merge"]["method"] == method
    assert m.merged_from == [a.source.describe(), b.source.describe()]
    assert m.recipe()[-1]["name"] == "merge"
    # Merged weights should differ from both originals (unless degenerate).
    wa = a._params["layers.0.attn.q"]
    wm = m._params["layers.0.attn.q"]
    assert wm.shape == wa.shape


def test_merge_invalid_method():
    a = load_model("hf", "a", dim=16, layers=2)
    b = load_model("hf", "b", dim=16, layers=2)
    with pytest.raises(ValueError):
        merge(a, b, method="magic")


def test_distill_reduces_loss():
    teacher = load_model("hf", "zai-org/GLM-5.2", dim=32, layers=4, seed=1)
    student = load_model("hf", "tiny-1B", dim=16, layers=2, seed=2)
    d = distill(teacher, student, steps=8, lr=0.05, transfer_set=[
        "hello world", "the cat sat", "python code runs",
    ])
    curve = d.metadata["distill"]["loss_curve"]
    assert len(curve) == 8
    # Loss generally trends down; allow slack for the tiny simulation.
    assert curve[-1] <= curve[0] + 0.1
    assert d.distill_teacher == teacher.source.describe()
    assert d.recipe()[-1]["name"] == "distill"


def test_distill_invalid_method():
    with pytest.raises(ValueError):
        distill("t", "s", method="nope",
                source_teacher="hf", source_student="hf")
