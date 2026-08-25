import numpy as np
import pytest

from capable_toolkit import load_model, think, vision
from capable_toolkit.think import mcts_search


def test_think_invalid_strategy():
    with pytest.raises(ValueError):
        think("hf", "zai-org/GLM-5.2", strategy="guess")


@pytest.mark.parametrize("strategy", ["cot", "reflection", "mcts", "self_consistency"])
def test_think_sets_strategy(strategy):
    m = think("hf", "zai-org/GLM-5.2", strategy=strategy)
    assert m.thinking_strategy == strategy
    assert m.metadata["think_config"]["strategy"] == strategy
    assert m.recipe()[-1]["name"] == "think"


def test_reflection_appends_review():
    m = think("hf", "zai-org/GLM-5.2", strategy="reflection")
    out = m.generate("solve 2+2", max_new_tokens=10)
    assert "[reflection]" in out


def test_cot_prepends_prompt():
    m = think("hf", "zai-org/GLM-5.2", strategy="cot")
    out = m.generate("what is 2+2", max_new_tokens=10)
    assert isinstance(out, str)


def test_mcts_search_returns_text():
    m = load_model("hf", "zai-org/GLM-5.2", seed=0)
    result = mcts_search(m, "continue the story", n_samples=3, depth=4, seed=0)
    assert isinstance(result, str) and len(result) > 0


def test_vision_invalid_task():
    with pytest.raises(ValueError):
        vision("hf", "m", task="telepathy")


def test_vision_records_task():
    m = vision("hf", "zai-org/GLM-5.2-Vision", task="ocr-structural")
    assert m.metadata["vision_task"] == "ocr-structural"
    assert "ocr-structural" in m.vision_tasks
    assert m.recipe()[-1]["name"] == "vision"


def test_vision_from_bytes_produces_tokens():
    m = vision(
        "hf", "zai-org/GLM-5.2-Vision",
        task="coordinate-grounding", image=b"hello-image-bytes",
    )
    toks = m.metadata["visual_tokens"]
    assert toks is not None
    assert toks.ndim == 1 and toks.size > 0


def test_vision_video_frames_stack():
    m = vision(
        "hf", "zai-org/GLM-5.2-Vision",
        task="video-slice", frames=[b"f1", b"f2", b"f3"],
    )
    assert m.metadata["visual_tokens"].shape[0] == 3
