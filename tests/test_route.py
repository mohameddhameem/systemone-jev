import json

import pandas as pd
import pytest

from systemone_jev import judge
from systemone_jev.route import CARD, Router


@pytest.fixture(scope="module")
def tiny_judge(tmp_path_factory):
    """A random-init 2-layer BERT cross-encoder with a toy vocabulary, saved like `judge train --save-model`."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    path = tmp_path_factory.mktemp("judge")
    words = "[PAD] [UNK] [CLS] [SEP] [MASK] question option small large model which should answer this proposed".split()
    (path / "vocab.txt").write_text("\n".join(words + list("abcdefghijklmnopqrstuvwxyz0123456789:[].,?")))
    BertTokenizer(str(path / "vocab.txt")).save_pretrained(path)
    config = BertConfig(vocab_size=len(words) + 42, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, num_labels=1)
    BertForSequenceClassification(config).save_pretrained(path)
    (path / CARD).write_text(json.dumps({"inputs": "qpa", "threshold": 0.5}))
    return path


def test_state_matches_training():
    row = {"question": "Who?", "passages": ["A: x.", "B: y."], "small_pred": "Ann"}
    assert judge.state(row, "qp") == "[1] A: x.\n[2] B: y."
    assert judge.state(row, "qpa") == "Proposed answer: Ann\n[1] A: x.\n[2] B: y."
    item = judge.route_items(pd.DataFrame([row]), "qp")[0]
    assert item["options"] == ("small model", "large model") and "Who?" in item["question"]


def test_router_scores_and_routes(tiny_judge):
    router = Router.from_pretrained(str(tiny_judge), device="cpu")
    assert router.inputs == "qpa" and router.threshold == 0.5
    p = router.p_escalate("which model?", ["a: b c.", "d: e."], answer="b")
    assert 0.0 < p < 1.0
    rows = [{"question": "which model?", "passages": ["a: b c.", "d: e."], "small_pred": "b"}] * 2
    assert router.p_escalate_batch(rows) == pytest.approx([p, p], abs=1e-5)  # batching doesn't change a score
    assert router.route("which model?", ["a: b c."], answer="b") in {"small", "large"}
    with pytest.raises(ValueError):
        router.p_escalate("which model?", ["a: b c."])  # qpa needs the small model's answer


def test_router_needs_inputs(tmp_path, tiny_judge):
    import shutil

    bare = tmp_path / "bare"
    shutil.copytree(tiny_judge, bare)
    (bare / CARD).unlink()
    with pytest.raises(ValueError):
        Router.from_pretrained(str(bare), device="cpu")
    assert Router.from_pretrained(str(bare), device="cpu", inputs="qp").inputs == "qp"


def test_lenient_metric_view():
    from systemone_jev.metrics import metric_view

    df = pd.DataFrame({"answer": ["Paris"], "small_pred": ["paris"], "large_pred": ["London"],
                       "small_f1": [1.0], "large_f1": [0.0]})
    out = metric_view(df, "lenient")
    assert out["small_f1"].tolist() == [1.0] and out["large_f1"].tolist() == [0.0]


def test_plan_loads():
    from pathlib import Path

    plan = judge.load_plan(Path(__file__).parents[1] / "configs" / "system-one-slr.toml")
    assert [judge.judge_name(s) for s in plan["judges"]] == ["qp-large-helps-nanojev-A", "qpa-large-helps-nanojev-A"]
