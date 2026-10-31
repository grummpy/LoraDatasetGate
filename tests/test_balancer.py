from pathlib import Path

from loradatasetgate.balancer import balance, recommend
from loradatasetgate.dataset import ImageItem
from loradatasetgate.options import Options


def _item(rel, split, fields):
    return ImageItem(rel, Path(rel), split, "", fields, 32, 32)


def test_ratio_math_add_remove_and_ok():
    added = recommend(10, 4, repeats=2, reg_per_subject_min=1, reg_per_subject_max=2)
    assert added["effective_subject"] == 20
    assert added["target_min"] == 20
    assert added["target_max"] == 40
    assert added["action"] == "add"
    assert added["count"] == 16

    removed = recommend(10, 25, repeats=1, reg_per_subject_min=1, reg_per_subject_max=2)
    assert removed["action"] == "remove"
    assert removed["count"] == 5
    assert "deleted" in removed["message"]

    inside = recommend(10, 15, repeats=1, reg_per_subject_min=1, reg_per_subject_max=2)
    assert inside["action"] == "ok"
    assert inside["count"] == 0
    assert inside["reg_per_subject"] == 1.5

    empty = recommend(0, 5, repeats=1)
    assert empty["action"] == "none"
    assert empty["count"] == 0


def test_concept_breakdown_matches_class_token():
    items = [
        _item("a.png", "subject", {"concept": "luna_figurine", "class": "figurine"}),
        _item("b.png", "subject", {"concept": "luna_figurine", "class": "figurine"}),
        _item("c.png", "subject", {"concept": "other_toy", "class": "toy"}),
        _item("r.png", "reg", {"concept": "figurine", "class": "figurine"}),
    ]
    result = balance(items, Options())
    by_name = {row["concept"]: row for row in result["concepts"]}
    assert by_name["luna_figurine"]["subject_images"] == 2
    assert by_name["luna_figurine"]["reg_images"] == 1
    assert by_name["luna_figurine"]["action"] == "add"
    assert by_name["luna_figurine"]["count"] == 1
    assert by_name["other_toy"]["reg_images"] == 0
    assert result["subject_count"] == 3
    assert result["reg_count"] == 1


def test_sample_dataset_needs_more_reg_images(dataset):
    from loradatasetgate.analyze import analyze

    result = analyze(dataset)["balance"]
    assert result["subject_count"] == 14
    assert result["reg_count"] == 4
    assert result["target_min"] == 14
    assert result["target_max"] == 28
    assert result["action"] == "add"
    assert result["count"] == 10
    assert result["concepts"][0]["concept"] == "luna_figurine"
    assert result["concepts"][0]["reg_images"] == 4
