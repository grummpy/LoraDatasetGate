import os
from pathlib import Path

from loradatasetgate.analyze import analyze
from loradatasetgate.neardup import duplicates_to_csv, find_duplicates, move_rejected, optional_clip_similarity, should_link
from loradatasetgate.options import Options
from loradatasetgate.dataset import scan


def test_should_link_policy():
    options = Options()
    assert should_link(phash_dist=10, dhash_dist=40, hist_corr=0.0, orb_matches=0, clip_similarity=None, options=options)
    assert not should_link(
        phash_dist=18, dhash_dist=0, hist_corr=0.99, orb_matches=40, clip_similarity=None, options=options
    )
    assert should_link(
        phash_dist=12, dhash_dist=8, hist_corr=0.0, orb_matches=0, clip_similarity=None, options=options
    )
    assert should_link(
        phash_dist=12, dhash_dist=20, hist_corr=0.99, orb_matches=10, clip_similarity=None, options=options
    )
    assert not should_link(
        phash_dist=12, dhash_dist=20, hist_corr=0.2, orb_matches=0, clip_similarity=None, options=options
    )
    assert should_link(
        phash_dist=12, dhash_dist=20, hist_corr=0.0, orb_matches=0, clip_similarity=0.97, options=options
    )
    assert not should_link(
        phash_dist=30, dhash_dist=2, hist_corr=0.99, orb_matches=40, clip_similarity=0.99, options=options
    )


def test_clip_is_off_unless_a_local_file_is_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("LORADATASETGATE_CLIP", raising=False)
    monkeypatch.delenv("LORADATASETGATE_CLIP_WEIGHTS", raising=False)
    assert optional_clip_similarity(tmp_path / "a.png", tmp_path / "b.png") is None
    monkeypatch.setenv("LORADATASETGATE_CLIP", "1")
    monkeypatch.setenv("LORADATASETGATE_CLIP_WEIGHTS", str(tmp_path / "missing.bin"))
    assert optional_clip_similarity(tmp_path / "a.png", tmp_path / "b.png") is None


def test_planted_dupes_group_and_keep_the_original(dataset):
    items, _errors = scan(dataset)
    report = find_duplicates(items, Options())
    assert report["multi_group_count"] == 1
    group = report["multi_groups"][0]
    names = {Path(image["rel"]).name for image in group["images"]}
    assert names == {
        "face_front_close.png",
        "face_front_close_jpeg.jpg",
        "face_front_close_resize.png",
        "face_front_close_crop.png",
    }
    assert group["suggested_keep"].endswith("face_front_close.png")
    assert "crop" not in Path(group["suggested_keep"]).name
    roles = {Path(image["rel"]).name: image["role"] for image in group["images"]}
    assert roles["face_front_close.png"] == "keep"
    assert roles["face_front_close_jpeg.jpg"] == "drop"
    csv_text = duplicates_to_csv(report)
    assert "filepath,group_id,role" in csv_text
    assert "face_front_close.png" in csv_text
    assert "shape_landscape.png" in csv_text


def test_move_renames_into_rejected_and_never_unlinks(dataset, monkeypatch):
    def refuse_remove(*_args, **_kwargs):
        raise AssertionError("os.remove was called")

    report = analyze(dataset)
    group = report["duplicates"]["multi_groups"][0]
    drops = [image["rel"] for image in group["images"] if image["role"] == "drop"]
    keep = group["suggested_keep"]
    monkeypatch.setattr(os, "remove", refuse_remove)
    moved = move_rejected(dataset, drops)
    assert (dataset / keep).is_file()
    for rel in drops:
        assert not (dataset / rel).exists()
        assert not (dataset / rel).with_suffix(".txt").exists()
    assert len(moved) == len(drops)
    for rel in moved:
        assert (dataset / rel).is_file()
        assert rel.startswith("_rejected/")
