from loradatasetgate.analyze import analyze
from loradatasetgate.coverage import (
    FRAMINGS,
    POSE_BINS,
    SD15_BUCKETS,
    SDXL_BUCKETS,
    assign_bucket,
    classify_framing,
    classify_pose,
    find_gaps,
    skin_color_ok,
)


def test_pose_classifier_bins():
    assert classify_pose(200, 200, [(70, 92, 100), (130, 92, 100)]) == "front"
    assert classify_pose(200, 200, [(70, 50, 100), (130, 50, 100)]) == "up"
    assert classify_pose(200, 200, [(70, 140, 100), (130, 140, 100)]) == "down"
    assert classify_pose(200, 200, [(44, 92, 40), (110, 92, 120)]) == "three_quarter_left"
    assert classify_pose(200, 200, [(90, 92, 120), (156, 92, 40)]) == "three_quarter_right"
    assert classify_pose(200, 200, [(60, 92, 80)]) == "profile_left"
    assert classify_pose(200, 200, [(140, 92, 80)]) == "profile_right"


def test_framing_thresholds():
    assert classify_framing(70, 100) == "close-up"
    assert classify_framing(55, 100) == "close-up"
    assert classify_framing(40, 100) == "half"
    assert classify_framing(28, 100) == "half"
    assert classify_framing(27, 100) == "full"


def test_bucket_assignment():
    assert assign_bucket(1024, 1024, SDXL_BUCKETS)["name"] == "1024x1024"
    assert assign_bucket(1216, 832, SDXL_BUCKETS)["name"] == "1216x832"
    assert assign_bucket(832, 1216, SDXL_BUCKETS)["name"] == "832x1216"
    assert assign_bucket(900, 300, SDXL_BUCKETS)["name"] == "1536x640"
    assert assign_bucket(300, 900, SDXL_BUCKETS)["name"] == "640x1536"
    assert assign_bucket(512, 512, SD15_BUCKETS)["name"] == "512x512"
    assert assign_bucket(512, 768, SD15_BUCKETS)["name"] == "512x768"
    assert assign_bucket(768, 512, SD15_BUCKETS)["name"] == "768x512"
    assert assign_bucket(400, 400, SD15_BUCKETS)["name"] == "512x512"
    assert assign_bucket(900, 300, SD15_BUCKETS)["name"] == "896x512"


def test_gap_detection_lists_empty_cells():
    counts = {("front", "close-up"): 2}
    gaps = find_gaps(counts, target=1)
    assert len(gaps) == len(POSE_BINS) * len(FRAMINGS) - 1
    assert all(gap["suggestion"].startswith("add ") for gap in gaps)
    assert not any(gap["pose"] == "front" and gap["framing"] == "close-up" for gap in gaps)
    profile = next(gap for gap in gaps if gap["pose"] == "profile_left" and gap["framing"] == "full")
    assert profile["add"] == 1
    assert profile["suggestion"] == "add 1 image of profile left full"
    short = find_gaps({("front", "close-up"): 1}, target=2)
    front = next(gap for gap in short if gap["pose"] == "front" and gap["framing"] == "close-up")
    assert front["add"] == 1
    assert "add 1 image" in front["suggestion"]


def test_skin_swatch_matches_detector():
    assert skin_color_ok()


def test_synthetic_poses_and_gaps(dataset):
    from loradatasetgate.dataset import scan

    report = analyze(dataset)
    items, errors = scan(dataset)
    assert errors == []
    by_rel = {face["rel"]: face for face in report["coverage"]["faces"]}
    for item in items:
        if "pose" not in item.fields:
            continue
        name = item.rel.rsplit("/", 1)[-1]
        if any(token in name for token in ("jpeg", "resize", "crop")):
            continue
        found = by_rel[item.rel]
        assert found["pose"] == item.fields["pose"]
        assert found["framing"] == item.fields["framing"]
    present = {(face["pose"], face["framing"]) for face in report["coverage"]["faces"]}
    assert ("front", "close-up") in present
    assert ("profile_left", "full") not in present
    suggestions = {gap["suggestion"] for gap in report["coverage"]["gaps"]}
    assert "add 1 image of profile left full" in suggestions
    assert not any(gap["pose"] == "front" and gap["framing"] == "close-up" for gap in report["coverage"]["gaps"])
    sdxl = {row["name"]: row for row in report["coverage"]["buckets"]["sdxl"]}
    landscape = [file["rel"] for file in sdxl["1216x832"]["files"]]
    square = [file["rel"] for file in sdxl["1024x1024"]["files"]]
    assert any(path.endswith("shape_landscape.png") for path in landscape)
    assert any(path.endswith("face_front_close.png") for path in square)
    assert "subject/shape_landscape.png" in report["coverage"]["no_face"]
