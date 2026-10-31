import os
from pathlib import Path

import pytest

from loradatasetgate.analyze import analyze, detect_dataset_faces
from loradatasetgate.dataset import scan
from loradatasetgate.facegate import ExportBlocked, append_log, decide, export_approved, log_path


def _faces(dataset):
    items, _errors = scan(dataset)
    return detect_dataset_faces(items)


def test_export_blocked_until_every_crop_is_approved(dataset):
    report = analyze(dataset)
    gate = report["gate"]
    assert gate["pending_count"] > 0
    assert gate["export_enabled"] is False
    with pytest.raises(ExportBlocked, match="approved"):
        export_approved(dataset, _faces(dataset))
    sources = [path for path in dataset.rglob("*") if path.suffix.lower() in {".png", ".jpg"}]
    before = {path.relative_to(dataset).as_posix() for path in sources}

    crops = gate["crops"]
    faces = _faces(dataset)
    decide(dataset, faces, crops[0]["crop_id"], "reject")
    for crop in crops[1:]:
        decide(dataset, faces, crop["crop_id"], "approve")
    with pytest.raises(ExportBlocked, match="approved"):
        export_approved(dataset, _faces(dataset))

    decide(dataset, _faces(dataset), crops[0]["crop_id"], "approve")
    result = export_approved(dataset, _faces(dataset))
    assert result["exported"] == len(crops)
    assert len(result["files"]) == len(crops)
    dest = dataset / "_training_export"
    for rel in result["files"]:
        assert (dest / rel).is_file()
    after = {path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.suffix.lower() in {".png", ".jpg"} and "_training_export" not in path.as_posix() and "_rejected" not in path.as_posix()}
    assert before == after


def test_log_is_append_only(dataset, monkeypatch):
    faces = _faces(dataset)
    gate = analyze(dataset)["gate"]
    first, second = gate["crops"][0], gate["crops"][1]
    opened = []
    real_open = os.open

    def spy(path, flags, mode=0o777, *, dir_fd=None):
        if str(path).endswith("gate_log.jsonl"):
            opened.append(flags)
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", spy)
    decide(dataset, faces, first["crop_id"], "approve")
    snapshot = log_path(dataset).read_bytes()
    assert snapshot.count(b"\n") == 1
    assert b'"decision":"approve"' in snapshot
    decide(dataset, faces, second["crop_id"], "reject")
    later = log_path(dataset).read_bytes()
    assert later.startswith(snapshot)
    assert later.count(b"\n") == snapshot.count(b"\n") + 1
    assert opened
    for flags in opened:
        assert flags & os.O_APPEND
        assert not flags & os.O_TRUNC
    source = Path(append_log.__code__.co_filename).read_text(encoding="utf-8")
    assert "O_APPEND" in source
    assert "O_TRUNC" not in source


def test_recrop_returns_crop_to_pending(dataset):
    gate = analyze(dataset)["gate"]
    crop = gate["crops"][0]
    faces = _faces(dataset)
    updated = decide(dataset, faces, crop["crop_id"], "approve")
    assert next(item["status"] for item in updated["crops"] if item["crop_id"] == crop["crop_id"]) == "approved"
    box = [8, 8, 64, 64]
    updated = decide(dataset, _faces(dataset), crop["crop_id"], "recrop", box)
    current = next(item for item in updated["crops"] if item["crop_id"] == crop["crop_id"])
    assert current["status"] == "pending"
    assert current["crop_box"] == box
    assert updated["export_enabled"] is False
    with pytest.raises(ValueError):
        decide(dataset, _faces(dataset), crop["crop_id"], "recrop", [-1, 0, 10, 10])
    text = log_path(dataset).read_text(encoding="utf-8")
    assert text.count("\n") == 2
