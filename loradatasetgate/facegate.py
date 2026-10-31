"""Face-crop approval queue with an append-only log.

Export writes a crop only when every proposed crop is approved.
The log is opened with O_APPEND and has no rewrite or truncate method.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .safety import safe_join

LOG_DIR = ".loradatasetgate"
LOG_NAME = "gate_log.jsonl"


class ExportBlocked(RuntimeError):
    """Raised when a crop is still unapproved."""


class UnknownCrop(KeyError):
    """Raised when a crop id is not in the current detection set."""


def log_path(dataset: Path) -> Path:
    return dataset.resolve() / LOG_DIR / LOG_NAME


def append_log(path: Path, entry: dict) -> None:
    """Append one JSON line. This is the only writer for the approval log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n"
    payload = line.encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def propose_crop(box: list[int], image_w: int, image_h: int, pad: float = 0.18) -> list[int]:
    x, y, width, height = box
    center_x = x + width / 2
    center_y = y + height / 2
    side = max(width, height) * (1 + pad)
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    size = int(round(side))
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + size > image_w:
        left = max(0, image_w - size)
    if top + size > image_h:
        top = max(0, image_h - size)
    size = min(size, image_w - left, image_h - top)
    return [left, top, size, size]


def parse_box(box, image_w: int, image_h: int) -> list[int]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("crop_box must be [x, y, w, h]")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
        raise ValueError("crop_box values must be integers")
    if any(int(value) != value for value in box):
        raise ValueError("crop_box values must be integers")
    x, y, width, height = (int(value) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError("crop box must have positive size")
    if x < 0 or y < 0 or x + width > image_w or y + height > image_h:
        raise ValueError("crop box is outside the image")
    return [x, y, width, height]


def crop_id_for(rel: str, index: int) -> str:
    return hashlib.sha256(f"{rel}|{index}".encode("utf-8")).hexdigest()[:16]


def hash_crop(path: Path, box: list[int]) -> str:
    x, y, width, height = box
    with Image.open(path) as image:
        cropped = image.convert("RGB").crop((x, y, x + width, y + height))
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _blank_crop(face: dict, index: int) -> dict:
    box = propose_crop(face["box"], face["image_width"], face["image_height"])
    return {
        "crop_id": crop_id_for(face["rel"], index),
        "file": face["rel"],
        "face_box": list(face["box"]),
        "crop_box": box,
        "pose": face["pose"],
        "framing": face["framing"],
        "status": "pending",
        "hash": None,
        "image_width": face["image_width"],
        "image_height": face["image_height"],
    }


def _apply_entries(crops: list[dict], entries: list[dict]) -> None:
    by_id = {crop["crop_id"]: crop for crop in crops}
    for entry in entries:
        crop = by_id.get(entry.get("crop_id"))
        if crop is None:
            continue
        if isinstance(entry.get("crop_box"), list) and len(entry["crop_box"]) == 4:
            crop["crop_box"] = [int(value) for value in entry["crop_box"]]
        crop["hash"] = entry.get("hash")
        decision = entry.get("decision")
        if decision == "approve":
            crop["status"] = "approved"
        elif decision == "reject":
            crop["status"] = "rejected"
        elif decision == "recrop":
            crop["status"] = "pending"


def _view(dataset: Path, crops: list[dict], entries: list[dict]) -> dict:
    pending = sum(1 for crop in crops if crop["status"] == "pending")
    approved = sum(1 for crop in crops if crop["status"] == "approved")
    rejected = sum(1 for crop in crops if crop["status"] == "rejected")
    if not crops:
        enabled = False
        reason = "No faces were detected, so there is nothing to export."
    elif pending:
        enabled = False
        noun = "crop is" if pending == 1 else "crops are"
        reason = (
            f"{pending} {noun} still pending. "
            "Export stays locked until every crop is approved."
        )
    elif rejected or approved != len(crops):
        enabled = False
        noun = "crop is" if rejected == 1 else "crops are"
        reason = (
            f"{rejected} {noun} rejected. "
            "Export stays locked until every crop is approved. Re-crop, then approve."
        )
    else:
        enabled = True
        reason = None
    return {
        "crops": crops,
        "pending_count": pending,
        "approved_count": approved,
        "rejected_count": rejected,
        "export_enabled": enabled,
        "block_reason": reason,
        "log_path": str(log_path(dataset)),
        "log": entries,
    }


def gate_from_faces(dataset: Path, faces: list[dict]) -> dict:
    """Build the queue. ``faces`` are detections tagged with rel and image size."""
    per_file: dict[str, list[dict]] = {}
    for face in faces:
        per_file.setdefault(face["rel"], []).append(face)
    crops: list[dict] = []
    for rel in sorted(per_file):
        ordered = sorted(per_file[rel], key=lambda face: (face["box"][1], face["box"][0]))
        for index, face in enumerate(ordered):
            crops.append(_blank_crop(face, index))
    entries = read_log(log_path(dataset))
    _apply_entries(crops, entries)
    return _view(dataset, crops, entries)


def decide(dataset: Path, faces: list[dict], crop_id: str, decision: str, crop_box=None) -> dict:
    if decision not in {"approve", "reject", "recrop"}:
        raise ValueError("decision must be approve, reject, or recrop")
    current = gate_from_faces(dataset, faces)
    crop = next((item for item in current["crops"] if item["crop_id"] == crop_id), None)
    if crop is None:
        raise UnknownCrop(crop_id)
    box = list(crop["crop_box"])
    if decision == "recrop":
        box = parse_box(crop_box, crop["image_width"], crop["image_height"])
    path = safe_join(dataset, crop["file"])
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": crop["file"],
        "crop_id": crop_id,
        "crop_box": box,
        "hash": hash_crop(path, box),
        "decision": decision,
    }
    append_log(log_path(dataset), entry)
    return gate_from_faces(dataset, faces)


def export_approved(dataset: Path, faces: list[dict], dest: Path | None = None) -> dict:
    view = gate_from_faces(dataset, faces)
    if not view["crops"]:
        return {
            "exported": 0,
            "files": [],
            "dest": None,
            "message": "No crops to export.",
        }
    if not view["export_enabled"]:
        raise ExportBlocked(view["block_reason"] or "export blocked until every crop is approved")
    dataset = dataset.resolve()
    dest = (dest or (dataset / "_training_export")).resolve()
    if dest == dataset or dataset.is_relative_to(dest):
        raise ValueError("refusing to export over the dataset root")
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for crop in view["crops"]:
        if crop["status"] != "approved":
            raise ExportBlocked("refusing to write a crop that is not approved")
        src = safe_join(dataset, crop["file"])
        x, y, width, height = crop["crop_box"]
        with Image.open(src) as image:
            cropped = image.convert("RGB").crop((x, y, x + width, y + height))
            stem = Path(crop["file"]).stem
            out_rel = f"{Path(crop['file']).parent.as_posix()}/{stem}__{crop['crop_id']}.png"
            if out_rel.startswith("./"):
                out_rel = out_rel[2:]
            out_path = dest / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(out_path, format="PNG")
        caption = src.with_suffix(".txt")
        if caption.is_file():
            text = caption.read_text(encoding="utf-8")
            out_path.with_suffix(".txt").write_text(text, encoding="utf-8")
        written.append(out_path.relative_to(dest).as_posix())
        if not src.exists():
            raise RuntimeError("export removed a source file")
    manifest = {
        "exported": written,
        "note": "Approved crops only. Source files were not deleted.",
    }
    (dest / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "exported": len(written),
        "files": written,
        "dest": str(dest),
        "message": f"Wrote {len(written)} approved crops to {dest}. Source files were left in place.",
    }
