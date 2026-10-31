"""Run every checker and assemble one report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .balancer import balance
from .coverage import coverage_report, detect_faces
from .dataset import load_rgb_image, scan
from .facegate import gate_from_faces
from .neardup import find_duplicates
from .options import Options


def detect_dataset_faces(items) -> list[dict]:
    faces = []
    for item in items:
        rgb = np.asarray(load_rgb_image(item.path))
        for face in detect_faces(rgb):
            faces.append(
                {
                    **face,
                    "rel": item.rel,
                    "image_width": item.width,
                    "image_height": item.height,
                    "split": item.split,
                }
            )
    return faces


def analyze(dataset: Path, options: Options | None = None) -> dict:
    options = options or Options()
    dataset = Path(dataset).resolve()
    items, errors = scan(dataset)
    faces = detect_dataset_faces(items)
    subject_count = sum(1 for item in items if item.split == "subject")
    reg_count = sum(1 for item in items if item.split == "reg")
    captioned = sum(1 for item in items if item.caption)
    return {
        "dataset": str(dataset),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "counts": {
            "subject": subject_count,
            "reg": reg_count,
            "total": len(items),
            "captioned": captioned,
        },
        "errors": errors,
        "duplicates": find_duplicates(items, options),
        "balance": balance(items, options),
        "coverage": coverage_report(items, faces, options),
        "gate": gate_from_faces(dataset, faces),
        "safety": {
            "deletes_files": False,
            "network": "loopback-only",
            "gpu": False,
            "remote_jobs": False,
        },
    }
