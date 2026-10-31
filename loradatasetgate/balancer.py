"""Subject vs regularization counts and a target ratio."""

from __future__ import annotations

import math

from .dataset import ImageItem
from .options import Options


def recommend(
    subject_count: int,
    reg_count: int,
    repeats: int = 1,
    reg_per_subject_min: float = 1.0,
    reg_per_subject_max: float = 2.0,
) -> dict:
    """How many reg images to add or remove for the configured ratio.

    The target is ``reg_per_subject_min`` to ``reg_per_subject_max``
    regularization images for each subject image times ``repeats``.
    The default 1 to 2 is the sheet's 1:1 to 1:2. Removal is advice only;
    this function never touches files.
    """
    repeats = int(repeats)
    effective = subject_count * repeats
    if subject_count <= 0:
        return {
            "subject_count": subject_count,
            "reg_count": reg_count,
            "repeats": repeats,
            "effective_subject": 0,
            "reg_per_subject": None,
            "reg_per_effective": None,
            "reg_per_subject_min": reg_per_subject_min,
            "reg_per_subject_max": reg_per_subject_max,
            "target_min": 0,
            "target_max": 0,
            "action": "none",
            "count": 0,
            "message": "No subject images found. Add subject images before balancing regularization.",
        }
    target_min = math.ceil(effective * reg_per_subject_min - 1e-9)
    target_max = math.floor(effective * reg_per_subject_max + 1e-9)
    if target_max < target_min:
        target_max = target_min
    if reg_count < target_min:
        action = "add"
        count = target_min - reg_count
        noun = "image" if count == 1 else "images"
        message = (
            f"Add {count} regularization {noun} to reach the {reg_per_subject_min:g}:1 minimum. "
            f"With {subject_count} subject images × {repeats} repeats, the target is "
            f"{target_min}–{target_max} regularization images. You have {reg_count}."
        )
    elif reg_count > target_max:
        action = "remove"
        count = reg_count - target_max
        noun = "image" if count == 1 else "images"
        message = (
            f"The set is above the {reg_per_subject_max:g}:1 maximum. "
            f"You could leave out {count} regularization {noun} "
            f"(target {target_min}–{target_max}, you have {reg_count}). "
            "This is advice only — nothing is deleted."
        )
    else:
        action = "ok"
        count = 0
        message = (
            f"Regularization is inside the target range "
            f"({reg_count} images, target {target_min}–{target_max} "
            f"for {subject_count} subject images × {repeats} repeats)."
        )
    return {
        "subject_count": subject_count,
        "reg_count": reg_count,
        "repeats": repeats,
        "effective_subject": effective,
        "reg_per_subject": reg_count / subject_count,
        "reg_per_effective": reg_count / effective,
        "reg_per_subject_min": reg_per_subject_min,
        "reg_per_subject_max": reg_per_subject_max,
        "target_min": target_min,
        "target_max": target_max,
        "action": action,
        "count": count,
        "message": message,
    }


def _reg_matches(item: ImageItem, class_token: str) -> bool:
    if not class_token:
        return False
    return item.fields.get("class") == class_token or item.fields.get("concept") == class_token


def balance(items: list[ImageItem], options: Options | None = None) -> dict:
    options = options or Options()
    subjects = [item for item in items if item.split == "subject"]
    regs = [item for item in items if item.split == "reg"]
    summary = recommend(
        len(subjects),
        len(regs),
        options.repeats,
        options.reg_per_subject_min,
        options.reg_per_subject_max,
    )
    grouped: dict[str, dict] = {}
    for item in subjects:
        concept = item.fields.get("concept") or "(no concept)"
        row = grouped.setdefault(
            concept,
            {"concept": concept, "class_token": item.fields.get("class") or ""},
        )
        row["subject_images"] = row.get("subject_images", 0) + 1
        if item.fields.get("class") and not row["class_token"]:
            row["class_token"] = item.fields["class"]
    concepts = []
    for concept in sorted(grouped):
        row = grouped[concept]
        token = row["class_token"]
        reg_n = sum(1 for reg in regs if _reg_matches(reg, token))
        rec = recommend(
            row["subject_images"],
            reg_n,
            options.repeats,
            options.reg_per_subject_min,
            options.reg_per_subject_max,
        )
        concepts.append(
            {
                "concept": concept,
                "class_token": token or None,
                "subject_images": row["subject_images"],
                "reg_images": reg_n,
                "action": rec["action"],
                "count": rec["count"],
                "target_min": rec["target_min"],
                "target_max": rec["target_max"],
                "message": rec["message"],
            }
        )
    summary["concepts"] = concepts
    return summary
