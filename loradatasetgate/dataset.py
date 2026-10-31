"""Scan a dataset folder into subject and regularization images."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REG_DIR_NAMES = {"reg", "class", "regularization", "reg_images", "class_images"}
SKIP_DIR_NAMES = {"_rejected", "_training_export", "_reports", ".loradatasetgate"}


@dataclass(frozen=True)
class ImageItem:
    rel: str
    path: Path
    split: str
    caption: str
    fields: dict[str, str]
    width: int
    height: int


def parse_caption(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            fields[key] = value
    return fields


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _is_reg(path: Path, root: Path, fields: dict[str, str]) -> bool:
    parents = path.relative_to(root).parts[:-1]
    if any(part.lower() in REG_DIR_NAMES for part in parents):
        return True
    return fields.get("split", "").lower() in {"reg", "class", "regularization"}


def scan(root: Path) -> tuple[list[ImageItem], list[str]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {root}")
    items: list[ImageItem] = []
    errors: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith(".")
        )
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as im:
                    width, height = im.size
            except Exception as exc:
                errors.append(f"{rel}: {exc}")
                continue
            caption_path = path.with_suffix(".txt")
            caption = ""
            if caption_path.is_file():
                caption = caption_path.read_text(encoding="utf-8", errors="replace").strip()
            fields = parse_caption(caption)
            items.append(
                ImageItem(
                    rel=rel,
                    path=path,
                    split="reg" if _is_reg(path, root, fields) else "subject",
                    caption=caption,
                    fields=fields,
                    width=width,
                    height=height,
                )
            )
    items.sort(key=lambda item: item.rel)
    return items, errors
