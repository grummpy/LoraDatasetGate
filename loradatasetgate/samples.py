"""Procedural sample dataset. Shapes and cartoon ellipses only — no photographs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .coverage import FRAMINGS, POSE_BINS
from .palette import BLUE, GOLD, GREEN, MOUTH, NAVY_BOTTOM, NAVY_TOP, NOSE, ORANGE, PURPLE, SKIN, TEAL, EYE

# Canvas width, height, face width, face height.
FRAMING_CANVAS = {
    "close-up": (1024, 1024, 640, 760),
    "half": (832, 1152, 360, 420),
    "full": (768, 1344, 170, 200),
}

# Eye centers and radii as fractions of the face box: (cx, cy, rx, ry).
POSE_EYES = {
    "front": ((0.35, 0.46, 0.08, 0.10), (0.65, 0.46, 0.08, 0.10)),
    "three_quarter_left": ((0.28, 0.46, 0.055, 0.08), (0.60, 0.46, 0.085, 0.11)),
    "three_quarter_right": ((0.40, 0.46, 0.085, 0.11), (0.72, 0.46, 0.055, 0.08)),
    "profile_left": ((0.34, 0.46, 0.08, 0.10),),
    "profile_right": ((0.66, 0.46, 0.08, 0.10),),
    "up": ((0.35, 0.26, 0.08, 0.10), (0.65, 0.26, 0.08, 0.10)),
    "down": ((0.35, 0.70, 0.08, 0.10), (0.65, 0.70, 0.08, 0.10)),
}

POSE_TINT = {
    "front": 0,
    "three_quarter_left": 6,
    "three_quarter_right": 12,
    "profile_left": 18,
    "profile_right": 22,
    "up": 8,
    "down": 14,
}

CONCEPT = "luna_figurine"
CLASS_TOKEN = "figurine"


def _gradient(width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int], tint: int = 0) -> np.ndarray:
    image = np.zeros((height, width, 3), np.float32)
    span = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    top_c = np.array(top, np.float32) + np.array([0, 0, tint], np.float32)
    bottom_c = np.array(bottom, np.float32) + np.array([0, tint, 0], np.float32)
    column = top_c * (1 - span) + bottom_c * span
    image[:] = column[:, None, :]
    return np.clip(image, 0, 255).astype(np.uint8)


def _ellipse(image: np.ndarray, cx: float, cy: float, rx: float, ry: float, color: tuple[int, int, int]) -> None:
    if rx <= 0 or ry <= 0:
        return
    height, width = image.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    image[mask] = color


def render_face(pose: str, framing: str) -> Image.Image:
    if pose not in POSE_BINS:
        raise ValueError(pose)
    canvas_w, canvas_h, face_w, face_h = FRAMING_CANVAS[framing]
    image = _gradient(canvas_w, canvas_h, NAVY_TOP, NAVY_BOTTOM, POSE_TINT[pose])
    origin_x = (canvas_w - face_w) // 2
    origin_y = (canvas_h - face_h) // 2
    _ellipse(
        image,
        origin_x + face_w / 2,
        origin_y + face_h / 2,
        face_w / 2,
        face_h / 2,
        SKIN,
    )
    nose_y = 0.48 if pose == "down" else 0.62
    mouth_y = 0.88 if pose == "down" else 0.82
    _ellipse(
        image,
        origin_x + 0.50 * face_w,
        origin_y + nose_y * face_h,
        0.07 * face_w,
        0.035 * face_h,
        NOSE,
    )
    _ellipse(
        image,
        origin_x + 0.50 * face_w,
        origin_y + mouth_y * face_h,
        0.16 * face_w,
        0.03 * face_h,
        MOUTH,
    )
    for cx, cy, rx, ry in POSE_EYES[pose]:
        _ellipse(
            image,
            origin_x + cx * face_w,
            origin_y + cy * face_h,
            rx * face_w,
            ry * face_h,
            EYE,
        )
    return Image.fromarray(image, "RGB")


def _triangle(width: int, height: int, fill: tuple[int, int, int]) -> Image.Image:
    image = _gradient(width, height, (8, 16, 28), (18, 12, 36), 0)
    yy, xx = np.mgrid[:height, :width]
    apex_x, apex_y = width * 0.5, height * 0.12
    left_x, right_x = width * 0.12, width * 0.88
    base_y = height * 0.88
    # Barycentric-style: below both slanted edges and above the base.
    denom = (base_y - apex_y) or 1
    progress = (yy - apex_y) / denom
    left_edge = apex_x + (left_x - apex_x) * progress
    right_edge = apex_x + (right_x - apex_x) * progress
    mask = (yy >= apex_y) & (yy <= base_y) & (xx >= left_edge) & (xx <= right_edge)
    image[mask] = fill
    return Image.fromarray(image, "RGB")


def _waves(width: int, height: int, ink: tuple[int, int, int]) -> Image.Image:
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (12, 20, 24)
    yy, xx = np.mgrid[:height, :width]
    phase = np.sin(xx / 28.0) * (height * 0.08)
    for offset in (0.25, 0.5, 0.75):
        center = height * offset + phase
        band = np.abs(yy - center) < max(6, height * 0.03)
        image[band] = ink
    return Image.fromarray(image, "RGB")


def _circles(width: int, height: int, fill: tuple[int, int, int], ring: tuple[int, int, int]) -> Image.Image:
    image = _gradient(width, height, (16, 18, 32), (24, 20, 40), 0)
    cx, cy = width / 2, height / 2
    for radius in (min(width, height) * scale for scale in (0.42, 0.30, 0.16)):
        _ellipse(image, cx, cy, radius, radius * 0.72, ring if int(radius) % 2 == 0 else fill)
    return Image.fromarray(image, "RGB")


def _blocks(width: int, height: int, colors: list[tuple[int, int, int]]) -> Image.Image:
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (18, 22, 34)
    step_x = width // 4
    step_y = height // 3
    for row in range(3):
        for col in range(4):
            color = colors[(row * 4 + col) % len(colors)]
            x0, y0 = col * step_x + 12, row * step_y + 12
            x1, y1 = (col + 1) * step_x - 12, (row + 1) * step_y - 12
            image[y0:y1, x0:x1] = color
    return Image.fromarray(image, "RGB")


def _caption(fields: dict[str, str]) -> str:
    order = ["concept", "class", "pose", "framing", "dup_group", "split"]
    parts = [f"{key}: {fields[key]}" for key in order if fields.get(key)]
    return ", ".join(parts) + "\n"


def _write(path: Path, image: Image.Image, fields: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=70)
    else:
        image.save(path, format="PNG")
    path.with_suffix(".txt").write_text(_caption(fields), encoding="utf-8")


def generate(root: Path) -> dict:
    """Write a deterministic synthetic dataset under ``root``."""
    root = Path(root)
    subject = root / "subject"
    reg = root / "reg"
    subject.mkdir(parents=True, exist_ok=True)
    reg.mkdir(parents=True, exist_ok=True)
    if set(POSE_EYES) != set(POSE_BINS):
        raise RuntimeError("pose drawings and pose bins diverged")
    if set(FRAMING_CANVAS) != set(FRAMINGS):
        raise RuntimeError("framing canvases and framing bins diverged")

    plan = [
        ("face_front_close.png", "front", "close-up", "face_front_close"),
        ("face_front_half.png", "front", "half", "face_front_half"),
        ("face_front_full.png", "front", "full", "face_front_full"),
        ("face_tq_left_close.png", "three_quarter_left", "close-up", "face_tq_left_close"),
        ("face_tq_right_half.png", "three_quarter_right", "half", "face_tq_right_half"),
        ("face_profile_left_close.png", "profile_left", "close-up", "face_profile_left_close"),
        ("face_profile_right_full.png", "profile_right", "full", "face_profile_right_full"),
        ("face_up_half.png", "up", "half", "face_up_half"),
        ("face_down_close.png", "down", "close-up", "face_down_close"),
    ]
    written = []
    front_close = render_face("front", "close-up")
    for name, pose, framing, group in plan:
        image = front_close if name == "face_front_close.png" else render_face(pose, framing)
        path = subject / name
        fields = {
            "concept": CONCEPT,
            "class": CLASS_TOKEN,
            "pose": pose,
            "framing": framing,
            "dup_group": group,
        }
        _write(path, image, fields)
        written.append(path.relative_to(root).as_posix())

    variants = {
        "face_front_close_jpeg.jpg": front_close.filter(ImageFilter.GaussianBlur(radius=1.6)),
        "face_front_close_resize.png": front_close.resize(
            (int(front_close.width * 0.85), int(front_close.height * 0.85)),
            Image.Resampling.LANCZOS,
        ),
        "face_front_close_crop.png": front_close.crop(
            (
                int(front_close.width * 0.05),
                int(front_close.height * 0.05),
                front_close.width - int(front_close.width * 0.05),
                front_close.height - int(front_close.height * 0.05),
            )
        ),
    }
    for name, image in variants.items():
        path = subject / name
        _write(
            path,
            image,
            {
                "concept": CONCEPT,
                "class": CLASS_TOKEN,
                "pose": "front",
                "framing": "close-up",
                "dup_group": "face_front_close",
            },
        )
        written.append(path.relative_to(root).as_posix())

    shapes = {
        "shape_landscape.png": _triangle(1216, 832, TEAL),
        "shape_portrait.png": _circles(640, 1536, ORANGE, PURPLE),
    }
    for name, image in shapes.items():
        path = subject / name
        _write(
            path,
            image,
            {"concept": CONCEPT, "class": CLASS_TOKEN, "dup_group": Path(name).stem},
        )
        written.append(path.relative_to(root).as_posix())

    reg_images = {
        "class_01.png": _blocks(1024, 1024, [BLUE, TEAL, GREEN, GOLD]),
        "class_02.png": _waves(768, 1344, GOLD),
        "class_03.png": _circles(1216, 832, GREEN, BLUE),
        "class_04.png": _blocks(640, 1536, [ORANGE, TEAL, PURPLE, BLUE]),
    }
    for name, image in reg_images.items():
        path = reg / name
        _write(
            path,
            image,
            {"concept": CLASS_TOKEN, "class": CLASS_TOKEN, "split": "reg", "dup_group": Path(name).stem},
        )
        written.append(path.relative_to(root).as_posix())

    note = (
        "Synthetic sample dataset for LoraDatasetGate.\n"
        "Procedural gradients, stripes, blocks, and cartoon ellipses only.\n"
        "No real people. No photographs. No children. No secrets.\n"
    )
    (root / "SYNTHETIC.txt").write_text(note, encoding="utf-8")
    manifest = {"synthetic": True, "files": written, "concept": CONCEPT, "class": CLASS_TOKEN}
    (root / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
