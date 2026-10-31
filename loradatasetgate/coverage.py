"""Pose bins, framing, and SDXL / SD1.5 aspect buckets."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .dataset import ImageItem
from .options import Options
from .palette import SKIN

POSE_BINS = (
    "front",
    "three_quarter_left",
    "three_quarter_right",
    "profile_left",
    "profile_right",
    "up",
    "down",
)
FRAMINGS = ("close-up", "half", "full")

SDXL_BUCKETS = (
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
)
SD15_BUCKETS = (
    (512, 512),
    (768, 512),
    (512, 768),
    (768, 768),
    (640, 512),
    (512, 640),
    (896, 512),
    (512, 896),
)


def pose_label(pose: str) -> str:
    return pose.replace("_", " ")


def suggestion_text(pose: str, framing: str, count: int) -> str:
    noun = "image" if count == 1 else "images"
    return f"add {count} {noun} of {pose_label(pose)} {framing}"


def classify_pose(face_w: int, face_h: int, eyes: list[tuple[float, float, int]]) -> str:
    """Map eye centers (relative to the face box) onto one pose bin.

    Pitch wins over yaw: eyes high in the head are ``up``, eyes low are
    ``down``. One eye is a profile. Two unequal eyes are a three-quarter view.
    """
    if face_w <= 0 or face_h <= 0 or not eyes:
        return "unknown"
    mean_y = sum(eye[1] for eye in eyes) / len(eyes) / face_h
    if mean_y < 0.36:
        return "up"
    if mean_y > 0.60:
        return "down"
    if len(eyes) == 1:
        return "profile_left" if eyes[0][0] < face_w * 0.5 else "profile_right"
    left, right = sorted(eyes, key=lambda eye: eye[0])
    ratio = left[2] / right[2] if right[2] else 1.0
    mid = ((left[0] + right[0]) / 2) / face_w
    if 0.75 <= ratio <= (1 / 0.75) and 0.42 <= mid <= 0.58:
        return "front"
    if ratio < 0.98:
        return "three_quarter_left"
    if ratio > 1.02:
        return "three_quarter_right"
    return "three_quarter_left" if mid < 0.5 else "three_quarter_right"


def classify_framing(face_h: int, image_h: int) -> str:
    if image_h <= 0:
        return "full"
    ratio = face_h / image_h
    if ratio >= 0.55:
        return "close-up"
    if ratio >= 0.28:
        return "half"
    return "full"


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    return (
        (red > 200)
        & (green > 140)
        & (green < 230)
        & (blue > 100)
        & (blue < 210)
        & (red > green)
        & (green > blue)
        & ((red - blue) > 40)
    )


def _select_eyes(comp_u8: np.ndarray, roi_rgb: np.ndarray) -> list[tuple[float, float, int]]:
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(comp_u8.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    filled = np.zeros_like(comp_u8)
    cv2.drawContours(filled, [hull], -1, 255, thickness=-1)
    holes = (filled > 0) & (gray < 60) & (comp_u8 == 0)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    height, width = gray.shape
    found: list[tuple[float, float, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 8 or area > 0.20 * width * height:
            continue
        cx, cy = centroids[index]
        found.append((float(cx), float(cy), area))
    found.sort(key=lambda eye: -eye[2])
    found = found[:2]
    if len(found) == 2 and found[1][2] < 0.30 * found[0][2]:
        found = found[:1]
    return found


def detect_faces(rgb: np.ndarray) -> list[dict]:
    """Find cartoon or high-contrast head ellipses and bin pose and framing.

    A head is a skin-colored component with one or two dark eye holes.
    OpenCV's Haar file is consulted only when this geometric pass finds
    nothing and the cascade file is actually installed. No model is downloaded.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return []
    mask = skin_mask(rgb)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    image_h, image_w = rgb.shape[:2]
    image_area = image_h * image_w
    faces: list[dict] = []
    for index in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[index])
        if area < 0.003 * image_area or width < 24 or height < 24:
            continue
        aspect = width / height
        if not 0.45 <= aspect <= 1.8:
            continue
        roi_labels = labels[y : y + height, x : x + width]
        comp = np.where(roi_labels == index, np.uint8(255), np.uint8(0))
        eyes = _select_eyes(comp, rgb[y : y + height, x : x + width])
        if not eyes:
            continue
        pose = classify_pose(width, height, eyes)
        if pose not in POSE_BINS:
            continue
        faces.append(
            {
                "box": [x, y, width, height],
                "pose": pose,
                "framing": classify_framing(height, image_h),
                "eye_count": len(eyes),
            }
        )
    if not faces:
        faces.extend(_haar_faces(rgb))
    faces.sort(key=lambda face: (face["box"][1], face["box"][0]))
    return faces


def _haar_faces(rgb: np.ndarray) -> list[dict]:
    if not hasattr(cv2, "CascadeClassifier"):
        return []
    cascade_path = ""
    data = getattr(cv2, "data", None)
    if data is not None:
        root = getattr(data, "haarcascades", "")
        if root:
            cascade_path = str(root) + "haarcascade_frontalface_default.xml"
    if not cascade_path:
        return []
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    image_h = rgb.shape[0]
    found = []
    for x, y, width, height in boxes:
        roi = rgb[y : y + height, x : x + width]
        comp = np.where(skin_mask(roi), np.uint8(255), np.uint8(0))
        eyes = _select_eyes(comp, roi) if comp.any() else []
        pose = classify_pose(int(width), int(height), eyes) if eyes else "front"
        if pose not in POSE_BINS:
            pose = "front"
        found.append(
            {
                "box": [int(x), int(y), int(width), int(height)],
                "pose": pose,
                "framing": classify_framing(int(height), image_h),
                "eye_count": len(eyes),
            }
        )
    return found


def assign_bucket(width: int, height: int, buckets: tuple[tuple[int, int], ...]) -> dict:
    aspect = width / height if height else 1.0

    def sort_key(bucket: tuple[int, int]) -> tuple[float, int]:
        bucket_aspect = bucket[0] / bucket[1]
        aspect_cost = abs(math.log(aspect) - math.log(bucket_aspect))
        pixel_cost = abs(width * height - bucket[0] * bucket[1])
        return (aspect_cost, pixel_cost)

    chosen = min(buckets, key=sort_key)
    return {"width": chosen[0], "height": chosen[1], "name": f"{chosen[0]}x{chosen[1]}"}


def find_gaps(counts: dict[tuple[str, str], int], target: int) -> list[dict]:
    gaps = []
    for pose in POSE_BINS:
        for framing in FRAMINGS:
            count = counts.get((pose, framing), 0)
            if count < target:
                missing = target - count
                gaps.append(
                    {
                        "pose": pose,
                        "framing": framing,
                        "count": count,
                        "add": missing,
                        "suggestion": suggestion_text(pose, framing, missing),
                    }
                )
    return gaps


def _bucket_report(items: list[ImageItem], buckets: tuple[tuple[int, int], ...]) -> list[dict]:
    grouped = {
        f"{width}x{height}": {
            "name": f"{width}x{height}",
            "width": width,
            "height": height,
            "count": 0,
            "files": [],
        }
        for width, height in buckets
    }
    for item in items:
        chosen = assign_bucket(item.width, item.height, buckets)
        row = grouped[chosen["name"]]
        row["count"] += 1
        row["files"].append({"rel": item.rel, "split": item.split, "width": item.width, "height": item.height})
    return list(grouped.values())


def coverage_report(items: list[ImageItem], faces: list[dict], options: Options | None = None) -> dict:
    options = options or Options()
    subject_rels = {item.rel for item in items if item.split == "subject"}
    counts: dict[tuple[str, str], int] = {}
    subject_faces = []
    other_faces = []
    faced_rels = set()
    for face in faces:
        bucket = subject_faces if face["rel"] in subject_rels else other_faces
        bucket.append(face)
        if face["rel"] in subject_rels and face["pose"] in POSE_BINS:
            key = (face["pose"], face["framing"])
            counts[key] = counts.get(key, 0) + 1
            faced_rels.add(face["rel"])
    heatmap = []
    for pose in POSE_BINS:
        for framing in FRAMINGS:
            count = counts.get((pose, framing), 0)
            files = [
                face["rel"]
                for face in subject_faces
                if face["pose"] == pose and face["framing"] == framing
            ]
            heatmap.append(
                {
                    "pose": pose,
                    "framing": framing,
                    "count": count,
                    "gap": count < options.target_per_cell,
                    "files": files,
                }
            )
    no_face = [item.rel for item in items if item.split == "subject" and item.rel not in faced_rels]
    return {
        "poses": list(POSE_BINS),
        "framings": list(FRAMINGS),
        "target_per_cell": options.target_per_cell,
        "heatmap": heatmap,
        "gaps": find_gaps(counts, options.target_per_cell),
        "faces": subject_faces,
        "reg_faces": other_faces,
        "no_face": no_face,
        "buckets": {
            "sdxl": _bucket_report(items, SDXL_BUCKETS),
            "sd15": _bucket_report(items, SD15_BUCKETS),
        },
    }


def skin_color_ok() -> bool:
    """The shared skin swatch must pass the detector mask."""
    swatch = np.zeros((8, 8, 3), dtype=np.uint8)
    swatch[:] = SKIN
    return bool(skin_mask(swatch).all())
