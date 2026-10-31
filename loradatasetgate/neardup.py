"""Near-duplicate groups from pHash/dHash, with a CPU similarity fallback.

open_clip ViT-B-32 is optional and only runs from a local weight file.
Nothing here downloads a model. Histogram correlation and ORB are the
fallback when that embedding is not configured.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from .dataset import ImageItem
from .options import Options
from .safety import move_into, safe_join

CLIP_SIMILARITY = 0.94


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def should_link(
    *,
    phash_dist: int,
    dhash_dist: int,
    hist_corr: float,
    orb_matches: int,
    clip_similarity: float | None,
    options: Options,
) -> bool:
    """Cluster on pHash, then confirm near-misses.

    dHash, a local CLIP embedding, or histogram+ORB may link a pair only when
    pHash is already inside ``threshold + hash_band``. A shared background can
    make dHash small for unrelated images, so dHash never links on its own.
    """
    if phash_dist <= options.phash_threshold:
        return True
    if phash_dist > options.phash_threshold + options.hash_band:
        return False
    if dhash_dist <= options.dhash_threshold:
        return True
    if clip_similarity is not None:
        return clip_similarity >= CLIP_SIMILARITY
    return hist_corr >= options.hist_threshold and orb_matches >= options.orb_min_matches


def color_histogram(rgb: np.ndarray) -> np.ndarray:
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 12000:
        step = max(1, len(pixels) // 12000)
        pixels = pixels[::step]
    hist, _ = np.histogramdd(
        pixels,
        bins=8,
        range=((0, 256), (0, 256), (0, 256)),
    )
    flat = hist.astype(np.float64).ravel()
    total = flat.sum()
    if total:
        flat /= total
    return flat


def histogram_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_c = left - left.mean()
    right_c = right - right.mean()
    denom = float(np.linalg.norm(left_c) * np.linalg.norm(right_c))
    if denom == 0:
        return 0.0
    return float(np.dot(left_c, right_c) / denom)


def laplacian_variance(path: Path) -> float:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L"))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def orb_matches(path_a: Path, path_b: Path) -> int:
    gray_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    gray_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if gray_a is None or gray_b is None:
        return 0
    orb = cv2.ORB_create(nfeatures=200)
    _, desc_a = orb.detectAndCompute(gray_a, None)
    _, desc_b = orb.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_a, desc_b)
    return sum(1 for match in matches if match.distance < 48)


def optional_clip_similarity(path_a: Path, path_b: Path) -> float | None:
    """CPU cosine similarity of two images with a local ViT-B-32 checkpoint.

    Returns None unless ``LORADATASETGATE_CLIP=1`` and
    ``LORADATASETGATE_CLIP_WEIGHTS`` is an existing file. Weights are loaded
    from that path only. This function does not contact a model host.
    """
    if os.environ.get("LORADATASETGATE_CLIP") != "1":
        return None
    weights = os.environ.get("LORADATASETGATE_CLIP_WEIGHTS", "")
    if not weights or not Path(weights).is_file():
        return None
    try:
        import open_clip
        import torch
    except ImportError:
        return None
    try:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        def embed(path: Path):
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                feature = model.encode_image(image)
                feature = feature / feature.norm(dim=-1, keepdim=True)
            return feature

        left = embed(path_a)
        right = embed(path_b)
        return float((left @ right.T).item())
    except Exception:
        return None


def _records(items: list[ImageItem]) -> list[dict]:
    records = []
    for item in items:
        with Image.open(item.path) as image:
            rgb_image = image.convert("RGB")
            phash = imagehash.phash(rgb_image)
            dhash = imagehash.dhash(rgb_image)
            rgb = np.asarray(rgb_image)
        sharp = laplacian_variance(item.path)
        records.append(
            {
                "rel": item.rel,
                "path": item.path,
                "split": item.split,
                "width": item.width,
                "height": item.height,
                "pixels": item.width * item.height,
                "sharpness": sharp,
                "phash": phash,
                "dhash": dhash,
                "hist": color_histogram(rgb),
            }
        )
    return records


def find_duplicates(items: list[ImageItem], options: Options | None = None) -> dict:
    options = options or Options()
    records = _records(items)
    union = UnionFind(len(records))
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            phash_dist = int(records[left]["phash"] - records[right]["phash"])
            dhash_dist = int(records[left]["dhash"] - records[right]["dhash"])
            hist = 0.0
            orb = 0
            clip = None
            in_band = options.phash_threshold < phash_dist <= options.phash_threshold + options.hash_band
            if in_band and dhash_dist > options.dhash_threshold:
                clip = optional_clip_similarity(records[left]["path"], records[right]["path"])
                if clip is None:
                    hist = histogram_correlation(records[left]["hist"], records[right]["hist"])
                    if hist >= options.hist_threshold:
                        orb = orb_matches(records[left]["path"], records[right]["path"])
            if should_link(
                phash_dist=phash_dist,
                dhash_dist=dhash_dist,
                hist_corr=hist,
                orb_matches=orb,
                clip_similarity=clip,
                options=options,
            ):
                union.union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(records)):
        grouped.setdefault(union.find(index), []).append(index)
    groups = []
    for number, indexes in enumerate(sorted(grouped.values(), key=lambda ids: records[ids[0]]["rel"]), start=1):
        members = []
        for index in indexes:
            record = records[index]
            members.append(
                {
                    "rel": record["rel"],
                    "split": record["split"],
                    "width": record["width"],
                    "height": record["height"],
                    "pixels": record["pixels"],
                    "sharpness": round(record["sharpness"], 4),
                    "phash": str(record["phash"]),
                    "dhash": str(record["dhash"]),
                    "_sharp": record["sharpness"],
                }
            )
        keep = max(
            members,
            key=lambda member: (
                member["pixels"],
                member["_sharp"],
                tuple(-ord(char) for char in member["rel"]),
            ),
        )
        for member in members:
            member["role"] = "keep" if member["rel"] == keep["rel"] else "drop"
            member.pop("_sharp", None)
        members.sort(key=lambda member: (member["role"] != "keep", member["rel"]))
        groups.append(
            {
                "id": number,
                "size": len(members),
                "suggested_keep": keep["rel"],
                "keep_reason": "highest resolution, then sharpness (Laplacian variance)",
                "images": members,
            }
        )
    groups.sort(key=lambda group: (-group["size"], group["suggested_keep"]))
    for number, group in enumerate(groups, start=1):
        group["id"] = number
    multi = [group for group in groups if group["size"] > 1]
    return {
        "groups": groups,
        "multi_groups": multi,
        "multi_group_count": len(multi),
        "unique_count": sum(1 for group in groups if group["size"] == 1),
        "thresholds": {
            "phash": options.phash_threshold,
            "dhash": options.dhash_threshold,
            "hist": options.hist_threshold,
            "hash_band": options.hash_band,
        },
        "similarity": (
            "pHash and dHash, with histogram and ORB inside a narrow hash band "
            "when a local open_clip ViT-B-32 checkpoint is not configured"
        ),
        "clip_enabled": os.environ.get("LORADATASETGATE_CLIP") == "1"
        and bool(os.environ.get("LORADATASETGATE_CLIP_WEIGHTS")),
    }


def duplicates_to_csv(duplicate_report: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["filepath", "group_id", "role", "width", "height", "pixels", "sharpness", "phash", "dhash"]
    )
    for group in duplicate_report["groups"]:
        for image in group["images"]:
            writer.writerow(
                [
                    image["rel"],
                    group["id"],
                    image["role"],
                    image["width"],
                    image["height"],
                    image["pixels"],
                    image["sharpness"],
                    image["phash"],
                    image["dhash"],
                ]
            )
    return buffer.getvalue()


def move_rejected(dataset: Path, relative_paths: list[str]) -> list[str]:
    """Move the named files into ``_rejected/``. Never deletes them."""
    dataset = dataset.resolve()
    moved: list[str] = []
    for rel in relative_paths:
        if rel.startswith("_") or ".loradatasetgate" in rel.split("/"):
            raise ValueError(f"refusing to move internal path: {rel}")
        src = safe_join(dataset, rel)
        if not src.is_file():
            raise FileNotFoundError(rel)
        dest = move_into(src, dataset / "_rejected" / rel)
        caption = src.with_suffix(".txt")
        if caption.is_file():
            move_into(caption, dest.with_suffix(".txt"))
        moved.append(dest.relative_to(dataset).as_posix())
    return moved
