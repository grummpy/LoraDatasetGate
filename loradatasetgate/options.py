"""Check settings. Defaults match a 1:1 to 1:2 reg-per-subject target."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Options:
    repeats: int = 1
    reg_per_subject_min: float = 1.0
    reg_per_subject_max: float = 2.0
    phash_threshold: int = 10
    dhash_threshold: int = 8
    hist_threshold: float = 0.97
    orb_min_matches: int = 8
    hash_band: int = 2
    target_per_cell: int = 1

    @classmethod
    def from_dict(cls, data: dict | None) -> "Options":
        data = dict(data or {})
        opts = cls()
        for key in (
            "repeats",
            "reg_per_subject_min",
            "reg_per_subject_max",
            "phash_threshold",
            "dhash_threshold",
            "hist_threshold",
            "orb_min_matches",
            "hash_band",
            "target_per_cell",
        ):
            if data.get(key) is not None:
                current = getattr(opts, key)
                setattr(opts, key, type(current)(data[key]))
        if opts.repeats < 1:
            raise ValueError("repeats must be an integer >= 1")
        if opts.reg_per_subject_min <= 0 or opts.reg_per_subject_max < opts.reg_per_subject_min:
            raise ValueError("regularization ratio must satisfy 0 < min <= max")
        if not 0 <= opts.phash_threshold <= 64 or not 0 <= opts.dhash_threshold <= 64:
            raise ValueError("hash thresholds must be between 0 and 64")
        if opts.hash_band < 0 or opts.hash_band > 32:
            raise ValueError("hash_band must be between 0 and 32")
        if opts.target_per_cell < 1:
            raise ValueError("target_per_cell must be >= 1")
        if not 0 <= opts.hist_threshold <= 1:
            raise ValueError("hist_threshold must be between 0 and 1")
        if opts.orb_min_matches < 1:
            raise ValueError("orb_min_matches must be >= 1")
        return opts
