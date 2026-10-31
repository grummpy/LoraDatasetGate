#!/usr/bin/env python3
"""Write the synthetic sample dataset (shapes and cartoon ellipses only)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loradatasetgate.samples import generate  # noqa: E402


def main() -> None:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "samples"
    manifest = generate(dest)
    print(f"Wrote {len(manifest['files'])} synthetic images to {dest}")


if __name__ == "__main__":
    main()
