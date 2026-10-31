"""One-click entry: generate samples if needed, then serve on 127.0.0.1."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from .safety import NonLoopbackHost, assert_loopback_host, install_network_guard
from .samples import generate

ROOT_HELP = "LoraDatasetGate local dataset checker"


def _open_when_ready(port: int) -> None:
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.2)
    try:
        webbrowser.open(f"http://127.0.0.1:{port}/")
    except Exception:
        print(f"Open http://127.0.0.1:{port}/ in a browser.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=ROOT_HELP)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        assert_loopback_host(args.host)
    except NonLoopbackHost as exc:
        print(exc, file=sys.stderr)
        return 2
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    icon = root / "art" / "icon.png"
    cover = root / "art" / "cover.png"
    if not icon.is_file() or not cover.is_file():
        print("Missing art/icon.png or art/cover.png.", file=sys.stderr)
        return 2
    samples = root / "samples"
    if not (samples / "subject").is_dir():
        print("Generating synthetic samples…")
        generate(samples)
    install_network_guard()
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(args.port,), daemon=True).start()
    print(f"LoraDatasetGate at http://127.0.0.1:{args.port}/  (CPU only, no outbound network)")
    uvicorn.run(
        "loradatasetgate.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
