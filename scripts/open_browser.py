#!/usr/bin/env python3
"""Open the local UI once it is accepting connections on 127.0.0.1."""

from __future__ import annotations

import socket
import sys
import time
import webbrowser


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    url = f"http://127.0.0.1:{port}/"
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), 0.3):
                break
        except OSError:
            time.sleep(0.2)
    webbrowser.open(url)


if __name__ == "__main__":
    main()
