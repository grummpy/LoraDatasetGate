#!/usr/bin/env bash
# One-click launcher. Creates a venv, installs deps, starts the loopback UI,
# and opens a browser. Uses art/icon.png and art/cover.png from this folder.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f art/icon.png || ! -f art/cover.png ]]; then
  echo "Missing art/icon.png or art/cover.png" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python 3.11 or newer is required." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "LoraDatasetGate — art/icon.png and art/cover.png"
python -m loradatasetgate --host 127.0.0.1 --port 8765 --no-browser &
pid=$!
cleanup() { kill "$pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for _ in $(seq 1 50); do
  if python -c 'import socket; socket.create_connection(("127.0.0.1", 8765), 0.3).close()' 2>/dev/null; then
    break
  fi
  sleep 0.2
done

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://127.0.0.1:8765/" >/dev/null 2>&1 || true
else
  echo "Open http://127.0.0.1:8765/"
fi
wait "$pid"
