import socket
from pathlib import Path

import pytest

from loradatasetgate.analyze import analyze
from loradatasetgate.safety import OutboundBlocked, assert_loopback_host, install_network_guard, is_loopback_host, uninstall_network_guard

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "__pycache__", "samples", ".pytest_cache"}
# Built without writing the contiguous host strings into this file.
FORBIDDEN = ("192.168.4." + "47", "81" + "88", "comfy" + "ui", "api.openai." + "com")


def test_analyze_opens_no_sockets(dataset, monkeypatch):
    def refuse_connect(self, address):
        raise AssertionError(f"socket connect to {address}")

    def refuse_create(*args, **kwargs):
        raise AssertionError("create_connection")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.setattr(socket, "create_connection", refuse_create)
    report = analyze(dataset)
    assert report["safety"]["network"] == "loopback-only"
    assert report["safety"]["gpu"] is False
    assert report["safety"]["remote_jobs"] is False
    assert report["safety"]["deletes_files"] is False


def test_guard_blocks_non_loopback_and_allows_loopback():
    install_network_guard()
    try:
        with pytest.raises(OutboundBlocked):
            socket.socket().connect(("example.invalid", 9))
        with pytest.raises(OutboundBlocked):
            socket.create_connection(("example.invalid", 9), timeout=0.2)
        sock = socket.socket()
        sock.settimeout(0.3)
        try:
            sock.connect(("127.0.0.1", 1))
        except OutboundBlocked:
            raise AssertionError("loopback was blocked")
        except OSError:
            pass
        finally:
            sock.close()
    finally:
        uninstall_network_guard()


def test_loopback_host_check():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("198.51.100.4")
    assert_loopback_host("127.0.0.1")
    with pytest.raises(Exception):
        assert_loopback_host("0.0.0.0")


def test_package_has_no_delete_or_remote_hosts():
    text_parts = []
    for path in (ROOT / "loradatasetgate").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "os.remove" not in text
        assert ".unlink(" not in text
        assert "rmtree" not in text
        assert "pretrained=\"openai\"" not in text
        assert "pretrained='openai'" not in text
        text_parts.append(text.lower())
    blob = "\n".join(text_parts)
    for token in FORBIDDEN:
        assert token not in blob


def test_repo_text_has_no_remote_training_host():
    allowed = {"BUILD_SHEET.md"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in allowed:
            continue
        if path.suffix.lower() not in {".py", ".js", ".css", ".html", ".md", ".yml", ".yaml", ".sh", ".bat", ".command", ".txt", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for token in FORBIDDEN:
            assert token not in text, f"{token} found in {path}"
