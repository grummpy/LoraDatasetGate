"""Loopback-only networking and path checks.

The running app installs a socket guard so dataset checks cannot open
outbound connections. Loopback is allowed for the local UI.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path

LOOPBACK_NAMES = {"localhost"}

_orig_connect = None
_orig_connect_ex = None
_orig_create_connection = None
_installed = False


class NonLoopbackHost(ValueError):
    """Raised when a bind address is not loopback."""


class OutboundBlocked(RuntimeError):
    """Raised when code tries to open a non-loopback socket."""


def is_loopback_host(host: str) -> bool:
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def assert_loopback_host(host: str) -> None:
    if not is_loopback_host(host):
        raise NonLoopbackHost(
            f"Refusing to bind {host}. LoraDatasetGate listens on 127.0.0.1 only."
        )


def _host_of(address) -> str:
    if isinstance(address, tuple) and address:
        host = address[0]
    elif isinstance(address, str):
        host = address
    else:
        host = ""
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    return str(host)


def _check_address(address) -> None:
    host = _host_of(address)
    if not is_loopback_host(host):
        raise OutboundBlocked(
            f"LoraDatasetGate blocked a non-loopback connection to {address!r}."
        )


def install_network_guard() -> None:
    """Reject connect() and create_connection() except to loopback."""
    global _orig_connect, _orig_connect_ex, _orig_create_connection, _installed
    if _installed:
        return
    _orig_connect = socket.socket.connect
    _orig_connect_ex = socket.socket.connect_ex
    _orig_create_connection = socket.create_connection

    def guarded_connect(self, address):
        _check_address(address)
        return _orig_connect(self, address)

    def guarded_connect_ex(self, address):
        _check_address(address)
        return _orig_connect_ex(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        _check_address(address)
        return _orig_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection
    _installed = True


def uninstall_network_guard() -> None:
    global _installed
    if not _installed:
        return
    socket.socket.connect = _orig_connect
    socket.socket.connect_ex = _orig_connect_ex
    socket.create_connection = _orig_create_connection
    _installed = False


def safe_join(root: Path, rel: str) -> Path:
    """Resolve ``rel`` inside ``root`` or raise ValueError."""
    root_resolved = root.resolve()
    raw = Path(rel)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"path escapes the dataset: {rel}")
    target = (root_resolved / raw).resolve()
    if not target.is_relative_to(root_resolved):
        raise ValueError(f"path escapes the dataset: {rel}")
    return target


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    number = 2
    while True:
        candidate = path.with_name(f"{stem}__{number}{suffix}")
        if not candidate.exists():
            return candidate
        number += 1


def move_into(src: Path, dest: Path) -> Path:
    """Rename ``src`` to ``dest``. Never deletes either file.

    Destination collisions get a numeric suffix. Cross-device rename is
    refused rather than copy-and-delete, so a file is never unlinked.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = unique_path(dest)
    try:
        os.rename(src, dest)
    except OSError as exc:
        raise OSError(
            f"Could not rename {src} to {dest} without deleting it ({exc}). "
            "Left the original in place."
        ) from exc
    return dest
