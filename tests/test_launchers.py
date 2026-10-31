from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launchers_create_a_venv_and_bind_loopback():
    expected = {
        "launch-linux.sh": "xdg-open",
        "launch-mac.command": "open ",
        "launch-windows.bat": "start ",
    }
    for name, opener in expected.items():
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "venv" in text
        assert "pip install" in text
        assert "127.0.0.1" in text
        assert "art/icon.png" in text or "art\\icon.png" in text
        assert "art/cover.png" in text or "art\\cover.png" in text
        assert opener in text
        assert "0.0.0.0" not in text
        assert "--host 127.0.0.1" in text or "--host 127.0.0.1" in text.replace("\r", "")
