from fastapi.testclient import TestClient

from loradatasetgate.app import app
from loradatasetgate.__main__ import main


def test_pages_assets_and_full_gate_flow(dataset):
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    for label in ("Dataset", "Duplicates", "Balance", "Coverage", "Face Gate", "Report", "Run all checks"):
        assert label in page.text
    assert "/art/icon.png" in page.text
    assert "/art/cover.png" in page.text
    assert client.get("/art/icon.png").status_code == 200
    assert client.get("/art/cover.png").content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get("/favicon.ico").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["gpu"] is False
    assert health["deletes_files"] is False

    analyzed = client.post("/api/analyze", json={"path": str(dataset)})
    assert analyzed.status_code == 200
    body = analyzed.json()
    assert body["duplicates"]["multi_group_count"] == 1
    assert body["balance"]["action"] == "add"
    assert body["gate"]["export_enabled"] is False
    assert body["gate"]["pending_count"] == len(body["gate"]["crops"])

    blocked = client.post("/api/gate/export", json={"path": str(dataset)})
    assert blocked.status_code == 409
    assert "approved" in blocked.json()["detail"].lower()

    for crop in body["gate"]["crops"]:
        decision = client.post(
            "/api/gate/decision",
            json={"path": str(dataset), "crop_id": crop["crop_id"], "decision": "approve"},
        )
        assert decision.status_code == 200

    exported = client.post("/api/gate/export", json={"path": str(dataset)})
    assert exported.status_code == 200
    payload = exported.json()
    assert payload["exported"] == len(body["gate"]["crops"])
    assert "left in place" in payload["message"]
    for rel in payload["files"]:
        assert (dataset / "_training_export" / rel).is_file()
    assert (dataset / "subject" / "face_front_close.png").is_file()

    log_file = dataset / ".loradatasetgate" / "gate_log.jsonl"
    snapshot = log_file.read_bytes()
    extra = client.post(
        "/api/gate/decision",
        json={"path": str(dataset), "crop_id": body["gate"]["crops"][0]["crop_id"], "decision": "approve"},
    )
    assert extra.status_code == 200
    assert log_file.read_bytes().startswith(snapshot)

    csv_response = client.post("/api/duplicates/csv", json={"path": str(dataset)})
    assert csv_response.status_code == 200
    assert "face_front_close.png" in csv_response.text
    refused = client.post(
        "/api/duplicates/move",
        json={"path": str(dataset), "files": ["subject/shape_portrait.png"], "confirm": False},
    )
    assert refused.status_code == 400
    assert (dataset / "subject" / "shape_portrait.png").is_file()
    escaped = client.get("/api/media", params={"dataset": str(dataset), "rel": "../etc/passwd"})
    assert escaped.status_code == 400
    report = client.post("/api/report/html", json={"path": str(dataset)})
    assert report.status_code == 200
    assert "add 1 image of profile left full" in report.text
    assert "Add 10 regularization" in report.text
    note = (dataset / "SYNTHETIC.txt").read_text(encoding="utf-8").lower()
    assert "synthetic" in note
    assert "no real people" in note


def test_server_refuses_public_bind():
    assert main(["--host", "0.0.0.0"]) == 2


def test_samples_endpoint_is_marked_synthetic():
    client = TestClient(app)
    response = client.get("/api/samples")
    assert response.status_code == 200
    body = response.json()
    assert body["synthetic"] is True
    assert body["path"]
