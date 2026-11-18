import io

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["app"]


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "offline" in body["providers_available"]


def test_models_listing():
    r = client.get("/api/models")
    assert r.status_code == 200
    assert len(r.json()) >= 10


def test_models_permissive_filter():
    r = client.get("/api/models", params={"permissive_only": True})
    assert r.status_code == 200
    assert all(m["commercial_ok"] for m in r.json())


def test_analyze_and_fetch(wav_bytes):
    files = {"file": ("sample.wav", wav_bytes, "audio/wav")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analysis_id"]
    assert len(body["graphs"]) >= 5
    assert body["disclaimer"]

    aid = body["analysis_id"]
    assert client.get(f"/api/analysis/{aid}").status_code == 200
    assert client.get(f"/api/diagnostics/{aid}/graphs").status_code == 200
    assert client.get(f"/api/diagnostics/{aid}/graphs/0").status_code == 200


def test_analyze_rejects_bad_extension():
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 415


def test_analysis_not_found():
    assert client.get("/api/analysis/does-not-exist").status_code == 404


def test_agent_endpoint_with_analysis(wav_bytes):
    files = {"file": ("sample.wav", wav_bytes, "audio/wav")}
    aid = client.post("/api/analyze", files=files).json()["analysis_id"]
    r = client.post("/api/agent", json={"question": "Lifestyle tips?", "analysis_id": aid})
    assert r.status_code == 200
    assert r.json()["answer"]


def test_agent_providers():
    r = client.get("/api/agent/providers")
    assert r.status_code == 200
    assert "offline" in r.json()["providers"]
