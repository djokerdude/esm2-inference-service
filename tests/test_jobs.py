"""
Tests for async job endpoints.

FastAPI's TestClient runs background tasks synchronously after the response,
so jobs are complete by the time client.post() returns. The poll helper
exists to document the intended usage pattern and to be safe if that
behaviour ever changes.
"""
import time
from fastapi.testclient import TestClient

HEMOGLOBIN_FRAGMENT = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPT"


def _poll(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("complete", "failed"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# POST /jobs/embed
# ---------------------------------------------------------------------------

def test_submit_embed_job_returns_202(client):
    r = client.post("/jobs/embed", json={"sequences": [HEMOGLOBIN_FRAGMENT]})
    assert r.status_code == 202


def test_submit_embed_job_returns_job_id(client):
    r = client.post("/jobs/embed", json={"sequences": [HEMOGLOBIN_FRAGMENT]})
    body = r.json()
    assert "job_id" in body
    assert len(body["job_id"]) == 36  # UUID4 format


def test_embed_job_completes_with_correct_shape(client):
    r = client.post("/jobs/embed", json={"sequences": [HEMOGLOBIN_FRAGMENT, "MKTLL"]})
    job_id = r.json()["job_id"]
    job = _poll(client, job_id)

    assert job["status"] == "complete"
    assert job["result"]["count"] == 2
    assert job["result"]["dim"] == 320
    assert len(job["result"]["embeddings"]) == 2


def test_embed_job_invalid_sequence_fails(client):
    r = client.post("/jobs/embed", json={"sequences": ["INVALID123"]})
    job_id = r.json()["job_id"]
    job = _poll(client, job_id)
    assert job["status"] == "failed"
    assert "error" in job


# ---------------------------------------------------------------------------
# POST /jobs/annotate
# ---------------------------------------------------------------------------

def test_submit_annotate_job_returns_202(client):
    r = client.post("/jobs/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT})
    assert r.status_code == 202


def test_annotate_job_completes(client):
    r = client.post("/jobs/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 3})
    job_id = r.json()["job_id"]
    job = _poll(client, job_id)
    assert job["status"] == "complete"


def test_annotate_job_result_has_go_terms(client):
    r = client.post("/jobs/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 5})
    job = _poll(client, r.json()["job_id"])
    assert len(job["result"]["predicted_go_terms"]) > 0


def test_annotate_job_result_has_neighbors(client):
    r = client.post("/jobs/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 3})
    job = _poll(client, r.json()["job_id"])
    assert len(job["result"]["nearest_neighbors"]) == 3


def test_annotate_job_has_inference_time(client):
    r = client.post("/jobs/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT})
    job = _poll(client, r.json()["job_id"])
    assert job["result"]["inference_time_ms"] > 0


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------

def test_poll_unknown_job_returns_404(client):
    r = client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_completed_job_includes_duration(client):
    r = client.post("/jobs/embed", json={"sequences": ["MKTLL"]})
    job = _poll(client, r.json()["job_id"])
    assert job["duration_ms"] is not None
    assert job["duration_ms"] >= 0


def test_completed_job_includes_timestamps(client):
    r = client.post("/jobs/embed", json={"sequences": ["MKTLL"]})
    job = _poll(client, r.json()["job_id"])
    assert job["created_at"] is not None
    assert job["completed_at"] is not None
    assert job["completed_at"] >= job["created_at"]


def test_two_jobs_have_different_ids(client):
    r1 = client.post("/jobs/embed", json={"sequences": ["MKTLL"]})
    r2 = client.post("/jobs/embed", json={"sequences": ["MVLSP"]})
    assert r1.json()["job_id"] != r2.json()["job_id"]
