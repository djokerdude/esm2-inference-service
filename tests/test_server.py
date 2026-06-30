"""
Integration tests against the live FastAPI app.
The session-scoped `client` fixture (conftest.py) loads the model and builds
the FAISS index once — all tests here share that startup cost.
"""

HEMOGLOBIN_FRAGMENT = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPT"
SYNUCLEIN_FRAGMENT  = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVL"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["index_ready"] is True


# ---------------------------------------------------------------------------
# POST /embed
# ---------------------------------------------------------------------------

def test_embed_returns_320_dim_vector(client):
    r = client.post("/embed", json={"sequence": HEMOGLOBIN_FRAGMENT})
    assert r.status_code == 200
    body = r.json()
    assert body["dim"] == 320
    assert len(body["embedding"]) == 320


def test_embed_all_floats(client):
    r = client.post("/embed", json={"sequence": "MKTLL"})
    assert r.status_code == 200
    assert all(isinstance(v, float) for v in r.json()["embedding"])


def test_embed_empty_sequence_returns_422(client):
    r = client.post("/embed", json={"sequence": ""})
    assert r.status_code == 422


def test_embed_invalid_characters_returns_422(client):
    r = client.post("/embed", json={"sequence": "MKTLL123"})
    assert r.status_code == 422


def test_embed_missing_sequence_returns_422(client):
    r = client.post("/embed", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /embed/batch
# ---------------------------------------------------------------------------

def test_batch_embed_returns_correct_count(client):
    seqs = [HEMOGLOBIN_FRAGMENT, SYNUCLEIN_FRAGMENT, "MKTLL"]
    r = client.post("/embed/batch", json={"sequences": seqs})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["embeddings"]) == 3


def test_batch_embed_each_vector_is_320_dim(client):
    r = client.post("/embed/batch", json={"sequences": ["MKTLL", "MVLSP"]})
    assert r.status_code == 200
    for vec in r.json()["embeddings"]:
        assert len(vec) == 320


def test_batch_embed_empty_list_returns_422(client):
    r = client.post("/embed/batch", json={"sequences": []})
    assert r.status_code == 422


def test_batch_embed_over_limit_returns_422(client):
    r = client.post("/embed/batch", json={"sequences": ["MKTLL"] * 65})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------

def test_search_returns_k_results(client):
    r = client.post("/search", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 3})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3


def test_search_results_sorted_by_similarity(client):
    r = client.post("/search", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 5})
    similarities = [res["similarity"] for res in r.json()["results"]]
    assert similarities == sorted(similarities, reverse=True)


def test_search_hemoglobin_fragment_finds_oxygen_carrier(client):
    r = client.post("/search", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 1})
    top = r.json()["results"][0]
    assert top["uniprot_id"] in ("P69905", "P68871", "P02144")


def test_search_invalid_sequence_returns_422(client):
    r = client.post("/search", json={"sequence": "NOT-A-PROTEIN", "k": 3})
    assert r.status_code == 422


def test_search_k_out_of_range_returns_422(client):
    r = client.post("/search", json={"sequence": "MKTLL", "k": 99})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /pipeline/annotate
# ---------------------------------------------------------------------------

def test_annotate_returns_predicted_go_terms(client):
    r = client.post("/pipeline/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predicted_go_terms"]) > 0


def test_annotate_go_terms_sorted_by_confidence(client):
    r = client.post("/pipeline/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 5})
    confidences = [t["confidence"] for t in r.json()["predicted_go_terms"]]
    assert confidences == sorted(confidences, reverse=True)


def test_annotate_confidence_bounded(client):
    r = client.post("/pipeline/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 5})
    for term in r.json()["predicted_go_terms"]:
        assert 0.0 <= term["confidence"] <= 1.0


def test_annotate_includes_inference_time(client):
    r = client.post("/pipeline/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT})
    assert r.status_code == 200
    assert "inference_time_ms" in r.json()
    assert r.json()["inference_time_ms"] > 0


def test_annotate_hemoglobin_predicts_heme_binding(client):
    r = client.post("/pipeline/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 5})
    go_ids = {t["id"] for t in r.json()["predicted_go_terms"]}
    assert "GO:0020037" in go_ids  # heme binding


def test_annotate_includes_nearest_neighbors(client):
    r = client.post("/pipeline/annotate", json={"sequence": HEMOGLOBIN_FRAGMENT, "k": 3})
    body = r.json()
    assert len(body["nearest_neighbors"]) == 3


def test_annotate_invalid_sequence_returns_422(client):
    r = client.post("/pipeline/annotate", json={"sequence": "BADSEQ123"})
    assert r.status_code == 422
