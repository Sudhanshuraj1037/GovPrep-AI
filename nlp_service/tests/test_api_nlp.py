import pytest
from fastapi.testclient import TestClient

from app import data as D
from app import tfidf as T
from app.main import app

client = TestClient(app)


@pytest.fixture
def trained(toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    T.train_all(seed=42)


# --------------------------------------------------------- not trained yet
def test_classify_503_when_not_trained(env):
    r = client.post("/classify/tfidf", json={"texts": ["river delta"]})
    assert r.status_code == 503
    d = r.json()["detail"]
    assert d["status"] == "not_trained_yet" and d["model"] == "tfidf"
    assert "predictions" not in str(r.json())  # no placeholder predictions


def test_metrics_404_when_not_trained(env):
    r = client.get("/models/tfidf/metrics")
    assert r.status_code == 404
    assert r.json()["detail"]["status"] == "not_trained_yet"


def test_health_says_not_trained(env):
    body = client.get("/health").json()
    tf = next(m for m in body["models"] if m["name"] == "tfidf")
    assert tf["trained"] is False and tf["targets_trained"] == []
    assert body["models_trained"] == 0


# ------------------------------------------------------------- preprocess
def test_preprocess_endpoint(env):
    r = client.post("/preprocess", json={"text": "Which is NOT correct? Empires fell."})
    assert r.status_code == 200
    b = r.json()
    assert "not" in b["final_tokens"] and "empir" in b["final_tokens"]
    r = client.post("/preprocess", json={"text": "Empires", "normalizer": "none"})
    assert r.json()["final_tokens"] == ["empires"]


def test_preprocess_validation(env):
    assert client.post("/preprocess", json={"text": ""}).status_code == 422
    assert client.post("/preprocess", json={"text": "x", "normalizer": "bogus"}).status_code == 422


# ---------------------------------------------------------------- trained
def test_classify_after_training(trained):
    r = client.post("/classify/tfidf", json={"texts": ["river delta monsoon", "empire dynasty"], "top_k": 2})
    assert r.status_code == 200
    b = r.json()
    assert b["model"] == "tfidf" and b["trained_at"]
    assert set(b["targets_available"]) == {"subject", "topic", "difficulty"}
    assert b["results"][0]["predictions"]["subject"]["label"] == "Geography"
    assert b["results"][1]["predictions"]["subject"]["label"] == "History"
    assert len(b["results"][0]["predictions"]["subject"]["top"]) == 2
    assert b["results"][0]["text_preview"].startswith("river")


def test_metrics_endpoint_returns_real_file(trained):
    b = client.get("/models/tfidf/metrics").json()
    assert b["model"] == "tfidf" and b["seed"] == 42
    assert "confusion_matrix" in b["targets"]["subject"]["splits"]["test"]


def test_health_after_training(trained):
    body = client.get("/health").json()
    tf = next(m for m in body["models"] if m["name"] == "tfidf")
    assert tf["trained"] is True
    assert tf["targets_trained"] == ["difficulty", "subject", "topic"]
    assert body["models_trained"] == 1
    assert body["datasets"]["public_train"]["present"] is True
    assert body["datasets"]["pdf_heldout"]["present"] is False


def test_model_cache_picks_up_retraining(trained):
    first = client.post("/classify/tfidf", json={"texts": ["river"]}).json()["trained_at"]
    T.train_all(targets=("subject",), seed=42)
    second = client.post("/classify/tfidf", json={"texts": ["river"]}).json()
    assert second["trained_at"] != first
    assert second["targets_available"] == ["subject"]


def test_classify_validation(trained):
    assert client.post("/classify/tfidf", json={"texts": []}).status_code == 422
    assert client.post("/classify/tfidf", json={"texts": [""]}).status_code == 422
    assert client.post("/classify/tfidf", json={"texts": ["x"] * 33}).status_code == 422
    assert client.post("/classify/tfidf", json={"texts": ["x"], "top_k": 0}).status_code == 422
    assert client.post("/classify/tfidf", json={"text": "x"}).status_code == 422


def test_classify_unusable_text(trained):
    b = client.post("/classify/tfidf", json={"texts": ["the of and"]}).json()
    assert b["results"][0]["predictions"]["subject"]["label"] is None
