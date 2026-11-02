"""Phase 1 tests: /health works and never claims a model is trained unless
its marker file really exists."""
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


def test_health_ok_and_shape():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "govprep-nlp-service"
    assert set(body) >= {"models", "datasets", "dependencies", "labels", "python"}
    names = [m["name"] for m in body["models"]]
    assert names == ["tfidf", "word2vec", "lstm", "bilstm", "bert"]


def test_labels_match_existing_app():
    labels = client.get("/health").json()["labels"]
    assert "History" in labels["subjects"] and "Polity" in labels["subjects"]
    assert labels["difficulties"] == ["easy", "medium", "hard"]


def test_fresh_checkout_reports_nothing_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifacts_dir", tmp_path)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    body = client.get("/health").json()
    assert body["models_trained"] == 0
    assert all(m["trained"] is False for m in body["models"])
    assert all(d["present"] is False for d in body["datasets"].values())


def test_trained_flag_needs_marker_and_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifacts_dir", tmp_path)
    (tmp_path / "tfidf").mkdir()
    # metrics.json alone (e.g. after a git clone without weights) is NOT "trained"
    (tmp_path / "tfidf" / "metrics.json").write_text("{}")
    by_name = {m["name"]: m["trained"] for m in client.get("/health").json()["models"]}
    assert by_name["tfidf"] is False
    # marker + weights file => trained
    (tmp_path / "tfidf" / "subject.joblib").write_bytes(b"x")
    body = client.get("/health").json()
    by_name = {m["name"]: m["trained"] for m in body["models"]}
    assert by_name["tfidf"] is True
    assert by_name["bert"] is False
    assert body["models_trained"] == 1


# ---------------------------------------------------------------- every model, same rule
import pytest  # noqa: E402

from app.registry import MODELS  # noqa: E402


@pytest.mark.parametrize("spec", MODELS, ids=[m.name for m in MODELS])
def test_no_model_is_trained_without_both_marker_and_weights(tmp_path, monkeypatch, spec):
    """For EVERY model: marker alone -> not trained; weights alone -> not trained; both -> trained."""
    assert spec.weights_suffix, f"{spec.name} must declare its weights file type"
    monkeypatch.setattr(settings, "artifacts_dir", tmp_path)
    d = tmp_path / spec.name
    d.mkdir()
    status = lambda: next(m for m in client.get("/health").json()["models"] if m["name"] == spec.name)  # noqa: E731
    assert status()["trained"] is False
    (d / spec.marker).write_text("{}")
    assert status()["trained"] is False
    (d / spec.marker).unlink()
    (d / f"x{spec.weights_suffix}").write_bytes(b"x")
    assert status()["trained"] is False
    (d / spec.marker).write_text("{}")
    assert status()["trained"] is True
    assert client.get("/health").json()["models_trained"] == 1
