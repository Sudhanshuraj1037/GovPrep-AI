"""API tests for the LSTM endpoints. Synthetic data only (see conftest).

The "not trained" tests need no PyTorch at all. The rest use the session-trained
tiny LSTM and are skipped when PyTorch is not installed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import api_lstm
from app.config import settings
from app.main import app

client = TestClient(app)
NLP_ROOT = Path(__file__).resolve().parent.parent


def _lstm_dir():
    return Path(settings.artifacts_dir) / "lstm"


def _health_lstm():
    return next(m for m in client.get("/health").json()["models"] if m["name"] == "lstm")


# ======================================================== not trained yet
def test_status_reports_not_trained_with_200(env):
    r = client.get("/models/lstm/status")
    assert r.status_code == 200
    b = r.json()
    assert b["trained"] is False and b["status"] == "not_trained_yet" and b["targets_trained"] == []
    assert isinstance(b["torch_installed"], bool)


def test_metrics_404_when_not_trained(env):
    r = client.get("/models/lstm/metrics")
    assert r.status_code == 404 and r.json()["detail"]["status"] == "not_trained_yet"


def test_classify_503_when_not_trained(env):
    r = client.post("/classify/lstm", json={"texts": ["river delta"]})
    assert r.status_code == 503
    d = r.json()["detail"]
    assert d["status"] == "not_trained_yet" and d["model"] == "lstm"
    assert "predictions" not in r.text                                  # no placeholder output


def test_health_lstm_not_trained(env):
    h = _health_lstm()
    assert h["trained"] is False and h["targets_trained"] == []
    assert client.get("/health").json()["models_trained"] == 0


def test_service_runs_without_pytorch(env, _lstm_baseline):
    """The whole service (and /health) must start with PyTorch missing; only classification needs it."""
    root, _ = _lstm_baseline
    code = (
        "import importlib.machinery as m\n"               # make torch genuinely 'not installed':
        "orig = m.PathFinder.find_spec\n"                 # no finder can locate it, and it is never
        "m.PathFinder.find_spec = staticmethod(lambda name, path=None, target=None: None\n"  # placed in sys.modules
        "    if name == 'torch' or name.startswith('torch.') else orig(name, path, target))\n"
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "c = TestClient(app)\n"
        "h = c.get('/health'); st = c.get('/models/lstm/status')\n"
        "cl = c.post('/classify/lstm', json={'texts': ['river delta']})\n"
        "lstm = [m for m in h.json()['models'] if m['name'] == 'lstm'][0]\n"
        "import json; print(json.dumps({'health': h.status_code, 'torch_dep': h.json()['dependencies']['torch']['installed'],"
        " 'status': st.status_code, 'trained': st.json()['trained'], 'torch_installed': st.json()['torch_installed'],"
        " 'inference': st.json()['inference_available'], 'classify': cl.status_code,"
        " 'classify_status': cl.json()['detail']['status']}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=NLP_ROOT, capture_output=True, text=True,
        env={**os.environ, "NLP_DATA_DIR": str(root / "data"), "NLP_ARTIFACTS_DIR": str(root / "artifacts"),
             "PYTHONPATH": str(NLP_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"health": 200, "torch_dep": False, "status": 200, "trained": True, "torch_installed": False,
                   "inference": False, "classify": 503, "classify_status": "dependency_missing"}


# ============================================================ trained
def test_status_when_trained(lstm_trained):
    b = client.get("/models/lstm/status").json()
    assert b["trained"] is True and b["status"] == "trained" and b["inference_available"] is True
    assert b["targets_trained"] == ["difficulty", "subject", "topic"] and b["seed"] == 42
    sub = b["targets"]["subject"]
    ref = lstm_trained["targets"]["subject"]
    assert sub["classes"] == len(ref["labels"]) and sub["n_train"] == ref["n_train"]
    assert sub["best_epoch"] == ref["training"]["best_epoch"]
    assert sub["scores"]["test"] == {
        "n": ref["splits"]["test"]["n"],
        "accuracy": ref["splits"]["test"]["accuracy"],
        "f1_macro": ref["splits"]["test"]["f1_macro"],
        "f1_weighted": ref["splits"]["test"]["f1_weighted"],
    }
    assert b["hyperparameters"]["hidden_dim"] == 16 and b["embeddings"]["init"] == "random"
    assert b["framework"]["device"] == "cpu" and b["pdf_heldout_evaluated"] is False


def test_metrics_endpoint_returns_the_real_file(lstm_trained):
    b = client.get("/models/lstm/metrics").json()
    on_disk = json.loads((_lstm_dir() / "metrics.json").read_text(encoding="utf-8"))
    assert b.pop("weights_available") is True
    assert b == on_disk
    assert "confusion_matrix" in b["targets"]["subject"]["splits"]["test"]


def test_classify_endpoint(lstm_trained):
    r = client.post("/classify/lstm", json={"texts": ["river delta monsoon", "ancient empire dynasty"], "top_k": 2})
    assert r.status_code == 200
    b = r.json()
    assert b["model"] == "lstm" and b["trained_at"] and b["device"] == "cpu"
    assert set(b["targets_available"]) == {"subject", "topic", "difficulty"} and b["targets_unavailable"] == {}
    assert b["results"][0]["predictions"]["subject"]["label"] == "Geography"
    assert b["results"][1]["predictions"]["subject"]["label"] == "History"
    assert len(b["results"][0]["predictions"]["subject"]["top"]) == 2
    assert b["results"][0]["text_preview"].startswith("river")


def test_api_equals_library_predictions(lstm_trained):
    from app import lstm as L

    texts = ["river delta monsoon", "atom photon energy", "the constitution article"]
    api = client.post("/classify/lstm", json={"texts": texts, "top_k": 3}).json()["results"]
    lib = L.LSTMService.load().predict(texts, top_k=3)
    for a, b in zip(api, lib):
        assert a["predictions"] == b["predictions"]


def test_classify_unreadable_text(lstm_trained):
    b = client.post("/classify/lstm", json={"texts": ["zzz qqq", "river delta"]}).json()
    p = b["results"][0]["predictions"]["subject"]
    assert p["label"] is None and "vocabulary" in p["warning"]
    assert b["results"][1]["predictions"]["subject"]["label"] == "Geography"


def test_classify_input_validation(lstm_trained):
    post = lambda **j: client.post("/classify/lstm", json=j)  # noqa: E731
    assert post(texts=[]).status_code == 422
    assert post(texts=[""]).status_code == 422
    assert post(texts=["x"] * 33).status_code == 422
    assert post(texts=["x" * 5001]).status_code == 422
    assert post(texts=["x"], top_k=0).status_code == 422
    assert post(texts=["x"], top_k=11).status_code == 422
    assert post(text="x").status_code == 422


# ============================================================ health
def test_health_lstm_trained(lstm_trained):
    h = _health_lstm()
    assert h["trained"] is True and h["targets_trained"] == ["difficulty", "subject", "topic"]
    assert client.get("/health").json()["models_trained"] == 1


def test_health_not_trained_with_metrics_but_no_checkpoint(lstm_trained):
    for f in _lstm_dir().glob("*.pt"):
        f.unlink()
    assert _health_lstm()["trained"] is False and _health_lstm()["targets_trained"] == []


def test_health_not_trained_with_checkpoint_but_no_metrics(lstm_trained):
    (_lstm_dir() / "metrics.json").unlink()
    assert _health_lstm()["trained"] is False


def test_metrics_only_state_is_reported_honestly(lstm_trained):
    for f in _lstm_dir().glob("*.pt"):
        f.unlink()
    st = client.get("/models/lstm/status").json()
    assert st["trained"] is False and st["metrics_available"] is True and "not committed to git" in st["message"]
    m = client.get("/models/lstm/metrics")
    assert m.status_code == 200 and m.json()["weights_available"] is False       # the past run's metrics are still real
    r = client.post("/classify/lstm", json={"texts": ["river"]})
    assert r.status_code == 503 and r.json()["detail"]["status"] == "not_trained_yet"


# ============================================================ robustness
def test_missing_dependency_gives_503_not_500(lstm_trained, monkeypatch):
    def no_torch():
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(api_lstm, "_import_lstm", no_torch)
    r = client.post("/classify/lstm", json={"texts": ["river"]})
    assert r.status_code == 503 and r.json()["detail"]["status"] == "dependency_missing"
    assert client.get("/models/lstm/status").status_code == 200       # status never needs torch
    assert client.get("/models/lstm/metrics").status_code == 200


def test_corrupt_checkpoint_gives_503_not_500(lstm_trained):
    (_lstm_dir() / "subject.pt").write_bytes(b"garbage")
    r = client.post("/classify/lstm", json={"texts": ["river"]})
    assert r.status_code == 503 and "could not load checkpoint" in r.json()["detail"]["message"]


def test_model_cache_picks_up_retraining(lstm_trained, toy_csv=None):
    from app import lstm as L
    from tests.conftest import TINY_LSTM

    first = client.post("/classify/lstm", json={"texts": ["river"]}).json()
    L.train_all(cfg=L.LSTMConfig(**{**TINY_LSTM, "epochs": 2}), targets=("subject",))
    second = client.post("/classify/lstm", json={"texts": ["river"]}).json()
    assert second["trained_at"] != first["trained_at"] and second["targets_available"] == ["subject"]
