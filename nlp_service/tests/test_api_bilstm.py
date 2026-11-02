"""API tests for the BiLSTM endpoints (equivalent to the LSTM ones). Synthetic data only.

The "not trained" tests need no PyTorch at all. The rest use the session-trained tiny
BiLSTM and are skipped when PyTorch is not installed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import api_bilstm
from app.config import settings
from app.main import app

client = TestClient(app)
NLP_ROOT = Path(__file__).resolve().parent.parent


def _dir(name="bilstm"):
    return Path(settings.artifacts_dir) / name


def _health(name="bilstm"):
    return next(m for m in client.get("/health").json()["models"] if m["name"] == name)


# ======================================================== not trained yet
def test_status_reports_not_trained_with_200(env):
    r = client.get("/models/bilstm/status")
    assert r.status_code == 200
    b = r.json()
    assert b["trained"] is False and b["status"] == "not_trained_yet" and b["targets_trained"] == []
    assert b["model"] == "bilstm" and isinstance(b["torch_installed"], bool)
    assert "train_bilstm" in b["message"]


def test_metrics_404_when_not_trained(env):
    r = client.get("/models/bilstm/metrics")
    assert r.status_code == 404
    d = r.json()["detail"]
    assert d["status"] == "not_trained_yet" and d["model"] == "bilstm" and "BiLSTM" in d["message"]


def test_classify_503_when_not_trained(env):
    r = client.post("/classify/bilstm", json={"texts": ["river delta"]})
    assert r.status_code == 503
    d = r.json()["detail"]
    assert d["status"] == "not_trained_yet" and d["model"] == "bilstm"
    assert "predictions" not in r.text                                  # no placeholder output


def test_health_bilstm_not_trained(env):
    h = _health()
    assert h["trained"] is False and h["targets_trained"] == []
    assert client.get("/health").json()["models_trained"] == 0


def test_openapi_has_both_endpoint_sets_with_unique_operation_ids():
    schema = app.openapi()
    for name in ("lstm", "bilstm"):
        assert f"/models/{name}/status" in schema["paths"] and f"/models/{name}/metrics" in schema["paths"]
        assert "post" in schema["paths"][f"/classify/{name}"]
    ids = [op["operationId"] for ops in schema["paths"].values() for op in ops.values()]
    assert len(ids) == len(set(ids))


def test_service_runs_without_pytorch(env, _bilstm_baseline):
    """The whole service (and /health) must start with PyTorch missing; only classification needs it."""
    root, _ = _bilstm_baseline
    code = (
        "import importlib.machinery as m\n"
        "orig = m.PathFinder.find_spec\n"
        "m.PathFinder.find_spec = staticmethod(lambda name, path=None, target=None: None\n"
        "    if name == 'torch' or name.startswith('torch.') else orig(name, path, target))\n"
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "c = TestClient(app)\n"
        "h = c.get('/health'); st = c.get('/models/bilstm/status')\n"
        "cl = c.post('/classify/bilstm', json={'texts': ['river delta']})\n"
        "import json; print(json.dumps({'health': h.status_code, 'torch_dep': h.json()['dependencies']['torch']['installed'],"
        " 'status': st.status_code, 'trained': st.json()['trained'], 'inference': st.json()['inference_available'],"
        " 'classify': cl.status_code, 'classify_status': cl.json()['detail']['status']}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=NLP_ROOT, capture_output=True, text=True,
        env={**os.environ, "NLP_DATA_DIR": str(root / "data"), "NLP_ARTIFACTS_DIR": str(root / "artifacts"),
             "PYTHONPATH": str(NLP_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {
        "health": 200, "torch_dep": False, "status": 200, "trained": True, "inference": False,
        "classify": 503, "classify_status": "dependency_missing"}


# ============================================================ trained
def test_status_when_trained(bilstm_trained):
    b = client.get("/models/bilstm/status").json()
    assert b["trained"] is True and b["status"] == "trained" and b["inference_available"] is True
    assert b["model"] == "bilstm" and b["targets_trained"] == ["difficulty", "subject", "topic"] and b["seed"] == 42
    sub, ref = b["targets"]["subject"], bilstm_trained["targets"]["subject"]
    assert sub["classes"] == len(ref["labels"]) and sub["n_train"] == ref["n_train"]
    assert sub["best_epoch"] == ref["training"]["best_epoch"]
    assert sub["trainable_parameters"] == ref["model"]["trainable_parameters"]
    assert sub["scores"]["test"] == {
        "n": ref["splits"]["test"]["n"], "accuracy": ref["splits"]["test"]["accuracy"],
        "f1_macro": ref["splits"]["test"]["f1_macro"], "f1_weighted": ref["splits"]["test"]["f1_weighted"]}
    assert b["hyperparameters"]["hidden_dim"] == 16 and b["embeddings"]["init"] == "random"
    assert b["framework"]["device"] == "cpu" and b["pdf_heldout_evaluated"] is False


def test_metrics_endpoint_returns_the_real_file(bilstm_trained):
    b = client.get("/models/bilstm/metrics").json()
    on_disk = json.loads((_dir() / "metrics.json").read_text(encoding="utf-8"))
    assert b.pop("weights_available") is True
    assert b == on_disk and b["model"] == "bilstm"
    assert "confusion_matrix" in b["targets"]["subject"]["splits"]["test"]


def test_classify_endpoint(bilstm_trained):
    r = client.post("/classify/bilstm", json={"texts": ["river delta monsoon", "ancient empire dynasty"], "top_k": 2})
    assert r.status_code == 200
    b = r.json()
    assert b["model"] == "bilstm" and b["trained_at"] and b["device"] == "cpu"
    assert set(b["targets_available"]) == {"subject", "topic", "difficulty"} and b["targets_unavailable"] == {}
    assert b["results"][0]["predictions"]["subject"]["label"] == "Geography"
    assert b["results"][1]["predictions"]["subject"]["label"] == "History"
    assert len(b["results"][0]["predictions"]["subject"]["top"]) == 2
    assert b["results"][0]["text_preview"].startswith("river")


def test_api_equals_library_predictions(bilstm_trained):
    from app import bilstm as B

    texts = ["river delta monsoon", "atom photon energy", "the constitution article"]
    api = client.post("/classify/bilstm", json={"texts": texts, "top_k": 3}).json()["results"]
    lib = B.BiLSTMService.load().predict(texts, top_k=3)
    for a, b in zip(api, lib):
        assert a["predictions"] == b["predictions"]


def test_classify_unreadable_text(bilstm_trained):
    b = client.post("/classify/bilstm", json={"texts": ["zzz qqq", "river delta"]}).json()
    p = b["results"][0]["predictions"]["subject"]
    assert p["label"] is None and "vocabulary" in p["warning"]
    assert b["results"][1]["predictions"]["subject"]["label"] == "Geography"


def test_classify_input_validation(bilstm_trained):
    post = lambda **j: client.post("/classify/bilstm", json=j)  # noqa: E731
    assert post(texts=[]).status_code == 422
    assert post(texts=[""]).status_code == 422
    assert post(texts=["x"] * 33).status_code == 422
    assert post(texts=["x" * 5001]).status_code == 422
    assert post(texts=["x"], top_k=0).status_code == 422
    assert post(texts=["x"], top_k=11).status_code == 422
    assert post(text="x").status_code == 422


# ============================================================ health
def test_health_bilstm_trained(bilstm_trained):
    h = _health()
    assert h["trained"] is True and h["targets_trained"] == ["difficulty", "subject", "topic"]
    assert client.get("/health").json()["models_trained"] == 1
    assert _health("lstm")["trained"] is False                          # independent of the LSTM


def test_health_not_trained_with_metrics_but_no_checkpoint(bilstm_trained):
    for f in _dir().glob("*.pt"):
        f.unlink()
    assert _health()["trained"] is False and _health()["targets_trained"] == []


def test_health_not_trained_with_checkpoint_but_no_metrics(bilstm_trained):
    (_dir() / "metrics.json").unlink()
    assert _health()["trained"] is False


def test_metrics_only_state_is_reported_honestly(bilstm_trained):
    for f in _dir().glob("*.pt"):
        f.unlink()
    st = client.get("/models/bilstm/status").json()
    assert st["trained"] is False and st["metrics_available"] is True and "not committed to git" in st["message"]
    m = client.get("/models/bilstm/metrics")
    assert m.status_code == 200 and m.json()["weights_available"] is False
    r = client.post("/classify/bilstm", json={"texts": ["river"]})
    assert r.status_code == 503 and r.json()["detail"]["status"] == "not_trained_yet"


# ============================================================ both models side by side
def test_lstm_and_bilstm_endpoints_are_independent(lstm_trained, _bilstm_baseline):
    """Only the LSTM is trained: its endpoints work, the BiLSTM's say 'not trained yet'."""
    assert client.post("/classify/lstm", json={"texts": ["river delta"]}).status_code == 200
    r = client.post("/classify/bilstm", json={"texts": ["river delta"]})
    assert r.status_code == 503 and r.json()["detail"]["model"] == "bilstm"
    assert client.get("/models/bilstm/status").json()["trained"] is False
    assert client.get("/models/lstm/status").json()["trained"] is True
    # now add the BiLSTM artifacts: both answer, each labelled with its own model name
    import shutil
    root, _ = _bilstm_baseline
    shutil.copytree(root / "artifacts" / "bilstm", _dir())
    a = client.post("/classify/lstm", json={"texts": ["river delta"]}).json()
    b = client.post("/classify/bilstm", json={"texts": ["river delta"]}).json()
    assert a["model"] == "lstm" and b["model"] == "bilstm"
    assert a["trained_at"] != b["trained_at"]


# ============================================================ robustness
def test_missing_dependency_gives_503_not_500(bilstm_trained, monkeypatch):
    def no_torch():
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(api_bilstm, "_import_bilstm", no_torch)
    r = client.post("/classify/bilstm", json={"texts": ["river"]})
    assert r.status_code == 503 and r.json()["detail"]["status"] == "dependency_missing"
    assert client.get("/models/bilstm/status").status_code == 200       # status never needs torch
    assert client.get("/models/bilstm/metrics").status_code == 200


def test_corrupt_checkpoint_gives_503_not_500(bilstm_trained):
    (_dir() / "subject.pt").write_bytes(b"garbage")
    r = client.post("/classify/bilstm", json={"texts": ["river"]})
    assert r.status_code == 503 and "could not load checkpoint" in r.json()["detail"]["message"]


def test_an_lstm_checkpoint_in_the_bilstm_folder_gives_503(lstm_trained):
    import shutil
    shutil.copytree(_dir("lstm"), _dir("bilstm"))                        # LSTM files filed under bilstm/
    r = client.post("/classify/bilstm", json={"texts": ["river"]})
    assert r.status_code == 503 and "expected 'bilstm'" in r.json()["detail"]["message"]


def test_model_cache_picks_up_retraining(bilstm_trained):
    from app import bilstm as B
    from tests.conftest import TINY_LSTM

    first = client.post("/classify/bilstm", json={"texts": ["river"]}).json()
    B.train_all(cfg=B.BiLSTMConfig(**{**TINY_LSTM, "epochs": 2}), targets=("subject",))
    second = client.post("/classify/bilstm", json={"texts": ["river"]}).json()
    assert second["trained_at"] != first["trained_at"] and second["targets_available"] == ["subject"]
