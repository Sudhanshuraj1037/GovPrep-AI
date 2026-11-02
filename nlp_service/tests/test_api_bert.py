"""API tests for the BERT endpoints (equivalent to the LSTM / BiLSTM ones). Synthetic data and a
TINY locally built stand-in for a pretrained BERT only (see test_bert.py): no real BERT is used.

The "not trained" and "dependency" tests need neither PyTorch nor Transformers; the rest use the
session fine-tuned tiny BERT and are skipped when either library is missing.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import api_bert
from app.config import settings
from app.main import app

client = TestClient(app)
NLP_ROOT = Path(__file__).resolve().parent.parent

# Run a snippet in a fresh interpreter in which one package is genuinely "not installed"
# (no finder can locate it and it is never placed in sys.modules).
HIDE = (
    "import importlib.machinery as m\n"
    "orig = m.PathFinder.find_spec\n"
    "HIDDEN = {hidden!r}\n"
    "m.PathFinder.find_spec = staticmethod(lambda name, path=None, target=None: None\n"
    "    if name.split('.')[0] in HIDDEN else orig(name, path, target))\n"
)


def _dir(name="bert"):
    return Path(settings.artifacts_dir) / name


def _health(name="bert"):
    return next(m for m in client.get("/health").json()["models"] if m["name"] == name)


def _run(code, root, hidden=()):
    proc = subprocess.run(
        [sys.executable, "-c", HIDE.format(hidden=set(hidden)) + code], cwd=NLP_ROOT, capture_output=True, text=True,
        env={**os.environ, "NLP_DATA_DIR": str(root / "data"), "NLP_ARTIFACTS_DIR": str(root / "artifacts"),
             "PYTHONPATH": str(NLP_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


PROBE = (
    "from fastapi.testclient import TestClient\n"
    "from app.main import app\n"
    "c = TestClient(app)\n"
    "h = c.get('/health'); st = c.get('/models/bert/status')\n"
    "cl = c.post('/classify/bert', json={'texts': ['river delta']})\n"
    "import json; print(json.dumps({'health': h.status_code, 'torch_dep': h.json()['dependencies']['torch']['installed'],"
    " 'tf_dep': h.json()['dependencies']['transformers']['installed'], 'status': st.status_code,"
    " 'trained': st.json()['trained'], 'torch': st.json()['torch_installed'],"
    " 'transformers': st.json()['transformers_installed'], 'inference': st.json()['inference_available'],"
    " 'classify': cl.status_code, 'classify_status': cl.json()['detail']['status'],"
    " 'message': cl.json()['detail']['message']}))\n"
)


# ======================================================== not trained yet
def test_status_reports_not_trained_with_200(env):
    r = client.get("/models/bert/status")
    assert r.status_code == 200
    b = r.json()
    assert b["trained"] is False and b["status"] == "not_trained_yet" and b["targets_trained"] == []
    assert b["model"] == "bert" and isinstance(b["torch_installed"], bool) and isinstance(b["transformers_installed"], bool)
    assert "train_bert" in b["message"]


def test_metrics_404_when_not_trained(env):
    r = client.get("/models/bert/metrics")
    assert r.status_code == 404
    d = r.json()["detail"]
    assert d["status"] == "not_trained_yet" and d["model"] == "bert" and "BERT" in d["message"]


def test_classify_503_when_not_trained(env):
    r = client.post("/classify/bert", json={"texts": ["river delta"]})
    assert r.status_code == 503
    d = r.json()["detail"]
    assert d["status"] == "not_trained_yet" and d["model"] == "bert"
    assert "predictions" not in r.text                                       # no placeholder output


def test_health_bert_not_trained(env):
    h = _health()
    assert h["trained"] is False and h["targets_trained"] == []
    assert client.get("/health").json()["models_trained"] == 0


def test_openapi_has_all_three_endpoint_sets_with_unique_operation_ids():
    schema = app.openapi()
    for name in ("lstm", "bilstm", "bert"):
        assert f"/models/{name}/status" in schema["paths"] and f"/models/{name}/metrics" in schema["paths"]
        assert "post" in schema["paths"][f"/classify/{name}"]
    ids = [op["operationId"] for ops in schema["paths"].values() for op in ops.values()]
    assert len(ids) == len(set(ids))


# ============================================================ lazy imports / missing dependencies
def test_starting_the_service_imports_neither_torch_nor_transformers(env):
    out = _run("import sys\nfrom app.main import app\nimport json\n"
               "print(json.dumps({'torch': 'torch' in sys.modules, 'transformers': 'transformers' in sys.modules}))\n",
               env)
    assert out == {"torch": False, "transformers": False}


def test_service_runs_without_transformers(_bert_baseline):
    """Everything except classification must work when Transformers is missing."""
    root, _ = _bert_baseline
    out = _run(PROBE, root, hidden=("transformers",))
    assert out["health"] == 200 and out["tf_dep"] is False and out["torch_dep"] is True
    assert out["status"] == 200 and out["trained"] is True
    assert out["transformers"] is False and out["torch"] is True and out["inference"] is False
    assert out["classify"] == 503 and out["classify_status"] == "dependency_missing"
    assert "Hugging Face Transformers is not installed" in out["message"]


def test_service_runs_without_pytorch(_bert_baseline):
    root, _ = _bert_baseline
    out = _run(PROBE, root, hidden=("torch",))
    assert out["health"] == 200 and out["torch_dep"] is False and out["status"] == 200
    assert out["inference"] is False and out["classify"] == 503 and out["classify_status"] == "dependency_missing"
    assert "PyTorch is not installed" in out["message"]


def test_cli_without_transformers_exits_with_a_clear_message(env):
    code = HIDE.format(hidden={"transformers"}) + "import sys; sys.argv = ['train_bert']\nfrom training.train_bert import main\nraise SystemExit(main([]))\n"
    proc = subprocess.run([sys.executable, "-c", code], cwd=NLP_ROOT, capture_output=True, text=True,
                          env={**os.environ, "NLP_DATA_DIR": str(env / "data"), "NLP_ARTIFACTS_DIR": str(env / "artifacts"),
                               "PYTHONPATH": str(NLP_ROOT)})
    assert proc.returncode == 3 and "PyTorch or Transformers is not installed" in proc.stderr


# ============================================================ trained
def test_status_when_trained(bert_trained):
    b = client.get("/models/bert/status").json()
    assert b["trained"] is True and b["status"] == "trained" and b["inference_available"] is True
    assert b["transformers_installed"] is True and b["torch_installed"] is True
    assert b["model"] == "bert" and b["targets_trained"] == ["difficulty", "subject", "topic"] and b["seed"] == 42
    sub, ref = b["targets"]["subject"], bert_trained["targets"]["subject"]
    assert sub["classes"] == len(ref["labels"]) and sub["n_train"] == ref["n_train"]
    assert sub["best_epoch"] == ref["training"]["best_epoch"]
    assert sub["trainable_parameters"] == ref["model"]["trainable_parameters"]
    assert sub["scores"]["test"] == {
        "n": ref["splits"]["test"]["n"], "accuracy": ref["splits"]["test"]["accuracy"],
        "f1_macro": ref["splits"]["test"]["f1_macro"], "f1_weighted": ref["splits"]["test"]["f1_weighted"]}
    assert b["hyperparameters"]["pretrained"] == bert_trained["hyperparameters"]["pretrained"]
    assert b["framework"]["device"] == "cpu" and b["pdf_heldout_evaluated"] is False


def test_metrics_endpoint_returns_the_real_file(bert_trained):
    b = client.get("/models/bert/metrics").json()
    on_disk = json.loads((_dir() / "metrics.json").read_text(encoding="utf-8"))
    assert b.pop("weights_available") is True
    assert b == on_disk and b["model"] == "bert"
    assert "confusion_matrix" in b["targets"]["subject"]["splits"]["test"]


def test_classify_endpoint(bert_trained):
    r = client.post("/classify/bert", json={"texts": ["river delta monsoon", "ancient empire dynasty"], "top_k": 2})
    assert r.status_code == 200
    b = r.json()
    assert b["model"] == "bert" and b["trained_at"] and b["device"] == "cpu"
    assert set(b["targets_available"]) == {"subject", "topic", "difficulty"} and b["targets_unavailable"] == {}
    assert b["results"][0]["predictions"]["subject"]["label"] == "Geography"
    assert b["results"][1]["predictions"]["subject"]["label"] == "History"
    assert len(b["results"][0]["predictions"]["subject"]["top"]) == 2
    assert b["results"][0]["text_preview"].startswith("river")


def test_api_equals_library_predictions(bert_trained):
    from app import bert as BT

    texts = ["river delta monsoon", "atom photon energy", "the constitution article"]
    api = client.post("/classify/bert", json={"texts": texts, "top_k": 3}).json()["results"]
    lib = BT.BertService.load().predict(texts, top_k=3)
    for a, b in zip(api, lib):
        assert a["predictions"] == b["predictions"]


def test_classify_unreadable_text(bert_trained):
    b = client.post("/classify/bert", json={"texts": ["zzz qqq", "river delta"]}).json()
    p = b["results"][0]["predictions"]["subject"]
    assert p["label"] is None and "unknown to the tokenizer" in p["warning"]
    assert b["results"][1]["predictions"]["subject"]["label"] == "Geography"


def test_classify_input_validation(bert_trained):
    post = lambda **j: client.post("/classify/bert", json=j)  # noqa: E731
    assert post(texts=[]).status_code == 422
    assert post(texts=[""]).status_code == 422
    assert post(texts=["x"] * 33).status_code == 422
    assert post(texts=["x" * 5001]).status_code == 422
    assert post(texts=["x"], top_k=0).status_code == 422
    assert post(texts=["x"], top_k=11).status_code == 422
    assert post(text="x").status_code == 422


# ============================================================ health
def test_health_bert_trained(bert_trained):
    h = _health()
    assert h["trained"] is True and h["targets_trained"] == ["difficulty", "subject", "topic"]
    assert client.get("/health").json()["models_trained"] == 1
    assert _health("lstm")["trained"] is False and _health("bilstm")["trained"] is False


def test_health_not_trained_with_metrics_but_no_checkpoint(bert_trained):
    for f in _dir().glob("*.pt"):
        f.unlink()
    assert _health()["trained"] is False and _health()["targets_trained"] == []


def test_health_not_trained_with_checkpoint_but_no_metrics(bert_trained):
    (_dir() / "metrics.json").unlink()
    assert _health()["trained"] is False


def test_metrics_only_state_is_reported_honestly(bert_trained):
    for f in _dir().glob("*.pt"):
        f.unlink()
    st = client.get("/models/bert/status").json()
    assert st["trained"] is False and st["metrics_available"] is True and "not committed to git" in st["message"]
    m = client.get("/models/bert/metrics")
    assert m.status_code == 200 and m.json()["weights_available"] is False
    r = client.post("/classify/bert", json={"texts": ["river"]})
    assert r.status_code == 503 and r.json()["detail"]["status"] == "not_trained_yet"


# ============================================================ models side by side
def test_bert_and_the_recurrent_models_are_independent(lstm_trained, _bert_baseline):
    """Only the LSTM is trained: BERT's endpoints say 'not trained yet'. Then add BERT's artifacts."""
    assert client.post("/classify/lstm", json={"texts": ["river delta"]}).status_code == 200
    r = client.post("/classify/bert", json={"texts": ["river delta"]})
    assert r.status_code == 503 and r.json()["detail"]["model"] == "bert"
    assert client.get("/models/bert/status").json()["trained"] is False
    root, _ = _bert_baseline
    shutil.copytree(root / "artifacts" / "bert", _dir())
    a = client.post("/classify/lstm", json={"texts": ["river delta"]}).json()
    b = client.post("/classify/bert", json={"texts": ["river delta"]}).json()
    assert a["model"] == "lstm" and b["model"] == "bert" and a["trained_at"] != b["trained_at"]


# ============================================================ robustness
def test_missing_dependency_gives_503_not_500(bert_trained, monkeypatch):
    def no_transformers():
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(api_bert, "_import_bert", no_transformers)
    r = client.post("/classify/bert", json={"texts": ["river"]})
    assert r.status_code == 503 and r.json()["detail"]["status"] == "dependency_missing"
    assert client.get("/models/bert/status").status_code == 200              # status never needs the libraries
    assert client.get("/models/bert/metrics").status_code == 200


def test_corrupt_checkpoint_gives_503_not_500(bert_trained):
    (_dir() / "subject.pt").write_bytes(b"garbage")
    r = client.post("/classify/bert", json={"texts": ["river"]})
    assert r.status_code == 503 and "could not load checkpoint" in r.json()["detail"]["message"]


def test_a_foreign_architecture_checkpoint_gives_503(bert_trained):
    import torch

    root = Path(settings.artifacts_dir)
    ck = torch.load(next((root / "bert").glob("subject.pt")), map_location="cpu", weights_only=True)
    ck["arch"] = "lstm"                                                      # a checkpoint that claims another architecture
    torch.save(ck, root / "bert" / "subject.pt")
    r = client.post("/classify/bert", json={"texts": ["river"]})
    assert r.status_code == 503 and "expected 'bert'" in r.json()["detail"]["message"]


def test_model_cache_picks_up_retraining(bert_trained, tiny_bert_dir):
    from app import bert as BT
    from tests.conftest import TINY_BERT

    first = client.post("/classify/bert", json={"texts": ["river"]}).json()
    BT.train_all(cfg=BT.BertFineTuneConfig(pretrained=str(tiny_bert_dir), **{**TINY_BERT, "epochs": 1}),
                 targets=("subject",))
    second = client.post("/classify/bert", json={"texts": ["river"]}).json()
    assert second["trained_at"] != first["trained_at"] and second["targets_available"] == ["subject"]
