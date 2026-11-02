"""API tests for GET /models/comparison and /models/comparison/rows.

Most use metrics files built from synthetic predictions by the real shared metrics function
(tests/metrics_factory.py) and need no PyTorch or Transformers; the last test uses the real
session-trained tiny models. Nothing here is a result about a real model.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import comparison as C
from app.config import settings
from app.main import app
from tests.metrics_factory import simple, write

client = TestClient(app)
NLP_ROOT = Path(__file__).resolve().parent.parent


def _three():
    write(simple("tfidf", accuracies={"val": 0.70, "test": 0.70}, n=300))
    write(simple("lstm", accuracies={"val": 0.80, "test": 0.80}, n=300))
    write(simple("bilstm", accuracies={"val": 0.92, "test": 0.92}, n=300))


# ============================================================ empty / single
def test_no_models_is_a_200_with_an_honest_status(env):
    r = client.get("/models/comparison")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "no_models_trained" and b["models_found"] == [] and b["targets"] == {}
    assert b["models_missing"] == ["tfidf", "lstm", "bilstm", "bert"] and b["warnings"]
    rows = client.get("/models/comparison/rows").json()
    assert rows["status"] == "no_models_trained" and rows["rows"] == []


def test_a_single_model_is_not_compared(env):
    write(simple("tfidf", accuracies={"val": 0.8, "test": 0.8}))
    b = client.get("/models/comparison").json()
    assert b["status"] == "single_model" and b["models_found"] == ["tfidf"]
    assert any("nothing to compare" in w for w in b["warnings"])


# ============================================================ with models
def test_comparison_equals_the_library_result(env):
    _three()
    api = client.get("/models/comparison", params={"n_bootstrap": 200}).json()
    assert api == json.loads(json.dumps(C.compare(n_bootstrap=200)))
    assert api["status"] == "ok" and api["models_found"] == ["tfidf", "lstm", "bilstm"]
    assert [e["model"] for e in api["targets"]["subject"]["test"]["models"]] == ["bilstm", "lstm", "tfidf"]
    assert api["summary"][0]["best_model"] == "bilstm" and api["parameters"]["n_bootstrap"] == 200


def test_default_and_zero_bootstrap(env):
    _three()
    assert client.get("/models/comparison").json()["parameters"]["n_bootstrap"] == 1000
    b = client.get("/models/comparison", params={"n_bootstrap": 0}).json()
    assert all(e["ci"] is None for e in b["targets"]["subject"]["test"]["models"])
    assert b["targets"]["subject"]["test"]["separated_from_runner_up"] is None


def test_responses_are_deterministic(env):
    _three()
    assert client.get("/models/comparison", params={"n_bootstrap": 150}).json() == \
        client.get("/models/comparison", params={"n_bootstrap": 150}).json()


def test_rows_endpoint_equals_the_comparison_rows(env):
    _three()
    b = client.get("/models/comparison/rows", params={"n_bootstrap": 100}).json()
    assert b["status"] == "ok" and b["models_found"] == ["tfidf", "lstm", "bilstm"] and len(b["rows"]) == 6
    assert b["rows"] == json.loads(json.dumps(C.comparison_rows(C.compare(n_bootstrap=100))))
    assert {"target", "split", "model", "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high"} <= set(b["rows"][0])


def test_warnings_reach_the_client(env):
    write(simple("tfidf", accuracies={"val": 0.8, "test": 0.8}, sha="A"))
    write(simple("lstm", accuracies={"val": 0.9, "test": 0.9}, sha="B"))
    b = client.get("/models/comparison", params={"n_bootstrap": 0}).json()
    assert any("DIFFERENT processed split files" in w for w in b["warnings"])
    assert client.get("/models/comparison/rows", params={"n_bootstrap": 0}).json()["warnings"] == b["warnings"]


def test_broken_metrics_are_reported_under_problems(env):
    _three()
    bad = json.loads((Path(settings.artifacts_dir) / "lstm" / "metrics.json").read_text())
    bad["targets"]["subject"]["splits"]["test"]["accuracy"] = 3
    write(bad)
    b = client.get("/models/comparison", params={"n_bootstrap": 0}).json()
    assert list(b["problems"]) == ["lstm"] and b["models_found"] == ["tfidf", "bilstm"]


@pytest.mark.parametrize("value", [-1, 5001, "abc"])
def test_bootstrap_validation(env, value):
    assert client.get("/models/comparison", params={"n_bootstrap": value}).status_code == 422
    assert client.get("/models/comparison/rows", params={"n_bootstrap": value}).status_code == 422


# ============================================================ no heavy dependencies
def test_the_endpoints_need_neither_torch_nor_transformers(env):
    _three()
    code = (
        "import importlib.machinery as m\n"
        "orig = m.PathFinder.find_spec\n"
        "m.PathFinder.find_spec = staticmethod(lambda name, path=None, target=None: None\n"
        "    if name.split('.')[0] in {'torch', 'transformers'} else orig(name, path, target))\n"
        "import sys, json\n"
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "c = TestClient(app)\n"
        "r = c.get('/models/comparison', params={'n_bootstrap': 50}); rows = c.get('/models/comparison/rows')\n"
        "print(json.dumps({'status': r.status_code, 'state': r.json()['status'], 'models': r.json()['models_found'],"
        " 'rows': rows.status_code, 'torch': 'torch' in sys.modules, 'transformers': 'transformers' in sys.modules}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=NLP_ROOT, capture_output=True, text=True,
                          env={**os.environ, "NLP_DATA_DIR": str(env / "data"),
                               "NLP_ARTIFACTS_DIR": str(settings.artifacts_dir), "PYTHONPATH": str(NLP_ROOT)})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {
        "status": 200, "state": "ok", "models": ["tfidf", "lstm", "bilstm"], "rows": 200,
        "torch": False, "transformers": False}


# ============================================================ the real (tiny, synthetic-data) models
def test_the_four_real_models_through_the_api(four_models_trained):
    b = client.get("/models/comparison", params={"n_bootstrap": 100}).json()
    assert b["status"] == "ok" and b["models_found"] == ["tfidf", "lstm", "bilstm", "bert"]
    assert b["warnings"] == [] and b["problems"] == {} and b["models_missing"] == []
    assert b == json.loads(json.dumps(C.compare(n_bootstrap=100)))
    entries = b["targets"]["subject"]["test"]["models"]
    assert {e["checkpoint_available"] for e in entries} == {True} and len(entries) == 4
    assert all(isinstance(e["trainable_parameters"], int) for e in entries if e["model"] != "tfidf")
    rows = client.get("/models/comparison/rows", params={"n_bootstrap": 0}).json()["rows"]
    assert len(rows) == 4 * 3 * 2
