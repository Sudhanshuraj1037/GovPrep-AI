"""API tests for the Word2Vec endpoints. Synthetic data only (see conftest)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import word2vec as W
from app.main import app

client = TestClient(app)
BASE = "/models/word2vec"

GET_ENDPOINTS = [
    (f"{BASE}/similar", {"word": "river"}),
    (f"{BASE}/similarity", {"word_a": "river", "word_b": "delta"}),
    (f"{BASE}/analogy", {"a": "river", "b": "delta", "c": "empire"}),
    (f"{BASE}/vector", {"word": "river"}),
]


# ======================================================== not trained yet
def test_status_reports_not_trained_with_200(env):
    r = client.get(f"{BASE}/status")
    assert r.status_code == 200
    b = r.json()
    assert b["trained"] is False and b["status"] == "not_trained_yet"
    assert b["variants_available"] == []


@pytest.mark.parametrize("path,params", GET_ENDPOINTS)
def test_get_endpoints_503_when_not_trained(env, path, params):
    r = client.get(path, params=params)
    assert r.status_code == 503
    d = r.json()["detail"]
    assert d["status"] == "not_trained_yet" and d["model"] == "word2vec"
    assert "neighbors" not in r.text and "vector" not in d  # no placeholder data


def test_visualize_503_when_not_trained(env):
    r = client.post(f"{BASE}/visualize", json={"method": "pca"})
    assert r.status_code == 503 and r.json()["detail"]["status"] == "not_trained_yet"


def test_health_word2vec_not_trained(env):
    body = client.get("/health").json()
    w = next(m for m in body["models"] if m["name"] == "word2vec")
    assert w["trained"] is False and w["variants_trained"] == []
    assert body["models_trained"] == 0


# ============================================================ trained
def test_status_when_trained(w2v_trained):
    b = client.get(f"{BASE}/status").json()
    assert b["trained"] is True and b["status"] == "trained"
    assert b["variants_available"] == ["cbow", "skipgram"] and b["default_variant"] == "skipgram"
    assert b["variants"]["cbow"]["vocab_size"] == b["variants"]["skipgram"]["vocab_size"] > 0
    assert b["corpus"]["source"] == "public_train" and b["seed"] == 42
    assert "val" in b["vocabulary_coverage"] and "test" in b["vocabulary_coverage"]
    assert b["variant_comparison"]["k"] == 10


def test_similar_words(w2v_trained):
    r = client.get(f"{BASE}/similar", params={"word": "River", "top_k": 4, "variant": "cbow"})
    assert r.status_code == 200
    b = r.json()
    assert b["query"] == "river" and b["variant"] == "cbow" and len(b["neighbors"]) == 4
    sims = [n["similarity"] for n in b["neighbors"]]
    assert sims == sorted(sims, reverse=True)
    svc = W.EmbeddingService.load()
    assert b == svc.most_similar("river", 4, "cbow")        # API == library result


def test_similarity_endpoint(w2v_trained):
    b = client.get(f"{BASE}/similarity", params={"word_a": "river", "word_b": "delta"}).json()
    svc = W.EmbeddingService.load()
    v = svc.vectors["skipgram"]
    assert b["cosine_similarity"] == pytest.approx(W.cosine_similarity(v["river"], v["delta"]), abs=1e-6)
    assert -1 <= b["cosine_similarity"] <= 1 and b["variant"] == "skipgram"


def test_word_vector_lookup(w2v_trained):
    b = client.get(f"{BASE}/vector", params={"word": "river"}).json()
    assert b["word"] == "river" and b["dimension"] == 50 == len(b["vector"])
    assert b["norm"] == pytest.approx(float(np.linalg.norm(b["vector"])), rel=1e-5)
    assert b["count"] > 0


def test_analogy_endpoint(w2v_trained):
    b = client.get(f"{BASE}/analogy", params={"a": "river", "b": "delta", "c": "empire", "top_k": 3}).json()
    assert len(b["results"]) == 3 and b["question"] == "river : delta :: empire : ?"
    assert not {"river", "delta", "empire"} & {x["word"] for x in b["results"]}
    assert "no analogy accuracy is claimed" in b["caveat"]


def test_out_of_vocabulary_word_is_404_with_the_words(w2v_trained):
    r = client.get(f"{BASE}/vector", params={"word": "zzzunknown"})
    assert r.status_code == 404
    assert r.json()["detail"] == {"status": "word_not_in_vocabulary", "words": ["zzzunknown"], "variant": "skipgram"}
    r = client.get(f"{BASE}/similarity", params={"word_a": "river", "word_b": "the"})
    assert r.status_code == 404 and r.json()["detail"]["words"] == ["the"]       # stop-word
    r = client.get(f"{BASE}/vector", params={"word": "भारत"})                       # Hindi, URL-encoded by the client
    assert r.status_code == 404 and r.json()["detail"]["words"] == ["भारत"]
    r = client.get(f"{BASE}/analogy", params={"a": "river", "b": "zzz1", "c": "zzz2"})
    assert r.status_code == 404 and r.json()["detail"]["words"] == ["zzz1", "zzz2"]


def test_variant_not_trained_is_404(w2v_trained):
    W.train_all(variants=("cbow",), cfg=W.Word2VecConfig(vector_size=50, epochs=5))
    r = client.get(f"{BASE}/vector", params={"word": "river", "variant": "skipgram"})
    assert r.status_code == 404
    assert r.json()["detail"] == {"status": "variant_not_trained", "variant": "skipgram", "available": ["cbow"]}
    assert client.get(f"{BASE}/vector", params={"word": "river"}).json()["variant"] == "cbow"


def test_input_validation(w2v_trained):
    assert client.get(f"{BASE}/similar").status_code == 422                                  # missing word
    assert client.get(f"{BASE}/similar", params={"word": ""}).status_code == 422
    assert client.get(f"{BASE}/similar", params={"word": "river", "top_k": 0}).status_code == 422
    assert client.get(f"{BASE}/similar", params={"word": "river", "top_k": 51}).status_code == 422
    assert client.get(f"{BASE}/similar", params={"word": "river", "variant": "glove"}).status_code == 422
    assert client.get(f"{BASE}/similarity", params={"word_a": "river"}).status_code == 422
    assert client.get(f"{BASE}/analogy", params={"a": "x", "b": "y"}).status_code == 422
    assert client.get(f"{BASE}/vector", params={"word": "x" * 101}).status_code == 422


# ===================================================== visualization data
def test_visualize_pca_json(w2v_trained):
    r = client.post(f"{BASE}/visualize", json={"method": "pca", "top_n": 25})
    assert r.status_code == 200
    b = r.json()
    assert b["method"] == "pca" and b["n_points"] == 25 == len(b["points"])
    p = b["points"][0]
    assert set(p) == {"word", "x", "y", "count", "source", "dominant_subject", "dominant_share"}
    assert len(b["explained_variance_ratio"]) == 2 and 0 < sum(b["explained_variance_ratio"]) <= 1
    assert set(b["subjects"]) == {"History", "Geography", "Science", "Polity"}
    assert b["missing_words"] == [] and b["notes"]
    svc = W.EmbeddingService.load()
    assert b["points"] == svc.visualize(method="pca", top_n=25)["points"]  # API == library result


def test_visualize_tsne_with_selected_words(w2v_trained):
    words = ["river", "delta", "empire", "dynasty", "atom", "photon", "constitution", "ghost"]
    r = client.post(f"{BASE}/visualize", json={"method": "tsne", "variant": "cbow", "words": words, "perplexity": 2})
    assert r.status_code == 200
    b = r.json()
    assert b["missing_words"] == ["ghost"] and b["n_points"] == 7 and b["variant"] == "cbow"
    assert b["parameters"]["perplexity"] == 2 and np.isfinite(b["kl_divergence"])
    by = {p["word"]: p for p in b["points"]}
    assert by["river"]["dominant_subject"] == "Geography" and by["atom"]["dominant_subject"] == "Science"
    assert all(p["source"] == "requested" for p in b["points"])


def test_visualize_neighbour_expansion(w2v_trained):
    b = client.post(f"{BASE}/visualize", json={"method": "pca", "words": ["river", "empire", "atom"],
                                                "neighbors_per_word": 2}).json()
    assert b["n_points"] > 3 and {p["source"] for p in b["points"]} == {"requested", "neighbor"}


def test_visualize_validation_and_errors(w2v_trained):
    post = lambda **j: client.post(f"{BASE}/visualize", json=j)  # noqa: E731
    assert post(method="umap").status_code == 422
    assert post(top_n=0).status_code == 422
    assert post(top_n=501).status_code == 422
    assert post(words=[]).status_code == 422
    assert post(words=[""]).status_code == 422
    assert post(neighbors_per_word=21).status_code == 422
    assert post(perplexity=1).status_code == 422
    assert post(variant="glove").status_code == 422
    r = post(method="pca", words=["river", "delta"])                    # too few points
    assert r.status_code == 422 and "at least 3" in r.json()["detail"]
    r = post(method="tsne", words=["river", "delta", "empire", "atom"])
    assert r.status_code == 422 and "at least 5" in r.json()["detail"]
    r = post(method="pca", words=["zzz1", "zzz2", "zzz3"])
    assert r.status_code == 404 and r.json()["detail"]["status"] == "word_not_in_vocabulary"


# ============================================================ health + cache
def test_health_word2vec_trained(w2v_trained):
    body = client.get("/health").json()
    w = next(m for m in body["models"] if m["name"] == "word2vec")
    assert w["trained"] is True and w["variants_trained"] == ["cbow", "skipgram"]
    assert body["models_trained"] == 1
    assert body["dependencies"]["gensim"]["installed"] is True


def test_health_not_trained_when_only_metadata_exists(w2v_trained):
    for f in W.artifacts_dir().glob("*.model*"):
        f.unlink()
    w = next(m for m in client.get("/health").json()["models"] if m["name"] == "word2vec")
    assert w["trained"] is False and w["variants_trained"] == []


def test_model_cache_picks_up_retraining(w2v_trained):
    first = client.get(f"{BASE}/status").json()["trained_at"]
    W.train_all(variants=("skipgram",), cfg=W.Word2VecConfig(vector_size=50, epochs=5))
    second = client.get(f"{BASE}/status").json()
    assert second["trained_at"] != first and second["variants_available"] == ["skipgram"]
    assert client.get(f"{BASE}/vector", params={"word": "river"}).json()["variant"] == "skipgram"


def test_missing_model_file_gives_503_not_500(w2v_trained):
    (W.artifacts_dir() / "skipgram.model").unlink()
    r = client.get(f"{BASE}/vector", params={"word": "river"})
    assert r.status_code == 503 and r.json()["detail"]["status"] == "not_trained_yet"
    assert "missing" in r.json()["detail"]["message"]
