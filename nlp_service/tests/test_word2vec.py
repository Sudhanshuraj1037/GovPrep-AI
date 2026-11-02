"""Phase 3 tests: Word2Vec training, held-out discipline, similarity / analogy /
vectors, PCA + t-SNE data. All data here is SYNTHETIC (see conftest) and only
exercises the code; nothing computed on it is a result."""
import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gensim.models import KeyedVectors, Word2Vec

from app import data as D
from app import word2vec as W
from app.preprocessing import PreprocessConfig, TextPreprocessor
from app.registry import model_status
from tests.conftest import POOLS, make_toy_frame

NLP_ROOT = Path(__file__).resolve().parent.parent
POOL_OF = {w: s for s, ws in POOLS.items() for w in ws}


def _out():
    return W.artifacts_dir()


def _wv(variant="skipgram"):
    return Word2Vec.load(str(_out() / f"{variant}.model")).wv


def _metric_keys(obj) -> set:
    """All dict keys anywhere in a nested structure (values such as notes are ignored)."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _metric_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            keys |= _metric_keys(v)
    return keys


def _append_rows(split: str, text: str, n: int = 10) -> None:
    p = D.processed_path(split)
    df = pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else pd.DataFrame(columns=D.DATA_COLUMNS)
    extra = pd.DataFrame(
        [{"text": text, "subject": "History", "topic": "", "difficulty": ""}] * n
    )
    pd.concat([df, extra], ignore_index=True).to_csv(p, index=False)


# =============================================================== not trained
def test_no_data_means_not_trained_and_nothing_written(env):
    with pytest.raises(D.DataNotFoundError):
        W.train_all()
    assert not _out().exists() or not any(_out().iterdir())
    with pytest.raises(W.Word2VecNotTrainedError, match="not trained yet"):
        W.EmbeddingService.load()


def test_cli_reports_not_trained_yet(env, capsys):
    from training.train_word2vec import main

    assert main([]) == 2
    assert "NOT TRAINED YET" in capsys.readouterr().err
    assert not _out().exists() or not any(_out().iterdir())


def test_vocabulary_too_small_writes_nothing(toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    with pytest.raises(W.Word2VecTrainingError):
        W.train_all(cfg=W.Word2VecConfig(min_count=10_000))
    assert not _out().exists() or not any(_out().iterdir())


def test_invalid_variants_rejected(toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    with pytest.raises(ValueError):
        W.train_all(variants=("cbow", "glove"))
    with pytest.raises(ValueError):
        W.train_all(variants=())


# ================================================================== training
def test_trains_cbow_and_skipgram_and_saves_artifacts(w2v_trained):
    for f in ("cbow.model", "skipgram.model", "vocab_stats.json", "metadata.json"):
        assert (_out() / f).is_file()
    assert Word2Vec.load(str(_out() / "cbow.model")).sg == 0
    assert Word2Vec.load(str(_out() / "skipgram.model")).sg == 1
    m = w2v_trained
    assert m["model"] == "word2vec" and m["seed"] == 42
    assert m["corpus"]["source"] == "public_train"
    assert m["corpus"]["sha256"] == D.file_sha256(D.processed_path("public_train"))
    assert m["preprocessing"] == W.default_preprocess_config().to_dict()
    assert m["preprocessing"]["normalizer"] == "none"
    for v in ("cbow", "skipgram"):
        assert m["variants"][v]["vocab_size"] == len(_wv(v))
    assert m["variants"]["cbow"]["architecture"] == "CBOW"
    assert m["variants"]["skipgram"]["architecture"] == "skip-gram"
    assert not _metric_keys(m) & {"accuracy", "f1", "f1_macro", "precision", "recall", "confusion_matrix"}


def test_single_variant_training_and_stale_cleanup(w2v_trained):
    W.train_all(variants=("cbow",), cfg=W.Word2VecConfig(vector_size=50, epochs=5))
    assert (_out() / "cbow.model").exists()
    assert not (_out() / "skipgram.model").exists()
    assert W.EmbeddingService.load().variants == ["cbow"]


def test_corpus_statistics_are_recomputable(w2v_trained):
    train = D.load_split("public_train")
    pre = TextPreprocessor(W.default_preprocess_config())
    docs = [t for t in (pre.tokens(x) for x in train["text"]) if len(t) >= 2]
    assert w2v_trained["corpus"]["documents"] == len(docs)
    assert w2v_trained["corpus"]["tokens"] == sum(len(d) for d in docs)
    assert sum(w2v_trained["corpus"]["documents_per_subject"].values()) == len(docs)


def test_word_counts_match_independent_count(w2v_trained):
    train = D.load_split("public_train")
    pre = TextPreprocessor(W.default_preprocess_config())
    from collections import Counter

    counts = Counter(w for x in train["text"] for w in pre.tokens(x))
    svc = W.EmbeddingService.load()
    for word in ("river", "empire", "atom"):
        assert svc.vector(word)["count"] == counts[word]


def test_vocabulary_coverage_matches_recomputation(w2v_trained):
    pre = TextPreprocessor(W.default_preprocess_config())
    vocab = set(_wv().index_to_key)
    val = D.load_split("public_val")["text"]
    toks = [t for x in val for t in pre.tokens(x)]
    cov = w2v_trained["vocabulary_coverage"]["val"]
    assert cov["tokens"] == len(toks)
    assert cov["in_vocabulary"] == sum(t in vocab for t in toks)
    assert cov["coverage"] == pytest.approx(cov["in_vocabulary"] / cov["tokens"])
    assert "pdf_heldout" not in w2v_trained["vocabulary_coverage"]  # none present


def test_heldout_coverage_reported_only_when_present(env, toy_csv):
    held = env / "held.csv"
    make_toy_frame(8, seed=11).to_csv(held, index=False)
    D.prepare_splits(toy_csv, heldout_csv=held, seed=42)
    m = W.train_all(cfg=W.Word2VecConfig(vector_size=20, epochs=5))
    assert m["vocabulary_coverage"]["pdf_heldout"]["tokens"] > 0


def test_neighbour_overlap_between_variants(w2v_trained):
    c = w2v_trained["variant_comparison"]
    assert c["k"] == 10 and 0.0 <= c["mean_jaccard"] <= 1.0
    W.train_all(variants=("skipgram",), cfg=W.Word2VecConfig(vector_size=20, epochs=5))
    assert W.EmbeddingService.load().metadata["variant_comparison"] is None


def test_vocab_stats_reflect_train_subjects(w2v_trained):
    svc = W.EmbeddingService.load()
    words = svc.vocab_stats["words"]
    assert set(svc.vocab_stats["subjects"]) == set(POOLS)
    for w, subject in POOL_OF.items():
        if w in words:
            assert words[w]["dominant_subject"] == subject
            assert words[w]["dominant_share"] == pytest.approx(1.0)
    # filler words occur in every subject -> low share
    assert words["following"]["dominant_share"] < 0.5


# ============================================ held-out data is never trained on
def test_vocab_built_from_public_train_only(env, toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    for split in ("public_val", "public_test", "pdf_heldout"):
        _append_rows(split, "zzsentinel zzsentinel zzsentinel zzsentinel history")
    W.train_all(cfg=W.Word2VecConfig(vector_size=20, epochs=5))
    for v in ("cbow", "skipgram"):
        assert "zzsentinel" not in _wv(v).key_to_index


def test_sentinel_control_is_found_when_in_train(env, toy_csv):
    """Control for the test above: the same sentinel DOES enter the vocabulary when it
    is in the training split, so the check cannot pass vacuously."""
    D.prepare_splits(toy_csv, seed=42)
    _append_rows("public_train", "zzsentinel zzsentinel zzsentinel zzsentinel history")
    W.train_all(cfg=W.Word2VecConfig(vector_size=20, epochs=5))
    assert "zzsentinel" in _wv().key_to_index


def test_vectors_unchanged_when_non_train_data_changes(env, toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    cfg = W.Word2VecConfig(vector_size=20, epochs=5)
    W.train_all(cfg=cfg)
    before = {v: (_wv(v).index_to_key[:], _wv(v).vectors.copy()) for v in ("cbow", "skipgram")}
    held = make_toy_frame(8, seed=11)
    held.to_csv(D.processed_path("pdf_heldout"), index=False)
    _append_rows("public_val", "zzsentinel river delta")
    _append_rows("public_test", "zzsentinel empire")
    W.train_all(cfg=cfg)
    for v, (keys, vecs) in before.items():
        assert _wv(v).index_to_key == keys
        assert np.array_equal(_wv(v).vectors, vecs)


# ============================================================ reproducibility
def test_training_reproducible_in_process(toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    cfg = W.Word2VecConfig(vector_size=20, epochs=5)
    W.train_all(cfg=cfg, seed=42)
    a = {v: _wv(v).vectors.copy() for v in ("cbow", "skipgram")}
    W.train_all(cfg=cfg, seed=42)
    for v in a:
        assert np.array_equal(_wv(v).vectors, a[v])


def test_training_depends_on_seed(toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    cfg = W.Word2VecConfig(vector_size=20, epochs=5)
    W.train_all(cfg=cfg, seed=1)
    a = _wv().vectors.copy()
    W.train_all(cfg=cfg, seed=2)
    assert not np.array_equal(_wv().vectors, a)


def test_reproducible_across_processes_with_different_hashseed(env, toy_csv):
    """Separate processes with different PYTHONHASHSEED values must produce identical vectors."""
    D.prepare_splits(toy_csv, seed=42)
    vectors = []
    for i, hashseed in enumerate(("1", "12345")):
        art = env / f"artifacts_run{i}"
        proc = subprocess.run(
            [sys.executable, "-m", "training.train_word2vec", "--epochs", "5", "--vector-size", "20"],
            cwd=NLP_ROOT,
            env={**os.environ, "NLP_DATA_DIR": str(env / "data"), "NLP_ARTIFACTS_DIR": str(art),
                 "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(NLP_ROOT)},
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        vectors.append(
            {v: Word2Vec.load(str(art / "word2vec" / f"{v}.model")).wv for v in ("cbow", "skipgram")}
        )
    for v in ("cbow", "skipgram"):
        assert vectors[0][v].index_to_key == vectors[1][v].index_to_key
        assert np.array_equal(vectors[0][v].vectors, vectors[1][v].vectors)


# ============================================================ health / marker
def test_marker_is_written_last(toy_csv, monkeypatch):
    D.prepare_splits(toy_csv, seed=42)
    real = W._write_json_atomic

    def fail_on_marker(path, payload):
        if Path(path).name == "metadata.json":
            raise OSError("disk full")
        return real(path, payload)

    monkeypatch.setattr(W, "_write_json_atomic", fail_on_marker)
    with pytest.raises(OSError):
        W.train_all(cfg=W.Word2VecConfig(vector_size=20, epochs=5))
    assert (_out() / "skipgram.model").exists()
    assert not (_out() / "metadata.json").exists()
    w2v = next(m for m in model_status() if m["name"] == "word2vec")
    assert w2v["trained"] is False and w2v["variants_trained"] == []


def test_registry_trained_only_with_marker_and_model_file(env):
    d = _out()
    d.mkdir(parents=True)
    status = lambda: next(m for m in model_status() if m["name"] == "word2vec")  # noqa: E731
    assert status()["trained"] is False
    (d / "metadata.json").write_text("{}")
    assert status()["trained"] is False            # marker alone is not enough
    (d / "cbow.model").write_bytes(b"x")
    assert status()["trained"] is True
    assert status()["variants_trained"] == ["cbow"]
    (d / "metadata.json").unlink()
    assert status()["trained"] is False            # weights alone are not enough


# ================================================================= cosine
def test_cosine_similarity_properties():
    u = np.array([1.0, 2.0, 3.0])
    assert W.cosine_similarity(u, u) == pytest.approx(1.0)
    assert W.cosine_similarity(u, -u) == pytest.approx(-1.0)
    assert W.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert W.cosine_similarity(u, 5 * u) == pytest.approx(1.0)  # scale invariant
    assert W.cosine_similarity(u, [0, 0, 0]) == 0.0
    v = np.array([-2.0, 0.5, 4.0])
    assert W.cosine_similarity(u, v) == pytest.approx(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))
    assert W.cosine_similarity(u, v) == pytest.approx(W.cosine_similarity(v, u))


# ============================================================= word queries
def test_similarity_equals_manual_cosine_and_gensim(w2v_trained):
    svc = W.EmbeddingService.load()
    wv = _wv()
    for a, b in itertools.islice(itertools.combinations(POOL_OF, 2), 15):
        r = svc.similarity(a, b)
        assert r["cosine_similarity"] == pytest.approx(
            float(np.dot(wv[a], wv[b]) / (np.linalg.norm(wv[a]) * np.linalg.norm(wv[b]))), abs=1e-6
        )
        assert r["cosine_similarity"] == pytest.approx(float(wv.similarity(a, b)), abs=1e-6)
        assert r["cosine_similarity"] == pytest.approx(svc.similarity(b, a)["cosine_similarity"])
    assert svc.similarity("river", "river")["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)


def test_input_is_normalised_like_training(w2v_trained):
    svc = W.EmbeddingService.load()
    assert svc.vector("River")["word"] == "river"
    assert svc.similarity("RIVER", " delta ")["word_b"] == "delta"


def test_most_similar_contract(w2v_trained):
    svc = W.EmbeddingService.load()
    wv = _wv()
    r = svc.most_similar("river", topn=7)
    sims = [n["similarity"] for n in r["neighbors"]]
    assert len(sims) == 7 and sims == sorted(sims, reverse=True)
    assert "river" not in [n["word"] for n in r["neighbors"]]
    assert r["variant"] == "skipgram"
    for n in r["neighbors"]:
        assert n["similarity"] == pytest.approx(W.cosine_similarity(wv["river"], wv[n["word"]]), abs=1e-5)
    # it really is the nearest: no other word is closer than the 7th neighbour
    others = [w for w in wv.index_to_key if w != "river"]
    best = sorted(others, key=lambda w: -W.cosine_similarity(wv["river"], wv[w]))[:7]
    assert [n["word"] for n in r["neighbors"]] == best


@pytest.mark.parametrize("variant", ["cbow", "skipgram"])
def test_neighbours_recover_subject_structure_on_synthetic_corpus(w2v_trained, variant):
    """Synthetic words from one subject co-occur, so their embeddings should neighbour each
    other. Checks the training/lookup chain end to end (not a performance claim)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    svc = W.EmbeddingService.load()
    vocab = set(svc.vectors[variant].key_to_index)
    # the only pool words missing from the vocabulary are ones the preprocessor drops as stop-words
    # (e.g. "bill" is in scikit-learn's English list)
    assert {w for w in POOL_OF if w not in vocab} == {w for w in POOL_OF if w in ENGLISH_STOP_WORDS}
    rates = []
    for w, subject in ((w, s) for w, s in POOL_OF.items() if w in vocab):
        near = [n["word"] for n in svc.most_similar(w, 5, variant)["neighbors"] if n["word"] in POOL_OF]
        if near:
            rates.append(np.mean([POOL_OF[x] == subject for x in near]))
    assert np.mean(rates) > 0.9  # chance level is about 0.25


def test_unknown_words_are_reported_not_guessed(w2v_trained):
    svc = W.EmbeddingService.load()
    for bad in ("zzzunknown", "the", "river delta", "भारत", "!!!"):
        with pytest.raises(W.WordNotInVocabularyError) as e:
            svc.vector(bad)
        assert e.value.words == [bad]
    with pytest.raises(W.WordNotInVocabularyError) as e:
        svc.similarity("river", "zzzunknown")
    assert e.value.words == ["zzzunknown"]


def test_variant_selection(w2v_trained):
    svc = W.EmbeddingService.load()
    assert svc.vector("river")["variant"] == "skipgram"
    assert svc.vector("river", "cbow")["variant"] == "cbow"
    assert not np.allclose(svc.vector("river", "cbow")["vector"], svc.vector("river", "skipgram")["vector"])
    W.train_all(variants=("cbow",), cfg=W.Word2VecConfig(vector_size=50, epochs=5))
    only_cbow = W.EmbeddingService.load()
    assert only_cbow.vector("river")["variant"] == "cbow"          # default falls back
    with pytest.raises(W.UnknownVariantError) as e:
        only_cbow.vector("river", "skipgram")
    assert e.value.available == ["cbow"]


def test_vector_lookup_contents(w2v_trained):
    svc = W.EmbeddingService.load()
    wv = _wv()
    r = svc.vector("river")
    assert r["dimension"] == 50 == len(r["vector"])
    assert np.allclose(r["vector"], wv["river"])
    assert r["norm"] == pytest.approx(float(np.linalg.norm(wv["river"])))
    assert all(isinstance(x, float) for x in r["vector"])


# ================================================================== analogy
def test_analogy_matches_gensim_and_excludes_inputs(w2v_trained):
    svc = W.EmbeddingService.load()
    wv = _wv()
    r = svc.analogy("river", "delta", "empire", topn=5)
    expected = wv.most_similar(positive=["delta", "empire"], negative=["river"], topn=5)
    assert [x["word"] for x in r["results"]] == [w for w, _ in expected]
    assert not {"river", "delta", "empire"} & {x["word"] for x in r["results"]}
    assert r["question"] == "river : delta :: empire : ?"
    assert "no analogy accuracy is claimed" in r["caveat"]


def test_analogy_logic_on_hand_built_vectors():
    """Independent of training: with constructed vectors the arithmetic must give 'queen'."""
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(
        ["man", "woman", "king", "queen", "apple", "car"],
        np.array([[1, 0, 0], [-1, 0, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1], [0.1, -1, 0.5]], dtype=np.float32),
    )
    meta = {"preprocessing": PreprocessConfig(normalizer="none").to_dict(), "variants": {"skipgram": {}}}
    svc = W.EmbeddingService({"skipgram": kv}, meta)
    r = svc.analogy("man", "king", "woman", topn=3)
    assert r["results"][0]["word"] == "queen"
    assert r["results"][0]["similarity"] > r["results"][1]["similarity"]
    assert {"man", "king", "woman"}.isdisjoint(x["word"] for x in r["results"])


def test_analogy_requires_all_words_in_vocabulary(w2v_trained):
    svc = W.EmbeddingService.load()
    with pytest.raises(W.WordNotInVocabularyError) as e:
        svc.analogy("river", "zzz1", "zzz2")
    assert e.value.words == ["zzz1", "zzz2"]


# ============================================================ visualization
def test_pca_coordinates_match_independent_svd(w2v_trained):
    svc = W.EmbeddingService.load()
    r = svc.visualize(method="pca", top_n=30)
    assert r["method"] == "pca" and r["variant"] == "skipgram" and r["n_points"] == 30 == len(r["points"])
    words = [p["word"] for p in r["points"]]
    assert words == _wv().index_to_key[:30]                       # most frequent first
    X = np.asarray(_wv()[words], dtype=np.float64)
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ref = Xc @ Vt[:2].T
    got = np.array([[p["x"], p["y"]] for p in r["points"]])
    assert np.allclose(np.abs(got), np.abs(ref), atol=1e-6)        # equal up to axis sign
    ratio = (S**2) / (S**2).sum()
    assert np.allclose(r["explained_variance_ratio"], ratio[:2], atol=1e-8)
    assert 0 < sum(r["explained_variance_ratio"]) <= 1


def test_visualization_is_deterministic(w2v_trained):
    svc = W.EmbeddingService.load()
    for m in ("pca", "tsne"):
        a = svc.visualize(method=m, top_n=30)
        b = svc.visualize(method=m, top_n=30)
        assert a["points"] == b["points"]


def test_tsne_output_and_parameters(w2v_trained):
    svc = W.EmbeddingService.load()
    r = svc.visualize(method="tsne", top_n=30)
    assert r["parameters"]["seed"] == 42 and r["parameters"]["perplexity"] == pytest.approx(29 / 3)
    assert np.isfinite(r["kl_divergence"]) and r["kl_divergence"] >= 0
    assert all(np.isfinite(p["x"]) and np.isfinite(p["y"]) for p in r["points"])
    assert "explained_variance_ratio" not in r
    r2 = svc.visualize(method="tsne", top_n=30, perplexity=5)
    assert r2["parameters"]["perplexity"] == 5


def test_requested_words_missing_words_and_neighbour_expansion(w2v_trained):
    svc = W.EmbeddingService.load()
    # input is normalised and de-duplicated ("River" and "river" are one word)
    r = svc.visualize(method="pca", words=["River", "delta", "river", "empire"], variant="cbow")
    assert [p["word"] for p in r["points"]] == ["river", "delta", "empire"] and r["variant"] == "cbow"
    r = svc.visualize(method="pca", words=["river", "empire", "atom", "zzzunknown"])
    assert [p["word"] for p in r["points"]] == ["river", "empire", "atom"]
    assert {p["source"] for p in r["points"]} == {"requested"}
    assert r["missing_words"] == ["zzzunknown"]
    r = svc.visualize(method="pca", words=["river", "empire", "atom"], neighbors_per_word=3)
    sources = [p["source"] for p in r["points"]]
    assert sources[:3] == ["requested"] * 3 and set(sources[3:]) == {"neighbor"}
    assert len({p["word"] for p in r["points"]}) == r["n_points"] > 3     # no duplicates
    capped = svc.visualize(method="pca", words=["river", "empire", "atom"], neighbors_per_word=10, max_points=8)
    assert capped["n_points"] == 8


def test_points_carry_subject_metadata(w2v_trained):
    r = W.EmbeddingService.load().visualize(method="pca", top_n=41)
    by_word = {p["word"]: p for p in r["points"]}
    assert by_word["river"]["dominant_subject"] == "Geography"
    assert by_word["river"]["dominant_share"] == pytest.approx(1.0)
    assert by_word["river"]["count"] > 0
    assert set(r["subjects"]) == set(POOLS)
    assert any("dominant_subject" in n for n in r["notes"])


def test_visualization_errors(w2v_trained):
    svc = W.EmbeddingService.load()
    with pytest.raises(W.VisualizationError, match="at least 3"):
        svc.visualize(method="pca", words=["river", "delta"])
    with pytest.raises(W.VisualizationError, match="at least 5"):
        svc.visualize(method="tsne", words=["river", "delta", "empire", "atom"])
    with pytest.raises(W.VisualizationError):
        svc.visualize(method="umap")
    with pytest.raises(W.VisualizationError):
        svc.visualize(method="pca", top_n=0)
    with pytest.raises(W.VisualizationError):
        svc.visualize(method="pca", top_n=501)
    with pytest.raises(W.WordNotInVocabularyError):
        svc.visualize(method="pca", words=["zzz1", "zzz2", "zzz3"])
