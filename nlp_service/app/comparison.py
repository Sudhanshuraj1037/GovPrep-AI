"""TF-IDF vs LSTM vs BiLSTM vs BERT, from the real metrics.json files only (Phase 7).

Everything here is derived from ``artifacts/<model>/metrics.json`` through the shared
``app.model_metrics`` contract. Nothing is trained, estimated or filled in: a model that has
not been trained simply does not appear, and with fewer than two models there is nothing to
compare.

What the data supports, and what it does not
--------------------------------------------
* Each ``metrics.json`` stores a confusion matrix per split, not per-example predictions.
* A confusion matrix is a complete summary of the (true, predicted) pairs, so accuracy,
  macro-F1 and weighted-F1 can be recomputed from it EXACTLY (a test checks this against the
  reported values) and **bootstrap intervals** can be computed from it: resampling n examples
  with replacement is the same as drawing n cells from the empirical joint distribution.
* The intervals are UNPAIRED (each model is resampled independently). Paired tests such as
  McNemar need per-example predictions, which are not stored, so none is claimed. The rule used
  is deliberately conservative: the best model is called "separated" from the runner-up only if
  its interval lies entirely above the runner-up's. Overlapping intervals mean the data do not
  establish a difference, whatever the point estimates say.

Comparability
-------------
Models are compared on a (target, split) only if they were scored on the same rows and classes;
anything else is listed under ``excluded`` with the reason. The split identity of the models
(same processed split files / same split manifest) is checked and reported, and a mismatch is
raised as a warning because it would invalidate the comparison.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from . import model_metrics as MM
from .config import settings

RANK_BY = "f1_macro"
CI_METRICS = ("accuracy", "f1_macro", "f1_weighted")
DEFAULT_BOOTSTRAP = 1000
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05

NOTES = [
    "Every number comes from a real metrics.json; models that are not trained are absent, never estimated.",
    "Intervals are unpaired bootstrap intervals computed from each model's confusion matrix. 'Separated' means the "
    "best model's interval lies entirely above the runner-up's (a conservative rule); overlapping intervals mean the "
    "data do not establish a difference. Per-example predictions are not stored, so no paired test is claimed.",
    "Models are compared on a (target, split) only if they were scored on the same rows and classes.",
    "pdf_heldout is the domain-shift evaluation (GovPrep-AI PDF text, never used for training or tuning). "
    "A gap between test and pdf_heldout is a finding, not an error.",
    "trainable_parameters are not directly comparable across families (TF-IDF has none in this sense); BiLSTM has about "
    "twice the recurrent parameters of an LSTM trained with the same flags.",
]


# --------------------------------------------------------- matrix -> metrics
def metrics_from_matrices(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Accuracy, macro-F1 and weighted-F1 for one matrix ``(k, k)`` or a stack ``(B, k, k)``.

    Same definitions as ``app.metrics.classification_metrics`` (scikit-learn, zero_division=0,
    macro over every label of the matrix, weights = true support)."""
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim == 2:
        cm = cm[None]
    tp = np.einsum("bii->bi", cm)
    fp = cm.sum(axis=1) - tp
    fn = cm.sum(axis=2) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
    support = cm.sum(axis=2)
    total = support.sum(axis=1)
    return {
        "accuracy": tp.sum(axis=1) / total,
        "f1_macro": f1.mean(axis=1),
        "f1_weighted": (f1 * support).sum(axis=1) / total,
    }


def bootstrap_intervals(
    cm: np.ndarray, *, n_bootstrap: int, seed: int, alpha: float = DEFAULT_ALPHA
) -> dict[str, list[float]] | None:
    """Percentile bootstrap intervals for accuracy / macro-F1 / weighted-F1 from a confusion matrix."""
    if n_bootstrap <= 0:
        return None
    cm = np.asarray(cm, dtype=np.int64)
    n, k = int(cm.sum()), cm.shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, (cm / n).ravel(), size=n_bootstrap).reshape(n_bootstrap, k, k)
    stats = metrics_from_matrices(draws)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {name: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for name, v in stats.items()}


def _seed_for(base: int, *parts: str) -> int:
    return (base * 1_000_003 + zlib.crc32("|".join(parts).encode("utf-8"))) % (2**32)


# --------------------------------------------------------------- comparison
def _split_identity(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Did the models use the same processed split? Uses the file checksums where recorded
    (LSTM / BiLSTM / BERT) and the split manifest (all models, when prepare_data produced one)."""
    files = {m: r["data"].get("split_files_sha256") for m, r in raw.items() if r["data"].get("split_files_sha256")}
    manifests = {}
    for m, r in raw.items():
        man = r["data"].get("split_manifest")
        if man:
            manifests[m] = json.dumps(
                {"input": (man.get("public_input") or {}).get("sha256"), "split": man.get("split")}, sort_keys=True
            )
    out: dict[str, Any] = {
        "files_sha256_equal": (len({json.dumps(v, sort_keys=True) for v in files.values()}) == 1) if len(files) > 1 else None,
        "manifest_equal": (len(set(manifests.values())) == 1) if len(manifests) > 1 else None,
        "models_with_file_checksums": sorted(files),
        "models_with_manifest": sorted(manifests),
        "models_without_evidence": sorted(m for m in raw if m not in files and m not in manifests),
    }
    return out


def _entry(model: str, target: str, split: str, raw: dict[str, Any], cm: np.ndarray,
           checkpoint: bool | None, n_bootstrap: int, seed: int, alpha: float) -> dict[str, Any]:
    e = raw["targets"][target]
    m = e["splits"][split]
    return {
        "model": model,
        "n": m["n"],
        "accuracy": m["accuracy"],
        "precision_macro": m["precision_macro"],
        "recall_macro": m["recall_macro"],
        "f1_macro": m["f1_macro"],
        "precision_weighted": m["precision_weighted"],
        "recall_weighted": m["recall_weighted"],
        "f1_weighted": m["f1_weighted"],
        "ci": bootstrap_intervals(cm, n_bootstrap=n_bootstrap, seed=_seed_for(seed, model, target, split), alpha=alpha),
        "trainable_parameters": (e.get("model") or {}).get("trainable_parameters"),
        "checkpoint_available": checkpoint,
        "trained_at": raw["trained_at"],
    }


def compare(
    artifacts_dir: str | Path | None = None,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Aligned comparison of every model that has a valid metrics.json."""
    base = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)
    res = MM.collect(base)
    found = res["models_found"]
    raw = {m: json.loads((base / m / "metrics.json").read_text(encoding="utf-8")) for m in found}
    checkpoint = {
        m: next((r["checkpoint_available"] for r in res["rows"] if r["model"] == m), None) for m in found
    }

    identity = _split_identity(raw)
    warnings: list[str] = []
    if identity["files_sha256_equal"] is False:
        warnings.append("The models were trained on DIFFERENT processed split files: the comparison is not valid.")
    if identity["manifest_equal"] is False:
        warnings.append("The models report DIFFERENT split manifests (input file or split parameters): the comparison is not valid.")
    if identity["models_without_evidence"]:
        warnings.append(f"No split evidence recorded for {identity['models_without_evidence']}: identical splits cannot be verified.")
    if len(found) < 2:
        warnings.append("Fewer than two trained models: there is nothing to compare yet.")

    targets: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []
    gaps: dict[str, Any] = {}
    all_targets = [t for t in MM.TARGET_ORDER if any(raw[m]["targets"].get(t, {}).get("status") == "trained" for m in found)]
    for target in all_targets:
        targets[target] = {}
        have = [m for m in found if raw[m]["targets"].get(target, {}).get("status") == "trained"]
        skipped = {m: raw[m]["targets"][target].get("reason") for m in found
                   if raw[m]["targets"].get(target, {}).get("status") == "skipped"}
        for split in MM.SPLITS:
            cands = {}
            for m in have:
                s = raw[m]["targets"][target]["splits"].get(split)
                if s is not None:
                    cands[m] = (s, np.array(s["confusion_matrix"]["matrix"], dtype=np.int64))
            if not cands:
                continue
            # models are comparable only if they scored the same rows and classes
            sig = {m: (tuple(s["confusion_matrix"]["labels"]), s["n"]) for m, (s, _) in cands.items()}
            groups: dict[Any, list[str]] = {}
            for m, g in sig.items():
                groups.setdefault(g, []).append(m)
            keep_sig = max(groups, key=lambda g: (len(groups[g]), -min(MM.MODELS.index(x) for x in groups[g])))
            kept = sorted(groups[keep_sig], key=MM.MODELS.index)
            excluded = {m: f"scored on a different set of rows/classes (n={sig[m][1]}, {len(sig[m][0])} classes)"
                        for m in cands if m not in kept}
            entries = [
                _entry(m, target, split, raw[m], cands[m][1], checkpoint[m], n_bootstrap, seed, alpha) for m in kept
            ]
            order = sorted(entries, key=lambda e: (-e[RANK_BY], MM.MODELS.index(e["model"])))
            ranks: dict[str, int] = {}
            for i, e in enumerate(order):
                ranks[e["model"]] = ranks[order[i - 1]["model"]] if i and e[RANK_BY] == order[i - 1][RANK_BY] else i + 1
            majority = raw[kept[0]]["targets"][target]["splits"][split]["majority_class_baseline"]["accuracy"]
            tfidf = next((e for e in entries if e["model"] == "tfidf"), None)
            for e in entries:
                e["rank_f1_macro"] = ranks[e["model"]]
                e["accuracy_minus_majority_baseline"] = e["accuracy"] - majority
                e["f1_macro_minus_tfidf"] = (e["f1_macro"] - tfidf["f1_macro"]) if tfidf and e is not tfidf else None
            entries = order
            best = entries[0]
            separated = None
            if len(entries) > 1 and best["ci"] and entries[1]["ci"]:
                separated = best["ci"]["f1_macro"][0] > entries[1]["ci"]["f1_macro"][1]
            targets[target][split] = {
                "n": cands[kept[0]][0]["n"],
                "labels": list(keep_sig[0]),
                "majority_class_accuracy": majority,
                "models": entries,
                "best_by_f1_macro": best["model"],
                "tied_for_best": [e["model"] for e in entries if e["f1_macro"] == best["f1_macro"]],
                "separated_from_runner_up": separated,
                "excluded": excluded,
                "skipped_target": skipped,
            }
            if split == "test":
                summary.append({
                    "target": target, "split": "test", "best_model": best["model"], "f1_macro": best["f1_macro"],
                    "tied_for_best": [e["model"] for e in entries if e["f1_macro"] == best["f1_macro"]],
                    "models_compared": len(entries), "separated_from_runner_up": separated,
                })
        # test vs pdf_heldout: how much does each model lose on real GovPrep-AI text?
        t, h = targets[target].get("test"), targets[target].get("pdf_heldout")
        if t and h:
            hmap = {e["model"]: e for e in h["models"]}
            gaps[target] = {
                e["model"]: {
                    "test_f1_macro": e["f1_macro"], "pdf_heldout_f1_macro": hmap[e["model"]]["f1_macro"],
                    "f1_macro_gap": e["f1_macro"] - hmap[e["model"]]["f1_macro"],
                    "test_accuracy": e["accuracy"], "pdf_heldout_accuracy": hmap[e["model"]]["accuracy"],
                }
                for e in t["models"] if e["model"] in hmap
            }

    status = "no_models_trained" if not found else "single_model" if len(found) == 1 else "ok"
    return {
        "status": status,
        "models_found": found,
        "models_missing": res["models_missing"],
        "problems": res["problems"],
        "parameters": {"rank_by": RANK_BY, "n_bootstrap": n_bootstrap, "seed": seed, "alpha": alpha,
                       "interval": f"{round(100 * (1 - alpha))}% percentile bootstrap, unpaired"},
        "split_identity": identity,
        "warnings": warnings,
        "summary": summary,
        "targets": targets,
        "generalisation_gap": gaps,
        "notes": NOTES,
    }


# ------------------------------------------------------------------- exports
def comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat rows (one per target / split / model) with the interval bounds, for CSV or a table."""
    rows = []
    for target, splits in comparison["targets"].items():
        for split, block in splits.items():
            for e in block["models"]:
                row = {
                    "target": target, "split": split, "model": e["model"], "rank_f1_macro": e["rank_f1_macro"],
                    "n": e["n"],
                    **{k: e[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                                         "precision_weighted", "recall_weighted", "f1_weighted")},
                }
                for name in CI_METRICS:
                    ci = (e["ci"] or {}).get(name)
                    row[f"{name}_ci_low"], row[f"{name}_ci_high"] = (ci if ci else (None, None))
                row.update(
                    majority_class_accuracy=block["majority_class_accuracy"],
                    f1_macro_minus_tfidf=e["f1_macro_minus_tfidf"],
                    trainable_parameters=e["trainable_parameters"],
                    best_in_group=e["model"] == block["best_by_f1_macro"],
                    separated_from_runner_up=block["separated_from_runner_up"] if e["model"] == block["best_by_f1_macro"] else None,
                )
                rows.append(row)
    return rows


def to_markdown(comparison: dict[str, Any]) -> str:
    """A human-readable report of the comparison, for the project report / viva."""
    p = comparison["parameters"]
    out = ["# Model comparison: " + " vs ".join(comparison["models_found"]) if comparison["models_found"] else "# Model comparison", ""]
    out.append(f"Models compared: {', '.join(comparison['models_found']) or 'none'}. "
               f"Not trained (absent): {', '.join(comparison['models_missing']) or 'none'}.")
    out.append(f"Ranking metric: macro-F1. Intervals: {p['interval']} ({p['n_bootstrap']} resamples, seed {p['seed']}).")
    out.append("")
    for w in comparison["warnings"]:
        out.append(f"> WARNING: {w}")
    if comparison["warnings"]:
        out.append("")
    f = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
    ci = lambda e, k: "" if not (e["ci"] and k in e["ci"]) else f" [{e['ci'][k][0]:.3f}, {e['ci'][k][1]:.3f}]"  # noqa: E731
    for target, splits in comparison["targets"].items():
        for split, block in splits.items():
            out.append(f"## {target} - {split} (n={block['n']}, {len(block['labels'])} classes, "
                       f"majority-class accuracy {block['majority_class_accuracy']:.4f})")
            out.append("")
            out.append("| rank | model | accuracy | macro-F1 | weighted-F1 | macro precision | macro recall | trainable params |")
            out.append("|---:|---|---|---|---|---|---|---:|")
            for e in block["models"]:
                params = f"{e['trainable_parameters']:,}" if e["trainable_parameters"] else "-"
                out.append(f"| {e['rank_f1_macro']} | {e['model']} | {f(e['accuracy'])}{ci(e, 'accuracy')} | "
                           f"{f(e['f1_macro'])}{ci(e, 'f1_macro')} | {f(e['f1_weighted'])}{ci(e, 'f1_weighted')} | "
                           f"{f(e['precision_macro'])} | {f(e['recall_macro'])} | {params} |")
            sep = block["separated_from_runner_up"]
            verdict = ("not assessed" if sep is None else
                       "its interval lies entirely above the runner-up's" if sep else
                       "intervals overlap: no difference is established")
            out.append("")
            if len(block["tied_for_best"]) > 1:
                out.append(f"Tied for best by macro-F1 (identical point estimates): **{', '.join(block['tied_for_best'])}**.")
            else:
                out.append(f"Best by macro-F1: **{block['best_by_f1_macro']}** ({verdict}).")
            for m, why in block["excluded"].items():
                out.append(f"Excluded: {m} ({why}).")
            for m, why in block["skipped_target"].items():
                out.append(f"Skipped target for {m}: {why}")
            out.append("")
    if comparison["generalisation_gap"]:
        out.append("## Test vs PDF held-out (domain shift), macro-F1")
        out.append("")
        out.append("| target | model | test | pdf_heldout | gap |")
        out.append("|---|---|---|---|---|")
        for target, per in comparison["generalisation_gap"].items():
            for model, g in per.items():
                out.append(f"| {target} | {model} | {f(g['test_f1_macro'])} | {f(g['pdf_heldout_f1_macro'])} | {g['f1_macro_gap']:+.4f} |")
        out.append("")
    out.append("## Notes")
    out.extend(f"- {n}" for n in comparison["notes"])
    return "\n".join(out) + "\n"
