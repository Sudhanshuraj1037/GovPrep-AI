"""LSTM endpoints (Phase 4), and the factory that also builds the BiLSTM endpoints (Phase 5).

``build_router(name, ...)`` creates the three endpoints for one recurrent model:

* GET  /models/<name>/status    always 200: trained? targets, hyper-parameters, headline scores
* GET  /models/<name>/metrics   the real metrics.json of the last training run
* POST /classify/<name>         subject / topic / difficulty for texts

``api_lstm`` uses it for ``lstm``; ``api_bilstm`` uses it for ``bilstm``, so the two sets of
endpoints are equivalent by construction.

This module does NOT import PyTorch at load time, so the service and /health keep
working when PyTorch is not installed. Only /classify/<name> needs it; without it the
endpoint answers 503 ``dependency_missing``. With no trained model the endpoints
answer 503 / 404 ``not_trained_yet``. Nothing is ever faked.
"""
from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException

from .api_nlp import ClassifyRequest
from .config import settings

def _installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def _torch_installed() -> bool:
    return _installed("torch")


_PACKAGE_NAMES = {"torch": "PyTorch", "transformers": "Hugging Face Transformers"}


def _headline(entry: dict[str, Any]) -> dict[str, Any]:
    """Compact per-target summary taken verbatim from metrics.json."""
    if entry.get("status") != "trained":
        return {"status": entry.get("status"), "reason": entry.get("reason")}
    t = entry.get("training", {})
    out: dict[str, Any] = {
        "status": "trained",
        "classes": len(entry.get("labels", [])),
        "n_train": entry.get("n_train"),
        "epochs_run": t.get("epochs_run"),
        "best_epoch": t.get("best_epoch"),
        "stopped_early": t.get("stopped_early"),
        "model_selection": t.get("model_selection"),
        "trainable_parameters": entry.get("model", {}).get("trainable_parameters"),
        "scores": {},
    }
    for split, m in entry.get("splits", {}).items():
        out["scores"][split] = {
            "n": m["n"],
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
        }
    return out


def build_router(
    name: str,
    label: str,
    importer: Callable[[], Any],
    service_attr: str,
    cache: dict,
    not_trained_exc: str = "LSTMNotTrainedError",
    packages: tuple[str, ...] = ("torch",),
) -> APIRouter:
    """Create the status / metrics / classify endpoints for one neural model.

    ``importer`` returns the (torch-dependent) module on demand; it is looked up lazily so
    tests can patch it. ``service_attr`` names the service class inside that module and
    ``not_trained_exc`` the exception it raises when no usable model exists. ``packages`` lists
    the optional libraries inference needs (``torch``; BERT also needs ``transformers``): their
    availability is reported by /status and a missing one makes /classify answer
    503 ``dependency_missing``.
    """
    router = APIRouter(tags=[label])
    lock = threading.Lock()

    def _dir() -> Path:
        return Path(settings.artifacts_dir) / name

    def _has_weights() -> bool:
        return any(_dir().glob("*.pt"))

    def _not_trained(message: str) -> dict:
        return {"status": "not_trained_yet", "model": name, "message": message}

    def _read_metrics() -> dict[str, Any] | None:
        f = _dir() / "metrics.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None

    def _service():
        metrics_file = _dir() / "metrics.json"
        if not metrics_file.is_file() or not _has_weights():
            cache.clear()
            raise HTTPException(
                status_code=503,
                detail=_not_trained(
                    f"{label} not trained yet (needs artifacts/{name}/metrics.json and at least one .pt checkpoint)."
                ),
            )
        try:
            mod = importer()
        except ImportError as exc:
            missing = [_PACKAGE_NAMES.get(p, p) for p in packages if not _installed(p)]
            what = " and ".join(missing) + (" is" if len(missing) == 1 else " are") + " not installed" \
                if missing else f"a required package could not be imported ({exc})"
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "dependency_missing",
                    "model": name,
                    "message": f"{what}, so the {label} cannot run inference ({exc}).",
                },
            ) from exc
        key = (str(metrics_file), metrics_file.stat().st_mtime_ns)  # a retrain is picked up automatically
        with lock:
            if cache.get("key") != key:
                try:
                    cache["svc"] = getattr(mod, service_attr).load()
                except getattr(mod, not_trained_exc) as exc:
                    raise HTTPException(status_code=503, detail=_not_trained(str(exc))) from exc
                cache["key"] = key
            return cache["svc"]

    @router.get(f"/models/{name}/status", name=f"{name}_status", operation_id=f"{name}_status")
    def model_status() -> dict:
        """Always 200: whether a real trained model exists, and what it is."""
        metrics = _read_metrics()
        base = {"model": name, "torch_installed": _torch_installed()}
        for pkg in packages:
            base[f"{pkg}_installed"] = _installed(pkg)
        inference_ready = all(_installed(p) for p in packages)
        if metrics is None:
            return {
                **base,
                "trained": False,
                "status": "not_trained_yet",
                "targets_trained": [],
                "message": f"{label} has not been trained. Run training.prepare_data then training.train_{name}.",
            }
        if not _has_weights():
            return {
                **base,
                "trained": False,
                "status": "not_trained_yet",
                "targets_trained": [],
                "metrics_available": True,
                "message": "metrics.json exists but no .pt checkpoint does (checkpoints are not committed to git). "
                f"Re-run training.train_{name} to use the model.",
            }
        return {
            **base,
            "trained": True,
            "status": "trained",
            "inference_available": inference_ready,
            "trained_at": metrics.get("trained_at"),
            "seed": metrics.get("seed"),
            "targets_trained": sorted(t for t, e in metrics.get("targets", {}).items() if e.get("status") == "trained"),
            "targets": {t: _headline(e) for t, e in metrics.get("targets", {}).items()},
            "hyperparameters": metrics.get("hyperparameters"),
            "preprocessing": metrics.get("preprocessing"),
            "embeddings": metrics.get("embeddings"),
            "vocabulary": metrics.get("vocabulary"),
            "framework": metrics.get("framework"),
            "pdf_heldout_evaluated": metrics.get("data", {}).get("pdf_heldout_evaluated"),
        }

    @router.get(f"/models/{name}/metrics", name=f"{name}_metrics", operation_id=f"{name}_metrics")
    def model_metrics() -> dict:
        """The real metrics.json written by the last training run."""
        metrics = _read_metrics()
        if metrics is None:
            raise HTTPException(
                status_code=404,
                detail=_not_trained(f"No metrics yet: the {label} has not been trained."),
            )
        # tell the caller whether this run's checkpoints are still on disk (not stored in the file)
        return {**metrics, "weights_available": _has_weights()}

    @router.post(f"/classify/{name}", name=f"classify_{name}", operation_id=f"classify_{name}")
    def classify(req: ClassifyRequest) -> dict:
        """Classify texts (subject / topic / difficulty)."""
        svc = _service()
        results = svc.predict(req.texts, top_k=req.top_k)
        for text, res in zip(req.texts, results):
            res["text_preview"] = text[:120]
        entries = svc.metrics.get("targets", {})
        return {
            "model": name,
            "trained_at": svc.trained_at,
            "device": "cpu",
            "targets_available": svc.targets,
            "targets_unavailable": {
                t: e.get("reason", "not trained") for t, e in entries.items() if e.get("status") != "trained"
            },
            "results": results,
        }

    return router


# ---------------------------------------------------------------------- LSTM
MODEL_NAME = "lstm"
_cache: dict = {}


def _import_lstm():
    """Import the torch-dependent module on demand (separate function so tests can patch it)."""
    from . import lstm

    return lstm


# the lambda looks `_import_lstm` up at call time, so patching the module attribute still works
router = build_router(MODEL_NAME, "LSTM", lambda: _import_lstm(), "LSTMService", _cache)
