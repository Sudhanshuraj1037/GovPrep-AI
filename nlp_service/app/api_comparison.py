"""Model-comparison endpoints (Phase 7): TF-IDF vs LSTM vs BiLSTM vs BERT.

* GET /models/comparison         aligned tables, rankings, bootstrap intervals, comparability checks
* GET /models/comparison/rows    the same numbers as flat rows (one per target / split / model)

Always answers 200 with a ``status`` of ``no_models_trained``, ``single_model`` or ``ok`` so a
dashboard can show an honest empty state. Everything comes from the real metrics.json files
(see ``app.comparison``); nothing is estimated and no library beyond NumPy is needed, so these
endpoints work without PyTorch or Transformers.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from .comparison import DEFAULT_BOOTSTRAP, comparison_rows, compare

router = APIRouter(tags=["Comparison"])

Bootstrap = Annotated[int, Query(ge=0, le=5000, description="bootstrap resamples for the intervals; 0 disables them")]


@router.get("/models/comparison")
def models_comparison(n_bootstrap: Bootstrap = DEFAULT_BOOTSTRAP) -> dict:
    return compare(n_bootstrap=n_bootstrap)


@router.get("/models/comparison/rows")
def models_comparison_rows(n_bootstrap: Bootstrap = DEFAULT_BOOTSTRAP) -> dict:
    c = compare(n_bootstrap=n_bootstrap)
    return {"status": c["status"], "models_found": c["models_found"], "warnings": c["warnings"],
            "rows": comparison_rows(c)}
