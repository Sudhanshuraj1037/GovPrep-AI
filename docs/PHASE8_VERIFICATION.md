# Phase 8 verification record

Run on 2026-11-03 in a Linux sandbox (Python 3.12.3, Node 22, PyTorch 2.15 CPU execution, transformers 5.22).
Nothing here was run on the owner's Windows machine; repeat the commands below there before relying on it.

| Check | Command | Result |
|---|---|---|
| Node suite (existing app + proxy + real FastAPI process) | `cd backend && npm test` | **156 passed, 0 failed, 0 skipped** |
| Python suite (service, all 5 model families, API) | `cd nlp_service && python -m pytest -q` | **501 passed** (before Phase 10 additions) |
| Live proxy, real servers, no trained models | see below | behaves as documented |

## Live Node -> FastAPI requests (FastAPI on :8001, backend on :5055, `NLP_SERVICE_URL` set)

| Request | Observed |
|---|---|
| `GET /api/nlp/health` without token | 401, never reaches FastAPI |
| `GET /api/nlp/health` with token | 200, header `X-NLP-Proxy: forwarded`, real service payload |
| `POST /api/nlp/preprocess` | 200, real tokens/stems |
| `POST /api/nlp/classify/{tfidf,lstm,bilstm,bert}` | 503 `not_trained_yet`, forwarded unchanged (no invented prediction) |
| `GET /api/nlp/models/word2vec/similar` | 503 `not_trained_yet`, forwarded |
| `GET /api/nlp/models/comparison` | 200 `no_models_trained` |
| `POST /api/nlp/classify/tfidf` with unknown field | 400 `invalid_request`, `source: node-proxy` |
| FastAPI stopped | 503 `nlp_service_unavailable` (`ECONNREFUSED`) from the proxy; existing `/api/health` still 200 |

## What this does NOT prove
* No model has been trained, so no real prediction has gone through the proxy yet.
* BERT was never loaded (no pretrained weights could be downloaded in the sandbox); only its not-trained path ran.
* The test token/secret were local throw-aways. Production secrets were not used.
