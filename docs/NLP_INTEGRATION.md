# NLP service integration (Phase 8)

```
React (not connected yet)  ->  Node/Express backend (/api/nlp/*, login required)  ->  Python NLP service (FastAPI)
                                       |
                                       +-> Groq (question generation, assistant): unchanged
```

The backend now proxies the Python NLP service (`nlp_service/`). Everything GovPrep-AI did before is unchanged:
authentication, PDF upload, question generation, scoring, results and the assistant. The proxy lives in
`backend/nlp/` and is mounted at `/api/nlp` with four added lines in `backend/server.js`. **The React frontend is not
changed yet; connecting it is the next phase.**

## What the proxy guarantees

- **Login required.** Every `/api/nlp` route uses the existing JWT middleware. Without a valid token the request is a
  401 and never reaches the NLP service. The user's token and cookies are never forwarded to it.
- **Real answers only.** The NLP service's response is passed back byte for byte with its own status code (including
  `503 not_trained_yet` and `404 word_not_in_vocabulary`) and the header `X-NLP-Proxy: forwarded`. The proxy never
  invents a prediction, a metric or a "not trained" message.
- **Validated input.** Requests are checked in Node against the same limits the Python service enforces, and only
  whitelisted fields are forwarded. Unknown fields are rejected. (A test compares Node's and Python's decisions at every limit.)
- **Clear failures.** Anything that is not an answer from the NLP service carries `"source": "node-proxy"` and a stable
  `code`:

| Status | `code` | Meaning |
|---|---|---|
| 400 | `invalid_request`, `invalid_json` | refused here, never forwarded |
| 401 | (existing auth message) | not logged in |
| 404 | `not_found` | unknown `/api/nlp` route, or wrong HTTP method |
| 413 / 415 | `payload_too_large`, `unsupported_media_type` | body larger than `NLP_MAX_BODY_KB`; not `application/json` |
| 503 | `nlp_service_not_configured` | `NLP_SERVICE_URL` is not set or invalid |
| 503 | `nlp_service_unavailable` | the service cannot be reached (stopped, wrong address, connection dropped) |
| 504 | `nlp_service_timeout` | no complete answer within the timeout |
| 502 | `nlp_service_bad_response` | the service answered with invalid JSON, or more than 5 MB |

A **503 from the NLP service itself** (for example `{"detail": {"status": "not_trained_yet", ...}}`) is forwarded
unchanged and has no `source` field and a `X-NLP-Proxy: forwarded` header: that is how a client tells "the model is not
trained" from "the proxy cannot reach the service".

## Routes

All routes need `Authorization: Bearer <token>` from `/api/auth/login`. `slow` routes use `NLP_SLOW_TIMEOUT_MS`.

| Method | Route | Forwards to | Notes |
|---|---|---|---|
| GET | `/api/nlp` | (answered by Node) | which routes exist and whether `NLP_SERVICE_URL` is configured |
| GET | `/api/nlp/health` | `/health` | what is installed, trained and present on the service |
| POST | `/api/nlp/preprocess` | `/preprocess` | body `{text, normalizer?, remove_stopwords?}` |
| POST | `/api/nlp/classify/tfidf` | `/classify/tfidf` | body `{texts: [1..32 strings of 1..5000 chars], top_k?: 1..10}` |
| POST | `/api/nlp/classify/lstm` | `/classify/lstm` | same body |
| POST | `/api/nlp/classify/bilstm` | `/classify/bilstm` | same body |
| POST | `/api/nlp/classify/bert` | `/classify/bert` | same body; slow |
| GET | `/api/nlp/models/tfidf/metrics` | `/models/tfidf/metrics` | real metrics of the last training run |
| GET | `/api/nlp/models/lstm/status` | `/models/lstm/status` | |
| GET | `/api/nlp/models/lstm/metrics` | `/models/lstm/metrics` | |
| GET | `/api/nlp/models/bilstm/status` | `/models/bilstm/status` | |
| GET | `/api/nlp/models/bilstm/metrics` | `/models/bilstm/metrics` | |
| GET | `/api/nlp/models/bert/status` | `/models/bert/status` | |
| GET | `/api/nlp/models/bert/metrics` | `/models/bert/metrics` | |
| GET | `/api/nlp/models/word2vec/status` | `/models/word2vec/status` | |
| GET | `/api/nlp/models/word2vec/similar` | `/models/word2vec/similar` | `?word=&top_k=&variant=` |
| GET | `/api/nlp/models/word2vec/similarity` | `/models/word2vec/similarity` | `?word_a=&word_b=&variant=` |
| GET | `/api/nlp/models/word2vec/analogy` | `/models/word2vec/analogy` | `?a=&b=&c=&top_k=&variant=` |
| GET | `/api/nlp/models/word2vec/vector` | `/models/word2vec/vector` | `?word=&variant=` |
| POST | `/api/nlp/models/word2vec/visualize` | `/models/word2vec/visualize` | PCA / t-SNE coordinates; slow |
| GET | `/api/nlp/models/comparison` | `/models/comparison` | `?n_bootstrap=`; TF-IDF vs LSTM vs BiLSTM vs BERT; slow |
| GET | `/api/nlp/models/comparison/rows` | `/models/comparison/rows` | flat rows for tables and charts; slow |

The meaning of each response is documented in `nlp_service/README.md`. Models that are not trained yet answer honestly
(`status`: `not_trained_yet`); there is no data in this repository to train them with, so expect that until you run the
training commands in `nlp_service/README.md`.

## Configuration (`backend/.env`)

Copy `backend/.env.example` to `backend/.env`. The new variables:

| Variable | Default | Meaning |
|---|---|---|
| `NLP_SERVICE_URL` | unset | base URL of the NLP service, e.g. `http://127.0.0.1:8001`. Unset = NLP features off (503), everything else works. |
| `NLP_TIMEOUT_MS` | `15000` | timeout for normal calls |
| `NLP_SLOW_TIMEOUT_MS` | `60000` | timeout for BERT, t-SNE and comparison |
| `NLP_MAX_BODY_KB` | `1024` | largest JSON body `/api/nlp` accepts (other routes keep the 100 KB default) |

## Running both services locally

You need **Node 18 or newer** (the proxy uses the built-in `fetch`) and **Python 3.11 or 3.12** for the NLP service.

**Terminal 1: the NLP service** (from the repository root)

```powershell
# Windows PowerShell
cd nlp_service
py -3.12 -m venv .venv-nlp
.\.venv-nlp\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings numpy pandas scikit-learn nltk joblib gensim
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

```bash
# macOS / Linux
cd nlp_service
python3.12 -m venv .venv-nlp && source .venv-nlp/bin/activate
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings numpy pandas scikit-learn nltk joblib gensim
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

That set covers preprocessing, TF-IDF, Word2Vec and the comparison. LSTM, BiLSTM and BERT additionally need
`pip install -r requirements.txt` (PyTorch and Transformers are large downloads). The service starts without them.

**Terminal 2: the backend**

```powershell
cd backend
copy .env.example .env      # macOS/Linux: cp .env.example .env   then edit the values
npm install
npm start
```

The backend log should say `[nlp] NLP service: http://127.0.0.1:8001 ...`. If it says "not configured", `NLP_SERVICE_URL`
is missing from `backend/.env`.

**Try it** (the frontend does not call these yet):

```bash
TOKEN=$(curl -s localhost:5000/api/auth/login -H 'content-type: application/json' \
        -d '{"name":"YourName","password":"yourpassword"}' | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -s localhost:5000/api/nlp/models/word2vec/status -H "Authorization: Bearer $TOKEN"
curl -s localhost:5000/api/nlp/models/comparison -H "Authorization: Bearer $TOKEN"
```

Both of those answer truthfully with "not trained yet" / "no models" until models have been trained.

## Tests

```bash
cd backend
npm test                        # everything below
node test/run-tests.js regression   # the existing features only
node test/run-tests.js nlp-proxy    # proxy against a programmable stand-in for the NLP service
node test/run-tests.js nlp-real     # proxy against the REAL Python service (trains tiny models in a temp folder)
node test/run-tests.js hygiene      # the repository hygiene guard
```

- `nlp-real` needs Python with the packages listed above. It looks for `nlp_service/.venv-nlp`, then `python3`, `python`,
  `py`; set `NLP_PYTHON` (e.g. `$env:NLP_PYTHON="..\nlp_service\.venv-nlp\Scripts\python.exe"`) to choose another. Without a
  usable Python those tests are **skipped with a stated reason**, not silently passed.
- The tests run a **copy** of the backend in a temp folder with a minimal environment and Groq redirected to a local stub:
  they never touch `backend/govprep-db.json`, `backend/uploads/` or your `.env`, and need no network or Groq key.
- To prove the integration changed nothing else, run the existing-feature tests against your original backend and the
  modified one: `GOVPREP_BACKEND_DIR=/path/to/original/backend node test/run-tests.js regression` (they must pass on both).

## Security notes

- The Python NLP service has **no authentication of its own**. Keep it bound to `127.0.0.1` (the default) or a private
  network; only the Node backend should be able to reach it. Do not expose port 8001 to the internet.
- The deployed app (Render) is not affected: without `NLP_SERVICE_URL` the NLP routes answer 503 and nothing else changes.
  The Python service and its models are not deployed by this repository.
- User text sent for classification is not written to the backend log (only the route and the error code are).
- There is no per-user rate limiting on `/api/nlp`; BERT/LSTM inference is CPU-heavy, so add limiting before exposing it widely.
- Known pre-existing quirk, unrelated to the NLP work: `server.js` cannot start without a `GROQ_API_KEY` (the OpenAI SDK
  throws), although the code suggests it was meant to. It is documented in `backend/.env.example` and left unchanged.

## Repository hygiene

`.gitignore` cannot protect a file that is already tracked or added with `git add -f`, so `scripts/repo-hygiene.js`
checks what git actually tracks. It flags the local database (`govprep-db.json`), `.env` files, virtual environments,
`node_modules`, user uploads, NLP datasets, model weights (`.pt`, `.joblib`, `.model`, ...), files over 25 MB and secrets
inside files (Groq/OpenAI-style keys, AWS keys, private keys, real-looking `JWT_SECRET`).

```bash
node scripts/repo-hygiene.js check                # exit 0 = clean, 1 = problems
node scripts/repo-hygiene.js fix                  # dry run: shows what it would untrack
node scripts/repo-hygiene.js fix --apply          # backs up, then untracks (keeps your working files)
node scripts/repo-hygiene.js install-hook         # optional: block bad commits with a pre-commit hook
```

`fix --apply` is deliberately safe: it first copies every affected file into `.hygiene-backup/<time>/` (git-ignored) and
verifies the copy byte for byte, then runs `git rm --cached`, which only removes the file from git's index. **It never
deletes or edits a file in your working folder**, so your local database and uploads stay exactly as they are. It refuses
to run if the backup folder is not ignored. Secrets inside files are only reported (rotate them). Older commits still hold
any file committed earlier; rewriting history (`git filter-repo`) is a separate, deliberate step this tool does not do.

If `backend/govprep-db.json` is currently tracked in your repository (it was on GitHub), run
`node scripts/repo-hygiene.js fix --apply`, then commit. If real passwords were used for the accounts in it, change them.
