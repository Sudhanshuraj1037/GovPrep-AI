# GovPrep-AI NLP/DL Service  (CSE472: Deep Learning for NLP)

A small Python (FastAPI) service that adds a genuine NLP + Deep Learning layer to
GovPrep-AI. It sits **beside** the existing app; nothing in `backend/` or
`frontend/` depends on it yet.

```
React  ->  Node/Express backend  ->  nlp_service (this folder)  ->  NLP + DL models
                 |
                 +-> Groq (generative AI: question generation, study assistant)
```

## Deep Learning NLP vs Generative AI (for the viva)

| Deep Learning NLP (this service, trained by us) | Generative AI (existing Node backend) |
|---|---|
| Preprocessing, TF-IDF, Word2Vec | Question generation (Groq / Llama) |
| LSTM, BiLSTM, fine-tuned BERT classifiers | AI study assistant (Groq / Llama) |
| Evaluation + model comparison | English -> Hindi output via the prompt |

We **train** the classifiers above. We do **not** train any LLM; Groq is an external,
pre-trained API.

## Status

| Phase | Item | State |
|---|---|---|
| 1 | Service skeleton + `GET /health` | done |
| 2 | Preprocessing + TF-IDF baseline + metrics + inference | **code done and tested; model NOT trained yet (no dataset present)** |
| 3 | Word2Vec (CBOW + Skip-Gram), similarity / analogy / vectors, PCA + t-SNE data | **code done and tested; model NOT trained yet (no dataset present)** |
| 4 | PyTorch LSTM classifier (subject / topic / difficulty) + metrics + API | **code done and tested; model NOT trained yet (no dataset present)** |
| 5 | PyTorch BiLSTM classifier + metrics + API + cross-model metrics contract | **code done and tested; model NOT trained yet (no dataset present)** |
| 6 | BERT fine-tuning (Transformers + PyTorch) + metrics + API, on the shared metrics contract | **code done and tested; model NOT trained yet (no dataset, and no pretrained weights downloadable in the dev sandbox)** |
| 7 | Model comparison (TF-IDF vs LSTM vs BiLSTM vs BERT): aligned tables, rankings, bootstrap intervals, report export, API | **code done and tested; nothing to compare until the models are trained** |
| 8+ | Node proxy, React UI, PDF pipeline, final regression | not started |

**No trained model, embedding or metric ships with the repo.** They are produced only by
running the training command on real data. Until then every classification
and embedding endpoint answers `503 not_trained_yet`.

`GET /health` reports what is *really* installed, trained and present on disk. A model
is shown as trained only if its training script wrote its marker file
(`artifacts/<model>/metrics.json`, or `word2vec.model` for Word2Vec). No metric is
hard-coded anywhere.

## Setup

Use **Python 3.11 or 3.12** (PyTorch / gensim wheels for 3.14 may not exist yet). `gensim` (Phase 3),
`torch` (Phase 4) and `transformers` (Phase 6) are new: run `pip install -r requirements.txt` again in your `.venv-nlp`. PyTorch is a large
download; a CPU build is enough for development (see the comment in `requirements.txt`).
Do not reuse the repo-level `.venv`; create a separate one.

Windows (PowerShell):
```powershell
cd nlp_service
py -3.12 -m venv .venv-nlp
.\.venv-nlp\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m uvicorn app.main:app --port 8001 --reload
```

macOS / Linux:
```bash
cd nlp_service
python3.12 -m venv .venv-nlp && source .venv-nlp/bin/activate
pip install -r requirements-dev.txt
python -m uvicorn app.main:app --port 8001 --reload
```

Then open http://127.0.0.1:8001/health or http://127.0.0.1:8001/docs.

Run the tests: `python -m pytest -q`  (uses synthetic data in temp folders; never writes to `data/` or `artifacts/`)

## Phase 2 workflow: from CSV to trained TF-IDF model

CSV schema (every dataset): `text, subject, topic, difficulty` (topic and difficulty may be blank;
subject must be one of the app's subjects, difficulty one of easy/medium/hard).

```bash
# 1. validate, de-duplicate and split (stratified on subject, seed 42) -> data/processed/
python -m training.prepare_data --input data/public/<labelled>.csv \
                                --heldout data/pdf_heldout/<govprep_pdfs>.csv   # heldout optional

# 2. train TF-IDF + Logistic Regression for subject / topic / difficulty
python -m training.train_tfidf          # -> artifacts/tfidf/*.joblib + metrics.json (last)
```

If `data/processed/public_train.csv` is missing, step 2 prints `NOT TRAINED YET`, writes
nothing and exits with code 2.

What the pipeline does:
- **Preprocessing** (`app/preprocessing.py`): Unicode clean-up, URL/e-mail/page-marker removal,
  tokenisation (digits and Devanagari kept), stop-word removal (negations kept), Porter stemming
  (or WordNet lemmatisation, or none). The exact config is saved inside every model.
- **TF-IDF**: word 1-2 grams, sublinear TF, fitted on the train split only.
- **Classifier**: Logistic Regression (class-balanced), one per target; C chosen on the validation split.
- **Evaluation** (`app/metrics.py`, reused by later models): accuracy, macro/weighted precision,
  recall, F1, per-class scores, confusion matrix, plus a majority-class baseline for context.
  Reported for validation, the public test split and (if present) the PDF held-out set.
- English-only stop-words/stemming. Hindi tokens are preserved but not otherwise processed.

### API (Phase 2)

| Method | Path | Purpose |
|---|---|---|
| POST | `/preprocess` | every preprocessing stage for a text (viva demo) |
| POST | `/classify/tfidf` | `{"texts": [...], "top_k": 3}` -> subject/topic/difficulty per text; `503` if not trained |
| GET | `/models/tfidf/metrics` | the real `metrics.json`; `404` if not trained |
| GET | `/health` | status, installed packages, which models/targets are really trained |

The Node/Express proxy to these endpoints arrives in Phase 9; nothing in `backend/` calls them yet.

## Phase 4 workflow: LSTM text classification (PyTorch)

```bash
# CPU-friendly development run (the defaults: 1 layer, hidden 128, embedding 100, <= 20 epochs)
python -m training.train_lstm

# Real run on a GPU / Google Colab: bigger model, more epochs
python -m training.train_lstm --device auto --embedding-dim 200 --hidden-dim 256 \
       --num-layers 2 --epochs 40 --batch-size 128 --patience 6

# Optional: start from the Phase 3 Word2Vec vectors (refused unless Word2Vec was trained on the
# current public_train.csv)
python -m training.train_lstm --init-embeddings word2vec
```

Run `training.prepare_data` first (see Phase 2). Every hyper-parameter is a flag (`--help`).
Progress is printed per epoch; results go to `artifacts/lstm/<target>.pt` and, last, `artifacts/lstm/metrics.json`.

- **Model:** tokens -> vocabulary ids -> pad/truncate -> `Embedding` -> `Dropout` -> `LSTM` (packed sequences, so padding
  never reaches the network) -> final hidden state -> `Dropout` -> `Linear`. One independent model per target
  (subject, topic, difficulty), like the TF-IDF baseline, so the two are directly comparable in Phase 8.
- **Data protocol (same as Phase 2):** reads the processed splits (stratified, seed 42) and never re-splits. The vocabulary
  comes from the TRAIN split only. The VALIDATION split drives early stopping and best-epoch selection. The public TEST
  split and the PDF held-out set are only scored after training; tests prove they cannot influence weights, vocabulary
  or epoch choice.
- **Imbalance:** class-weighted cross-entropy (`n / (k * count)`, the same formula as scikit-learn's "balanced"); `--no-class-weights` to disable.
- **Preprocessing:** the shared `TextPreprocessor` but keeping stop-words (word order is what an LSTM models);
  `--remove-stopwords` / `--normalizer` to change. The config is saved inside every checkpoint.
- **Metrics (`app/metrics.py`, identical code for every model):** accuracy, macro and weighted precision / recall / F1,
  per-class scores, confusion matrix and a majority-class baseline, for validation, test and (if present) PDF held-out.
  The per-epoch loss / validation history, best epoch and the SHA-256 of the split files are recorded too.
- **Reproducible:** seeded (default 42), deterministic algorithms requested; tests check identical weights across runs.
  Bit-identical results across *different hardware* (CPU vs GPU) are not guaranteed.
- **Honest limits:** a text whose tokens are all out of vocabulary gets `label: null` and a warning, never a guess. Training
  prints `NOT TRAINED YET` and writes nothing if data is missing or nothing is trainable (exit code 2; exit code 3 = PyTorch missing).
- **PyTorch is optional for the rest of the service:** `/health` and every other endpoint work without it; only training
  and `/classify/lstm` need it (the latter answers `503 dependency_missing`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/models/lstm/status` | always 200; `trained` flag, targets, hyper-parameters, headline scores, `torch_installed` |
| GET | `/models/lstm/metrics` | the real `metrics.json` (+ `weights_available`); `404` if never trained |
| POST | `/classify/lstm` | `{"texts": [...], "top_k": 3}` -> subject/topic/difficulty per text; `503 not_trained_yet` / `dependency_missing` |

`/health` marks the LSTM trained only when `artifacts/lstm/metrics.json` AND at least one `.pt` checkpoint exist
(checkpoints are git-ignored, so a fresh clone with only `metrics.json` correctly reads "not trained").

## Phase 7 workflow: comparing TF-IDF, LSTM, BiLSTM and BERT

```bash
python -m training.compare_models                 # after the models are trained; 1000 bootstrap resamples
python -m training.compare_models --bootstrap 0   # no intervals
```

Reads the REAL `artifacts/<model>/metrics.json` files (whichever exist and satisfy the shared contract) and writes
`artifacts/comparison/comparison.json`, `comparison.md` (tables + caveats, ready for the report) and `comparison.csv`.
Needs neither PyTorch nor Transformers. With fewer than two models it prints why, writes nothing and exits with code 2.

| Method | Path | Purpose |
|---|---|---|
| GET | `/models/comparison?n_bootstrap=1000` | aligned tables per target and split, ranking, intervals, comparability checks, test-vs-PDF gap |
| GET | `/models/comparison/rows` | the same numbers as flat rows (one per target / split / model) for tables and charts |

Always HTTP 200 with `status` = `no_models_trained` | `single_model` | `ok`, so a dashboard can show an honest empty state.

How to read it (and what it deliberately does not claim):
- **Ranking is by macro-F1**, the fair choice on imbalanced classes (a test covers a case where accuracy would rank the other way). Ties
  share a rank and are reported as a tie, never as a winner.
- **Intervals** are 95% percentile bootstrap intervals computed from each model's stored confusion matrix (accuracy, macro-F1, weighted-F1).
  The matrix determines these metrics exactly (a test recomputes every real metric from it), and resampling examples is the same as
  resampling matrix cells. They are UNPAIRED: per-example predictions are not stored, so no paired test (such as McNemar) is claimed.
- **"Separated from runner-up"** is a conservative rule: the best model's interval must lie entirely above the runner-up's. Overlapping
  intervals mean the data do not establish a difference, whatever the point estimates say. With the few hundred test rows a small
  course dataset gives, expect many overlaps; report them as such.
- **Comparable only if identical:** models are compared on a (target, split) only if they were scored on the same rows and classes;
  others are listed under `excluded`. The split identity (same split-file checksums / same split manifest) is verified and a mismatch
  is raised as a warning that invalidates the comparison.
- **`pdf_heldout`** is reported separately, with the test-vs-PDF macro-F1 gap per model (domain shift on real GovPrep-AI text).
- Parameter counts are shown but not directly comparable across families; a BiLSTM has about twice the recurrent parameters of an LSTM
  trained with the same flags.
- Nothing is estimated: a model that has not been trained is simply absent (`models_missing`), and a model whose `metrics.json` breaks
  the contract is listed under `problems` with its rows withheld.

## Phase 6 workflow: BERT fine-tuning (Hugging Face Transformers + PyTorch)

A genuine **pretrained** BERT is fine-tuned for subject / topic / difficulty. It is loaded with `from_pretrained`
and is never trained from scratch (a test forbids `from_config` during training).

**Recommended environment: Google Colab with a GPU.** A CPU run works but is slow.

```bash
# Real training (GPU / Colab):
python -m training.train_bert --device auto --pretrained bert-base-uncased \
       --epochs 4 --batch-size 32 --max-len 128 --lr 2e-5 --patience 2

# CPU development run (lighter model, shorter sequences; slow on real data):
python -m training.train_bert --device cpu --pretrained distilbert-base-uncased \
       --max-len 64 --epochs 2 --batch-size 8

# Hindi / multilingual text:
python -m training.train_bert --pretrained bert-base-multilingual-cased
```

`--pretrained` is any Hugging Face id (downloaded on first use: **internet required**) or a local directory holding a
saved model + tokenizer. `--freeze-encoder` trains only the classification head (a cheap CPU smoke test, recorded
in `metrics.json` as not full fine-tuning). Run `training.prepare_data` first (Phase 2); every flag is in `--help`.

### Google Colab, step by step

1. Runtime > Change runtime type > **GPU** (a T4 is enough for bert-base).
2. Upload the `nlp_service` folder (or its zip) and the **already prepared** `data/processed/` folder from your
   machine, so the split is byte-identical to the one the other models used (the SHA-256 of every split file is stored
   in each `metrics.json`, so any difference would be visible).
   ```
   %cd nlp_service
   !pip install -q -r requirements.txt        # Colab already has torch; this adds transformers, gensim, ...
   !python -m training.train_bert --device auto --pretrained bert-base-uncased --epochs 4 --batch-size 32
   !zip -qr bert_artifacts.zip artifacts/bert
   ```
3. Download `bert_artifacts.zip` and unzip it into your local `nlp_service/artifacts/` so you get `artifacts/bert/`
   (`<target>.pt` x3, `tokenizer/`, `metrics.json`). The local service then serves `/classify/bert` on CPU with no internet.
4. If the GPU runs out of memory lower `--batch-size` (or `--max-len`). Mixed precision and gradient accumulation are
   not implemented.

**Disk:** a bert-base checkpoint is about 440 MB per target (about 1.3 GB for all three; use `--targets subject` to save
one). `.pt` files are git-ignored; `metrics.json` and the small `tokenizer/` folder are not.

### What is identical to TF-IDF / LSTM / BiLSTM

Same processed splits and seed 42, never re-split; one independent model per target with the same sorted class order;
validation macro-F1 drives best-epoch selection and early stopping; test and PDF held-out are only scored afterwards (tests
prove that replacing them changes neither weights nor history); class-weighted loss (`n / (k * count)`); the same
`app/metrics.py`. `metrics.json` satisfies the shared `app/model_metrics.py` contract (validated in the tests), so BERT
appears as one more row set next to the other three: same 18 fields, same classes, same rows scored.

### What is inherent to BERT (documented, not hidden)

- **Tokenizer and vocabulary come from the pretrained checkpoint**, not from our data (a test checks the saved tokenizer is
  byte-identical to the pretrained one). No stop-word removal or stemming; only the shared noise removal (URLs, e-mails,
  "Page 3 of 70", control characters) plus Unicode NFKC. Case handling is the tokenizer's own.
- **Far more parameters** (recorded in `metrics.json`); fine-tuning uses AdamW, linear warm-up then decay, gradient clipping.
- A tokenizer that does not belong to the model (for example a folder with weights but no tokenizer files, which some
  transformers versions would silently replace by an empty one) is detected and refused.
- Artifacts are self-contained: `artifacts/bert/tokenizer/` is saved with the model, so inference needs no network and no
  access to the original pretrained folder (tests check both).
- **Dependencies are optional for the rest of the service.** `app.main` imports neither torch nor transformers (a test checks
  it); `/health`, `/models/bert/status` and `/models/bert/metrics` work without them and `/classify/bert` answers
  `503 dependency_missing`.

### Honest limits of what was verified during development

Hugging Face was not reachable from the development sandbox, so **no real pretrained BERT was ever downloaded or run**. All
tests fine-tune a tiny stand-in (random weights, 2 layers, hidden size 32) that the tests build locally and load through the
same `from_pretrained` code path. That proves the machinery (tokenizer, attention masks, loop, checkpoints, no leakage,
metrics, contract) but says nothing about real BERT quality or about the hub download itself. It was also only run with
transformers 5.19.0 and torch 2.14.1; the `transformers>=4.40` floor in `requirements.txt` was not exercised. Those are exercised the first
time you run the command above on a machine with internet access.

| Method | Path | Purpose |
|---|---|---|
| GET | `/models/bert/status` | always 200; `trained`, targets, hyper-parameters, headline scores, `torch_installed` / `transformers_installed` |
| GET | `/models/bert/metrics` | the real `metrics.json` (+ `weights_available`); `404` if never trained |
| POST | `/classify/bert` | `{"texts": [...], "top_k": 3}` -> subject/topic/difficulty per text; `503 not_trained_yet` / `dependency_missing` |

## Phase 5 workflow: BiLSTM text classification (PyTorch)

```bash
python -m training.train_bilstm            # CPU development run; same flags as train_lstm (see --help)
python -m training.train_bilstm --device auto --embedding-dim 200 --hidden-dim 256 --num-layers 2 \
       --epochs 40 --batch-size 128 --patience 6          # Colab / GPU
python -m training.train_bilstm --init-embeddings word2vec
```

The BiLSTM is a bidirectional extension of the LSTM, built on the SAME code (`app/lstm.py`, selected with
`BILSTM_SPEC`; `app/bilstm.py` is a thin façade). Both models therefore use identical data, the Phase 2
splits (seed 42), preprocessing, vocabulary, class weights, training loop, early stopping on validation
macro-F1, held-out discipline, checkpoint format and metrics, and differ ONLY in the architecture:

- **Architecture:** `nn.LSTM(bidirectional=True)`. One LSTM reads left-to-right, one right-to-left; the classifier
  receives `[forward state after the LAST token ; backward state after the FIRST token]` (`2 * hidden_dim` features).
  Packed sequences make the backward direction start at each text's true last token, never at padding.
- **Fair-comparison caveat:** `--hidden-dim` is per direction, so with the same flags a BiLSTM has about twice the
  recurrent parameters of the LSTM. Parameter counts are saved in `metrics.json` (`targets.<t>.model.trainable_parameters`).
- **Artifacts:** `artifacts/bilstm/<target>.pt` and, last, `artifacts/bilstm/metrics.json`. Checkpoints carry
  `arch: "bilstm"`; an LSTM checkpoint cannot be loaded as a BiLSTM or the other way round.
- **Endpoints** (equivalent to the LSTM's, built by the same router factory): `GET /models/bilstm/status`,
  `GET /models/bilstm/metrics`, `POST /classify/bilstm`.
- **Health:** the BiLSTM is trained only when `artifacts/bilstm/metrics.json` AND a `.pt` checkpoint exist; it is independent of the LSTM.
  Every model in the registry now follows the same marker-plus-weights rule (a test enforces it for all of them).

### One metrics contract for every model (for the comparison dashboard)

TF-IDF, LSTM, BiLSTM and BERT all score each split with the same `app/metrics.py`. `app/model_metrics.py` makes
that an explicit contract: `validate_metrics()` checks a `metrics.json` (structure plus consistency: confusion-matrix totals,
accuracy = trace / n, per-class supports, a shared sorted class order) and `collect()` returns one flat row per
(model, target, split) with the same fields for every model:

`model, target, split, n, accuracy, precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted,
majority_class_accuracy, n_classes, n_train, trainable_parameters, trained_at, seed, checkpoint_available`

A model with a broken file is reported under `problems` and its rows are withheld; nothing is ever estimated or filled in.
BERT writes a conforming `metrics.json` and saves `<target>.pt` checkpoints like the LSTM / BiLSTM (tests validate all four
models together).

## Phase 3 workflow: Word2Vec embeddings

```bash
python -m training.train_word2vec        # after training.prepare_data (see Phase 2)
# -> artifacts/word2vec/{cbow,skipgram}.model, vocab_stats.json and, last, metadata.json
```

- **Corpus:** `data/processed/public_train.csv` ONLY. Validation, test and the PDF held-out set are never
  part of the training corpus or the vocabulary (tests prove this with sentinel words). Afterwards the
  script *measures* how many tokens of val/test/PDF the vocabulary covers; this is never fed back into training.
- **Variants:** CBOW and Skip-Gram, both trained by default (`--variants cbow skipgram`). The API defaults to Skip-Gram.
- **Preprocessing:** the shared `TextPreprocessor`, but without stemming (readable neighbour lists).
- **Reproducible:** seed 42, one worker, stable hash; tests check identical vectors across separate processes.
- **Small corpora need more passes:** the default is 30 epochs. A few hundred short questions are under-trained
  at that setting; raise `--epochs` (e.g. 100) and check the neighbour lists the script prints.
- **No benchmark score:** Word2Vec is unsupervised, so there is no accuracy/F1. `metadata.json` holds descriptive
  statistics only (corpus size, vocabulary size, token coverage, CBOW-vs-Skip-Gram neighbour overlap).
  Analogy answers carry a caveat and no analogy accuracy is claimed.

| Method | Path | Purpose |
|---|---|---|
| GET | `/models/word2vec/status` | always 200; `trained` flag, variants, corpus, vocabulary, coverage |
| GET | `/models/word2vec/similar?word=&top_k=&variant=` | nearest words (cosine) |
| GET | `/models/word2vec/similarity?word_a=&word_b=&variant=` | cosine similarity of two words |
| GET | `/models/word2vec/analogy?a=&b=&c=&top_k=&variant=` | a : b :: c : ? (b - a + c) |
| GET | `/models/word2vec/vector?word=&variant=` | the embedding vector |
| POST | `/models/word2vec/visualize` | PCA or t-SNE coordinates as JSON for a scatter plot |

`POST /models/word2vec/visualize` body (all fields optional):
`{"method": "pca"|"tsne", "variant": "cbow"|"skipgram", "words": [...], "top_n": 100,
"neighbors_per_word": 0, "perplexity": 30}`. With no `words` the `top_n` most frequent words are plotted. The reply has
`points: [{word, x, y, count, source, dominant_subject, dominant_share}]` plus `explained_variance_ratio` (PCA) or
`kl_divergence` (t-SNE), `missing_words`, `subjects` (legend) and `notes`. `dominant_subject` colours the chart:
it is the subject whose TRAIN questions contain the word most often relative to class size, and a low
`dominant_share` marks a general word that appears in every subject. t-SNE distances between clusters are not meaningful.

Errors: `503 not_trained_yet`; `404 word_not_in_vocabulary` (lists the words; stop-words, multi-word input and
unseen words all land here); `404 variant_not_trained`; `422` for invalid parameters or too few points
(PCA needs 3 words, t-SNE 5).

Security: TF-IDF and Word2Vec models are saved with joblib / gensim (pickle): only load artifacts you trained
yourself. LSTM checkpoints are plain tensors and are loaded with `torch.load(..., weights_only=True)`, which refuses
pickled code (a test proves it).

## Ports

| Service | Port |
|---|---|
| React dev server | 3000 |
| Node/Express backend | 5000 |
| `rag_service` (older, unused) | 8000 |
| **nlp_service** | **8001** |

## Layout

```
nlp_service/
  app/            FastAPI app: main, config, labels, registry (status),
                  preprocessing, data (schema/split), metrics, tfidf, api_nlp,
                  word2vec, api_word2vec, lstm (LSTM + BiLSTM), api_lstm (+ router factory), bilstm, api_bilstm,
                  bert, api_bert, model_metrics, comparison, api_comparison
  training/       CLI: prepare_data.py, train_tfidf.py, train_word2vec.py, train_lstm.py, train_bilstm.py,
                  train_bert.py, compare_models.py
  data/           datasets (see data/README.md)
  artifacts/      saved models + metrics.json per model
  tests/
```
