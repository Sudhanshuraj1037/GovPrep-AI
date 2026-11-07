# Data policy (read before adding any dataset)

Two sources, with strictly different roles:

| Source | Folder | Role |
|---|---|---|
| Public labelled question dataset | `public/` | **Train / validation / test** (stratified split, fixed seed) |
| Text and questions from GovPrep-AI PDFs | `pdf_heldout/` | **Final evaluation only** |

Rules that keep the results honest:

1. The PDF-derived set is **never** used for training, hyper-parameter tuning, vocabulary
   building, Word2Vec training or early stopping. It is scored once, at the end.
   (Word2Vec reads only `processed/public_train.csv`; it merely *measures* token coverage on the others.)
2. Every model (TF-IDF, LSTM, BiLSTM, BERT) uses the **same** public train/val/test split.
3. Report public-test and PDF-held-out results **separately**. A gap between them is
   expected (domain shift) and is a finding, not something to hide.
4. Labels for PDF chunks come from the subject the user chose at upload, not from an LLM.
   Difficulty labels in the existing app come from Groq, so a difficulty model learns to
   approximate an LLM's judgement; say so when presenting it.
5. Record each dataset's name, source URL and licence in `SOURCES.md`.

Processed files are written by `python -m training.prepare_data` (not present until you run it):
`processed/public_train.csv`, `public_val.csv`, `public_test.csv`, `pdf_heldout.csv`
and `split_manifest.json` (seed, input-file SHA-256, class counts, dropped rows, leak removals),
each CSV with columns `text, subject, topic, difficulty`.

How the code enforces the rules above:
- stratified on `subject`, fixed seed 42, independent of input row order;
- exact duplicate texts removed; duplicates with conflicting subjects are dropped entirely;
- classes with fewer than 5 rows are dropped and reported (never silently);
- held-out rows whose text also appears in a public split are removed and counted;
- a test fails if training or tuning ever depends on test / held-out data.

## Starter dataset: L3Cube-IndicQuest-v2 (English subset)

A ready-to-use config is provided at `data/raw/sources.example.json`. Download only the English CSV and then run the build command below. This dataset supplies a genuine `Domain` label, but it is a general India-specific factual QA benchmark rather than a railway-exam-only corpus. The config keeps only the direct existing label matches (History, Geography, Science, and Political Science -> Polity); other domains are reported and dropped. Topic and difficulty stay blank because they are not supplied.

## Phase 10 workflow (raw files -> verified splits)

Run from `nlp_service/` (Windows: activate `.venv-nlp` first). No step invents labels.

```
# 0. From nlp_service/, download the English subset (CC BY 4.0; attribution is required):
#    New-Item -ItemType Directory -Force data/raw
#    Invoke-WebRequest -Uri "https://huggingface.co/datasets/l3cube-pune/IndicQuest-v2/resolve/main/English.csv" -OutFile "data/raw/English.csv"
# 1. Copy data/raw/sources.example.json to data/raw/sources.json, then:
python -m training.build_dataset --config data/raw/sources.json --out data/public/questions.csv
#    -> questions.csv (text, subject, topic, difficulty, source) + questions.build_report.json
#    Subjects with no entry in labels.SUBJECT_ALIASES are dropped and listed, never guessed.
# 2. Split. By default, use stratified splitting:
python -m training.prepare_data --input data/public/questions.csv
#    `source` records corpus provenance. Use --group-by source ONLY when every source entry
#    represents exactly one document/paper. If one source is a large corpus, do NOT group by it.
#    If the raw data has a document/paper ID, preserve it as a separate group column and group by that.
# 3. Statistics, quality checks, class-distribution chart:
python -m training.dataset_report        # -> data/processed/report/
```
`dataset_report` exits non-zero if exact or near-duplicate leakage is found. Do not train until it prints `PASS`.
It also warns when the training set is too small for the deep models to mean anything.

Open decision: the app has no "Computer Knowledge" subject (see `labels.py`). Rows labelled that way are dropped
and reported until you decide to add it to `SUBJECTS` and the frontend dropdown.
