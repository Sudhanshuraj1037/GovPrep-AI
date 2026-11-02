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
5. Record each dataset's name, source URL and licence in `public/SOURCES.md`.

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
