# Phase 10 dataset readiness report

Status: **PASS for subject classification dataset preparation**.

## Source and filtering

- Source: L3Cube-IndicQuest-v2, English subset.
- Raw rows: 3,471.
- Retained after direct subject mapping: 1,784.
- Dropped for unmapped domain (not guessed): 1,687.
- Empty/short text rows dropped: 0.
- Exact duplicate rows dropped from the retained dataset: 0.
- Conflicting-label duplicate rows dropped: 0.

## Class counts

| Subject | Rows |
|---|---:|
| History | 510 |
| Geography | 499 |
| Polity | 450 |
| Science | 325 |
| **Total** | **1,784** |

## Reproducible split

Seed: `11` (selected after the seed-42 split produced two TF-IDF near-duplicate candidates across train/validation; seed 11 places the similar question pairs within the same split). Stratification target: `subject`. Validation size: 15%. Test size: 15%.

| Split | Rows |
|---|---:|
| Train | 1,248 |
| Validation | 268 |
| Test | 268 |
| **Total** | **1,784** |

Quality report: **PASS**. All six exact-duplicate and near-duplicate checks across train/validation/test report zero cross-split matches at cosine similarity threshold 0.90. No classes were removed for insufficient count. Training class imbalance ratio: 1.57:1.

## Scope limitations

1. This dataset supports **four-class subject/domain classification only** (History, Geography, Polity, Science).
2. It has no topic or difficulty targets. Do not claim that topic or difficulty classifiers were trained from this dataset.
3. It is a general India-specific factual QA benchmark, not specifically an RRB NTPC/Railway exam dataset.
4. 1,248 training examples are a small dataset for deep neural networks. LSTM/BiLSTM/BERT results should be treated as controlled low-data experiments, not evidence of broad exam-domain generalisation.
5. The answers are not used as subject-classifier inputs or labels. Only question text and publisher-provided domain labels are used.

## Output files

- `raw/English.csv`: original downloaded source.
- `raw/sources.json`: provenance config used by `training.build_dataset`.
- `public/questions.csv`: canonical labelled dataset.
- `public/questions.build_report.json`: source mapping/drop report.
- `processed/public_train.csv`, `public_val.csv`, `public_test.csv`: reproducible splits.
- `processed/split_manifest.json`: seed, row counts and class counts.
- `processed/report/dataset_statistics.json`: quality report.
- `processed/report/class_distribution.png`: class distribution chart.
