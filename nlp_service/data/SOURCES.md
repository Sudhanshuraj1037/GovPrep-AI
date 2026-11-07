# Dataset sources and licences

## L3Cube-IndicQuest-v2 (English subset)

- Dataset: https://huggingface.co/datasets/l3cube-pune/IndicQuest-v2
- English CSV: https://huggingface.co/datasets/l3cube-pune/IndicQuest-v2/resolve/main/English.csv
- Licence shown on the dataset publisher's page: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/).
- Retrieved: 2026-11-03.
- Original English subset: 3,471 question-answer pairs across nine domains.
- Original columns: `Question`, `Answer`, `Domain`.
- Dataset scope: India-specific factual-knowledge QA, curated from educational curricula, competitive-examination materials and domain references. It is not exclusively a Railway/RRB/SSC previous-year-question dataset.

### Paper citation

Rinit Jain, Tirthraj Mahajan, Advait Joshi, and Raviraj Joshi. 2026. *L3Cube-IndicQuest v2: A Large-Scale Multilingual Benchmark for Evaluating Factual Knowledge of Large Language Models Across Indic Languages*. arXiv:2608.15535. https://arxiv.org/abs/2608.15535

### Subject mapping used in this project

The classifier uses the question text and the publisher-supplied `Domain` as its target. Only labels with direct equivalents in the existing GovPrep-AI subject taxonomy are retained:

| Publisher `Domain` | GovPrep-AI subject | Rows retained |
|---|---|---:|
| History | History | 510 |
| Geography | Geography | 499 |
| Political Science | Polity | 450 |
| Science | Science | 325 |
| **Total retained** | | **1,784** |

The following rows are dropped rather than assigned guessed labels: Culture (534), Law (494), Sports (273), Art (233), and Commercial Studies (153), totalling 1,687 rows.

The source does not supply topic or difficulty labels. Those columns are intentionally blank and must not be used to claim topic- or difficulty-classification training.

### Attribution and changes

This project redistributes the English CSV under the CC BY 4.0 licence. Attribution is given to L3Cube-Pune and the paper above. Changes made for this project: rows were filtered to four directly mapped domains; `Question` was renamed to `text`; `Domain` was mapped to the existing subject taxonomy; `topic` and `difficulty` columns were added blank; and the `Answer` column was retained only in the raw source file, not the subject-classification training table.

Licence terms: https://creativecommons.org/licenses/by/4.0/.
