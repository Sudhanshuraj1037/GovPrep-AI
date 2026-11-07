# Model comparison: tfidf vs lstm vs bilstm vs bert

Models compared: tfidf, lstm, bilstm, bert. Not trained (absent): none.
Ranking metric: macro-F1. Intervals: 95% percentile bootstrap, unpaired (1000 resamples, seed 42).

## subject - val (n=268, 4 classes, majority-class accuracy 0.2836)

| rank | model | accuracy | macro-F1 | weighted-F1 | macro precision | macro recall | trainable params |
|---:|---|---|---|---|---|---|---:|
| 1 | bert | 0.9515 [0.925, 0.974] | 0.9507 [0.924, 0.975] | 0.9516 [0.926, 0.974] | 0.9569 | 0.9465 | 109,485,316 |
| 2 | tfidf | 0.9104 [0.877, 0.940] | 0.9114 [0.878, 0.941] | 0.9102 [0.877, 0.940] | 0.9109 | 0.9132 | - |
| 3 | bilstm | 0.8731 [0.832, 0.910] | 0.8724 [0.830, 0.911] | 0.8731 [0.833, 0.911] | 0.8723 | 0.8761 | 118,514 |
| 4 | lstm | 0.8582 [0.817, 0.896] | 0.8558 [0.810, 0.895] | 0.8588 [0.817, 0.896] | 0.8579 | 0.8548 | 107,634 |

Best by macro-F1: **bert** (intervals overlap: no difference is established).

## subject - test (n=268, 4 classes, majority-class accuracy 0.2873)

| rank | model | accuracy | macro-F1 | weighted-F1 | macro precision | macro recall | trainable params |
|---:|---|---|---|---|---|---|---:|
| 1 | bert | 0.9328 [0.903, 0.963] | 0.9324 [0.900, 0.961] | 0.9329 [0.903, 0.963] | 0.9388 | 0.9278 | 109,485,316 |
| 2 | bilstm | 0.8694 [0.828, 0.907] | 0.8690 [0.827, 0.906] | 0.8695 [0.829, 0.907] | 0.8699 | 0.8683 | 118,514 |
| 3 | tfidf | 0.8657 [0.821, 0.907] | 0.8594 [0.811, 0.900] | 0.8644 [0.818, 0.905] | 0.8667 | 0.8561 | - |
| 4 | lstm | 0.8358 [0.791, 0.877] | 0.8331 [0.784, 0.878] | 0.8352 [0.788, 0.877] | 0.8355 | 0.8317 | 107,634 |

Best by macro-F1: **bert** (intervals overlap: no difference is established).

## Notes
- Every number comes from a real metrics.json; models that are not trained are absent, never estimated.
- Intervals are unpaired bootstrap intervals computed from each model's confusion matrix. 'Separated' means the best model's interval lies entirely above the runner-up's (a conservative rule); overlapping intervals mean the data do not establish a difference. Per-example predictions are not stored, so no paired test is claimed.
- Models are compared on a (target, split) only if they were scored on the same rows and classes.
- pdf_heldout is the domain-shift evaluation (GovPrep-AI PDF text, never used for training or tuning). A gap between test and pdf_heldout is a finding, not an error.
- trainable_parameters are not directly comparable across families (TF-IDF has none in this sense); BiLSTM has about twice the recurrent parameters of an LSTM trained with the same flags.
