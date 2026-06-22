# MMPB Clean Public Subset

This directory contains a deterministic 200-row incomplete MMPB clean subset for smoke tests and release examples. It is not the full benchmark dataset.

- `sample.csv`: public subset with preference/recognition and yes/no/MCQ examples across all five preference facets.
- `split.json`: 160 train indices and 40 test indices for the bundled subset.
- `pseudo_users.csv`: counterfactual pseudo-user bank used by the PrefMLLM residual loss.
- `images/` and `injection/`: paired query/profile PNG images referenced by `sample.csv`.
- `manifest.json`: coverage counts for the subset.

Paths in `sample.csv` are relative to `IMAGE_FOLDER=./data`.
