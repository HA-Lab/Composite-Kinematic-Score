# Changes from the received `Integrated_MotoRater_Pipeline_final`

This is a fork of the pipeline as received, with compatibility and robustness
fixes applied. It deliberately does **not** include the scenario-generalization
work (that lives in `Integrated_MotoRater_Pipeline_final/` and is pending
confirmation of the Scenario Types spec).

Verified: reproduces the January 2026 reference results — recovery scores
within 4e-6, every reported statistic identical (Ridge ANOVA 0.0367, Cohen's d
0.9090), same 44 selected features.

## Fixes applied

**Would not run on current libraries**
- `select_features_by_relevance` raised `KeyError: 'original_feature'` under
  pandas 3, which no longer returns the grouping column from
  `groupby().apply()`. Replaced with a rank-based equivalent that selects the
  same features in the same order.
- Extraction died with `ValueError: Too many bins for data range` under
  numpy >= 2.2, which refuses to histogram series that are constant apart from
  floating-point noise (138 of ~60,000 marker series). Now controlled by
  `extraction.drop_degenerate_series` (default `auto`): dropped only when the
  installed numpy cannot process them, kept otherwise, so older environments
  reproduce earlier results exactly.

**Colab survivability**
- Extraction runs in chunks of `extraction.chunk_size` files (default 200),
  each written to `outputs/extraction_cache/chunks/` as it completes. A
  disconnected run resumes from the last finished chunk instead of restarting.
- Each chunk reads only its own files and frees them afterwards. Previously all
  1312 trials were held in memory at once, which exhausted Colab's RAM; peak
  usage drops from 8.89 GB to 4.96 GB.
- Chunking cannot change results: all 47,255,616 feature values come out
  bit-identical to a single-pass extraction, because imputation — the one step
  that looks across trials — still runs once over the combined matrix.

**Correctness**
- The cached feature matrix stores the metadata current when it was written,
  treatment groups included, and a changed animal key did not invalidate it.
  Key-derived columns are now refreshed on load, so a corrected key takes
  effect instead of being silently ignored.
- `permutation_entropy` was patched for stable sorting (good — it removes a
  cross-platform difference), but the replacement function's name leaked into
  the feature names, so 13 columns became
  `..._stable_permutation_entropy__...` and no longer matched earlier runs.
  The original name is now preserved.

**Packaging**
- `requirements.txt` was UTF-16 encoded, which pip cannot read. Now UTF-8, and
  it includes `statsmodels`, `matplotlib` and `seaborn`, which stages 4-5 need.
- `run_pipeline.ipynb` imported `google.colab` unconditionally, so it only ran
  on Colab. It now detects the environment and works locally too; the install
  cell skips packages that are already present.
- `config.yaml`'s `project_root` is `null`, meaning "the folder this pipeline
  lives in", so a fresh copy runs without editing paths.

## Kept from the received version

The stable-sort `permutation_entropy` patch, `fixed_lambdas.json` and
`reproducibility.use_fixed_lambdas`, `force_single_threaded_blas`, the threaded
file reader, and the optional pinned-environment mode
(`scripts/pinned_env.py`).

## Known characteristics (unchanged, not bugs introduced here)

- Results depend on library versions: **numpy < 2.2 and pandas < 3.0** reproduce
  the January numbers. On newer versions the pipeline runs, but selects 37
  features instead of 44 and Ridge's ANOVA moves from 0.0367 to 0.0543.
- `use_fixed_lambdas` tightens run-to-run agreement but does not remove the
  pandas dependency: the `|skew| > 1` gate fires before the lambdas matter, and
  only 25 of the 148 flagged columns are covered by `fixed_lambdas.json`.
- Two copies of `pipeline_functions.py` exist (root and `scripts/`), as in the
  received version. The notebook imports the `scripts/` one. They are identical
  here; keep them in sync or consolidate.
