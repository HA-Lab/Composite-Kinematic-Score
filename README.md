# MotoRater gait recovery pipeline

Computes recovery scores from MotoRater gait tracking: one `.xlsx` of tracked
kinematics per trial goes in, per-animal recovery scores, curves and statistics
come out.

Runs from `run_pipeline.ipynb`, which calls
`scripts/pipeline_functions.py`. All settings in `config.yaml`.

## 1. Install

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install jupyterlab
```

Python 3.12 or 3.13.

> **PINNED NUMPY AND PANDAS VERSIONS ARE NECESSARY FOR REPLICATION.** numpy must be
> below 2.2 and pandas below 3.0. `requirements.txt` handles this, picking
> the right pair for your Python version. Newer versions still run, but select
> 37 features instead of 44 and shift one p-value across 0.05.

## 2. Add the data

```
Integrated_MotoRater_Pipeline_patched/
├── data/                    all the *_out.xlsx trial files, flat
└── animal_data_key.xlsx     animal / treatment group / timepoint / trial
```

Filenames should be `{datecode}_{animal}_{timepoint}_{trial}_out.xlsx`, e.g.
`20250620_251051_Baseline_01_out.xlsx`. The key file maps each filename to its
animal, group and timepoint; the column names it uses are declared in
`config.yaml` under `schema_mapping`, so point those at your file's headers
rather than renaming columns.

`data/` and `outputs/` are in `.gitignore`, so pushing this folder to GitHub
takes the code only. Uploading it to Google Drive takes everything, data
included — one folder serves both.

## 3. Run Locally

```bash
jupyter lab
```

Open `run_pipeline.ipynb` and run cells sequentially. Stages:

| Stage | Function |
|---|---|
| 0 | Load the animal key, apply the schema mapping, profile the dataset |
| 1 | Extract features with tsfresh, filter by relevance |
| 2 | Skew-correct, standardize, select features by ANOVA slope test |
| 3 | Compute recovery scores (CKS, Mahalanobis, kNN_Baseline, Ridge) |
| 4 | Recovery curves, bar plots, statistics |
| 5 | Consensus feature importance 

## 4. Results

Written to `outputs/`:

| Path | Contents |
|---|---|
| `recovery_scores/recovery_scores.csv` | per-trial scores, the main result |
| `recovery_score_analysis/` | recovery curves, per-metric bar plots, `recovery_statistics.xlsx` |
| `feature_imp_analysis/` | consensus feature importance |
| `recovery_scores/feature_importance/` | per-metric importance |
| `extracted_features.csv`, `transformed_selected_features.csv` | feature matrices |
| `extraction_cache/` | extracted features — **DO NOT DELETE** |

**`extraction_cache/`.** Contains batches of extracted features so resumed runs save progress.

## 5. Settings worth knowing

In `config.yaml`:

| Setting | Default | Options |
|---|---|---|
| `paths.project_root` | `null` | `null` means "this folder". Set a path only if the data lives elsewhere |
| `extraction.n_jobs` | `auto` | Parallel workers. `0` runs serially and uses much less memory |
| `extraction.chunk_size` | `200` | Trials per extraction chunk. Lower it on a machine with little RAM |
| `extraction.drop_degenerate_series` | `auto` | Handles marker series that are constant apart from floating-point noise. Leave on auto |
| `recovery_scores.metrics_requested` | all four | Which metrics to attempt |

Each metric is attempted only if the data supports it; if not, the run prints
which one was skipped and why, and continues.

## 6. Run on Colab

The same folder runs on Colab.

1. Upload this folder to **Drive**, with `data/` filled in.
2. Right-click `run_pipeline.ipynb` → **Open with, Google Colaboratory**.
3. In the setup cell, set `PROJECT_DIR` to the folder's path in your Drive,
   e.g. `/content/drive/MyDrive/Integrated_MotoRater_Pipeline_patched`.
4. **Runtime, Run all.**

The first run installs the pinned versions and restarts the runtime; Colab
shows a "session crashed" notice, which is expected. Run all again — later runs
skip it. Locally the same cell only checks versions and reports what to
install, so it never modifies an environment you manage yourself.

Colab gives 2 CPU cores, so extraction is much slower there. If the session
drops, run it again — extraction resumes from the last completed chunk.

## Files

| File | Purpose |
|---|---|
| `run_pipeline.ipynb` | execution notebook |
| `scripts/pipeline_functions.py` | all pipeline logic, defines functions only |
| `config.yaml` | every setting with comments |
| `animal_data_key.xlsx` | trial to animal / group / timepoint |
| `fixed_lambdas.json` | frozen transform parameters for reproducibility |
| `requirements.txt` | requirements; works locally and on Colab |
| `requirements-pinnedenv.txt` | only for the optional pinned-environment mode |
| `.gitignore` | keeps `data/` and `outputs/` out of git |
| `scripts/pinned_env.py` | that optional mode; off by default |
