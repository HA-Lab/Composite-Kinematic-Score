# =============================================================================
# pipeline_functions.py
# =============================================================================
# Every function used by the gait recovery pipeline: dataset profiling
# (schema mapping, scenario detection) plus all 5 analysis stages
# (extraction, transform/select, recovery scores, visualize/analyze scores,
# feature importance).
#
# This file only DEFINES functions; nothing here executes on its own.
# run_pipeline.ipynb is what actually runs: it imports this file and calls
# these functions in order, passing config through. Do not run this file
# directly, and avoid editing it unless changing the pipeline's
# underlying logic.
#
# Sections, in order:
#   1. DATASET PROFILE       (schema mapping, scenario detection)
#   2. SHARED HELPERS         (used across multiple stages below)
#   3. STAGE 1 - EXTRACTION
#   4. STAGE 2 - TRANSFORM & SELECT
#   5. STAGE 3 - RECOVERY SCORES
#   6. STAGE 4 - VISUALIZE & ANALYZE SCORES
#   7. STAGE 5 - VISUALIZE & ANALYZE FEATURE IMPORTANCE
# =============================================================================

from __future__ import annotations

import gc
import os
import re
import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.feature_selection.relevance import calculate_relevance_table, combine_relevance_tables
from tsfresh.utilities.dataframe_functions import impute

_warnings.filterwarnings('ignore')

# #############################################################################
# PLATFORM-INDEPENDENT TIE-BREAKING FOR permutation_entropy
# #############################################################################
# np.argsort's default kind='quicksort' has an unspecified tie-break order.
# On small windows (e.g. dimension=7), different platforms/numpy builds can
# resolve ties differently, producing different permutation_entropy
# values from bit-identical input (ex: Windows vs Colab/Linux gave
# entropy differences up to ~0.2 on identical x_final arrays, traced to line inside 
# tsfresh's feature_calculators.permutation_entropy). Forcing kind='stable' makes 
# the tie-break deterministic and reproducible across platforms.
from tsfresh.feature_extraction import feature_calculators as _fc


def _stable_permutation_entropy(x, tau, dimension):
    X = _fc._into_subchunks(x, dimension, tau)
    if len(X) == 0:
        return np.nan
    permutations = np.argsort(np.argsort(X, kind="stable"), kind="stable")
    _, symbol_counts = np.unique(permutations, axis=0, return_counts=True)
    probabilities = symbol_counts / len(permutations)
    return -np.sum(probabilities * np.log(probabilities))


_stable_permutation_entropy.fctype = _fc.permutation_entropy.fctype
# tsfresh derives feature column names from the calculator's __name__, so keep
# the original name -- otherwise every column becomes
# "..._stable_permutation_entropy__..." and no longer matches earlier runs.
_stable_permutation_entropy.__name__ = _fc.permutation_entropy.__name__
_fc.permutation_entropy = _stable_permutation_entropy

# #############################################################################
# 1. DATASET PROFILE
# #############################################################################
# Step 0 of the pipeline; runs before any feature extraction happens.
#
# Responsibilities:
# 1. Apply the config's schema mapping to the raw animal key file, producing
#    a standardized internal DataFrame with fixed column names regardless of
#    what the key file's own columns are called.
# 2. Validate that required columns are present and unambiguous; fill sensible
#    defaults for optional columns that are absent.
# 3. Profile the resulting dataset (groups present, timepoints present, per
#    group x timepoint sample counts, whether baseline and injured data are
#    both available).
# 4. Classify the data scenario (baseline_only / injured_only /
#    longitudinal_single_group / longitudinal_multi_group), honoring
#    scenario_override in config if set, and warning if it disagrees with
#    what was auto-detected.
# 5. Resolve the "reference" values used throughout the rest of the pipeline
#    (injury_reference_timepoint, reference_group) to concrete values, filling
#    in auto-detected defaults wherever config leaves them null.
#
# Nothing here touches the gait Excel files themselves; only reads the
# animal key, and is deliberately fast and fails loudly (missing column, empty
# key file, bad override value, etc.) before any expensive TSFresh extraction
# begins.

# Internal standard column names used by every downstream stage, regardless
# of what the raw key file's columns are called.
STANDARD_COLS = [
    "animal_id",
    "treatment_group",
    "timepoint_days",
    "run_number",
    "date_code",
    "filename",
    "condition",
    "injury_status",
]

REQUIRED_SCHEMA_KEYS = ["animal_id_col", "filename_col"]
# timepoint_days_col is conditionally required (see apply_schema_mapping): 
# either it or injury_status_col must be set.

VALID_SCENARIOS = {
    "baseline_only",
    "injured_only",
    "longitudinal_single_group",
    "longitudinal_multi_group",
}

DEFAULT_GROUP_LABEL = "AllAnimals"


class ConfigError(ValueError):
    """Raised when the config or key file is missing something required."""


@dataclass
class DatasetProfile:
    """Everything downstream stages need to know about the shape of the data."""

    n_rows: int
    groups: list
    timepoints: list
    baseline_value: object
    has_baseline: bool
    has_post_baseline: bool
    has_injured_data: bool
    has_multiple_timepoints: bool
    group_timepoint_counts: pd.DataFrame  # rows=group, cols=timepoint, values=n animals
    animals_per_group: dict
    scenario_detected: str
    scenario_final: str
    scenario_override_used: bool
    resolved_injury_reference_timepoint: Optional[float]
    resolved_reference_group: Optional[str]
    resolved_injury_status_baseline_label: Optional[str] = None
    resolved_injury_status_injured_label: Optional[str] = None
    warnings: list = field(default_factory=list)


def apply_schema_mapping(key_df: pd.DataFrame, schema_cfg: dict) -> pd.DataFrame:
    """
    Rename / synthesize columns in the raw animal key according to
    schema_mapping config, returning a DataFrame with the fixed STANDARD_COLS.

    Required columns: animal_id_col and filename_col must always exist in
    key_df under the names given in schema_cfg. At least one of
    timepoint_days_col or injury_status_col must also be set (see below).
    Missing/misconfigured columns raise ConfigError naming the specific
    problem.

    Optional columns left null in schema_cfg get filled with a single default
    value for every row:
      - treatment_group_col -> "AllAnimals"
      - run_number_col      -> 1
      - date_code_col       -> ""
      - condition_col       -> ""
    """
    warnings = []

    for key in REQUIRED_SCHEMA_KEYS:
        col_name = schema_cfg.get(key)
        if not col_name:
            raise ConfigError(f"schema_mapping.{key} is required but was not set.")
        if col_name not in key_df.columns:
            raise ConfigError(
                f"schema_mapping.{key} = '{col_name}' but no such column exists "
                f"in the animal key file. Columns found: {list(key_df.columns)}"
            )

    std = pd.DataFrame(index=key_df.index)
    std["animal_id"] = key_df[schema_cfg["animal_id_col"]].astype(str)
    std["filename"] = key_df[schema_cfg["filename_col"]].astype(str)

    # If only injury_status_col is set (e.g. a between-subjects design with
    # no repeated time axis at all), timepoint_days gets a constant
    # placeholder instead.
    timepoint_days_col_name = schema_cfg.get("timepoint_days_col")
    injury_status_col_name = schema_cfg.get("injury_status_col")

    if not timepoint_days_col_name and not injury_status_col_name:
        raise ConfigError(
            "At least one of schema_mapping.timepoint_days_col or "
            "schema_mapping.injury_status_col must be set."
        )

    if timepoint_days_col_name:
        if timepoint_days_col_name not in key_df.columns:
            raise ConfigError(
                f"schema_mapping.timepoint_days_col = '{timepoint_days_col_name}' but no such "
                f"column exists in the animal key file. Columns found: {list(key_df.columns)}"
            )
        std["timepoint_days"] = key_df[timepoint_days_col_name]
    else:
        warnings.append(
            "schema_mapping.timepoint_days_col not set; using a constant placeholder "
            "for all rows. Stage 2 (ANOVA-slope feature selection) and Stage 4 (recovery "
            "curves/statistics) require a real time axis and will be skipped."
        )
        std["timepoint_days"] = 0

    def optional_col(cfg_key: str, default_value):
        col_name = schema_cfg.get(cfg_key)
        if col_name is None:
            warnings.append(
                f"schema_mapping.{cfg_key} not set; using default value "
                f"'{default_value}' for all rows."
            )
            return pd.Series(default_value, index=key_df.index)
        if col_name not in key_df.columns:
            raise ConfigError(
                f"schema_mapping.{cfg_key} = '{col_name}' but no such column "
                f"exists in the animal key file. Columns found: {list(key_df.columns)}"
            )
        return key_df[col_name]

    std["treatment_group"] = optional_col("treatment_group_col", DEFAULT_GROUP_LABEL)
    std["run_number"] = optional_col("run_number_col", 1)
    std["date_code"] = optional_col("date_code_col", "")
    std["condition"] = optional_col("condition_col", "")

    # injury_status_col is an opt-in field (see Section B of
    # config.yaml); leave it unset and injury status is inferred from
    # timepoint_days instead.
    injury_status_col_name = schema_cfg.get("injury_status_col")
    if injury_status_col_name is not None:
        if injury_status_col_name not in key_df.columns:
            raise ConfigError(
                f"schema_mapping.injury_status_col = '{injury_status_col_name}' but no such "
                f"column exists in the animal key file. Columns found: {list(key_df.columns)}"
            )
        std["injury_status"] = key_df[injury_status_col_name]
    else:
        std["injury_status"] = None

    std.attrs["warnings"] = warnings
    return std


def _classify_scenario(has_baseline: bool, has_post_baseline: bool, n_groups: int) -> str:
    if has_baseline and not has_post_baseline:
        return "baseline_only"
    if has_post_baseline and not has_baseline:
        return "injured_only"
    if n_groups <= 1:
        return "longitudinal_single_group"
    return "longitudinal_multi_group"


def profile_dataset(std_df: pd.DataFrame, schema_cfg: dict, reference_cfg: dict, baseline_value) -> DatasetProfile:
    """
    Inspect the standardized animal key and produce a DatasetProfile: what
    groups/timepoints exist, whether baseline/injured data are present, per
    group x timepoint sample counts, and the detected/final scenario.

    Injury status ("is this row baseline/healthy or injured") is normally
    inferred purely from timepoint_days (== baseline_value means baseline,
    anything else means some degree of post-injury). If a design doesn't map
    injury status onto timepoint this way (e.g. a between-subjects design), 
    schema_mapping.injury_status_col (+ baseline_label /
    injured_reference_label) specifies injury status directly instead.
    """
    warnings = list(std_df.attrs.get("warnings", []))

    groups = sorted(std_df["treatment_group"].dropna().unique().tolist())
    timepoints = sorted(std_df["timepoint_days"].dropna().unique().tolist())

    # Resolve the injury-status labels
    injury_status_baseline_label = schema_cfg.get("injury_status_baseline_label")
    injury_status_injured_label = schema_cfg.get("injury_status_injured_label")
    uses_injury_status = injury_status_baseline_label is not None

    if uses_injury_status:
        has_baseline = (std_df["injury_status"] == injury_status_baseline_label).any()
        has_injured_data = (
            injury_status_injured_label is not None
            and (std_df["injury_status"] == injury_status_injured_label).any()
        )
        # "post_baseline" is treated as equivalent to has_injured_data here.
        has_post_baseline = has_injured_data
    else:
        has_baseline = baseline_value in timepoints
        has_post_baseline = any(tp != baseline_value for tp in timepoints)
        has_injured_data = has_post_baseline

    # Checks whether there's a real time axis to fit a trajectory against 
    # (Stage 2's ANOVA-slope selection, Stage 4's recovery curves)
    has_multiple_timepoints = len(timepoints) >= 2

    group_timepoint_counts = (
        std_df.groupby(["treatment_group", "timepoint_days"])["animal_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    animals_per_group = std_df.groupby("treatment_group")["animal_id"].nunique().to_dict()

    scenario_detected = _classify_scenario(has_baseline, has_post_baseline, len(groups))

    override = reference_cfg.get("scenario_override")
    scenario_override_used = False
    scenario_final = scenario_detected

    if override:
        if override not in VALID_SCENARIOS:
            raise ConfigError(
                f"reference.scenario_override = '{override}' is not a valid scenario. "
                f"Must be one of {sorted(VALID_SCENARIOS)} or null."
            )
        scenario_override_used = True
        scenario_final = override
        if override != scenario_detected:
            warnings.append(
                f"reference.scenario_override = '{override}' but auto-detection found "
                f"'{scenario_detected}' from the data. Proceeding with the override "
                f"('{override}'); double check this is intentional."
            )

    # Resolve injury_reference_timepoint: use configured value if present in
    # the data, otherwise auto-fill with earliest post-baseline timepoint.
    injury_ref_tp = reference_cfg.get("injury_reference_timepoint")
    if injury_ref_tp is None:
        post_baseline_tps = sorted(tp for tp in timepoints if tp != baseline_value)
        if post_baseline_tps:
            injury_ref_tp = post_baseline_tps[0]
            warnings.append(
                "reference.injury_reference_timepoint not set; auto-using earliest "
                f"post-baseline timepoint found in data: {injury_ref_tp}."
            )
        else:
            injury_ref_tp = None
            if not uses_injury_status:
                warnings.append(
                    "reference.injury_reference_timepoint not set and no post-baseline "
                    "timepoints exist in the data; metrics requiring an injured "
                    "reference will be skipped."
                )
    elif injury_ref_tp not in timepoints:
        warnings.append(
            f"reference.injury_reference_timepoint = {injury_ref_tp} but that "
            f"timepoint was not found in the data (found: {timepoints}). Metrics "
            "requiring an injured reference will be skipped."
        )

    # Resolve reference_group: fall back to "all animals at injury
    # reference timepoint" if configured group doesn't exist in the data.
    reference_group = reference_cfg.get("reference_group")
    if reference_group is not None and reference_group not in groups:
        warnings.append(
            f"reference.reference_group = '{reference_group}' but that group was "
            f"not found in the data (found: {groups}). Falling back to using ALL "
            "animals at the injury reference timepoint as the injured reference."
        )
        reference_group = None

    # No baseline/healthy reference at all: none of the 4 recovery-score
    # metrics can be computed (every one of them requires a baseline
    # reference as a first condition), regardless of group count. Feature selection 
    # may still be meaningful if multiple treatment groups and timepoints exist, so 
    # pipeline is left to run.
    if not has_baseline:
        warnings.append(
            "No baseline/healthy reference data present in this dataset. None of the "
            "4 recovery-score metrics (kNN, Mahalanobis, CKS, Ridge) can be computed "
            "without a baseline reference. "
            "TSFresh injury-relevance filtering (Stage 1) will also be skipped. "
            "Feature selection may still run if multiple treatment groups and "
            "timepoints are present."
        )

    return DatasetProfile(
        n_rows=len(std_df),
        groups=groups,
        timepoints=timepoints,
        baseline_value=baseline_value,
        has_baseline=has_baseline,
        has_post_baseline=has_post_baseline,
        has_injured_data=has_injured_data,
        has_multiple_timepoints=has_multiple_timepoints,
        group_timepoint_counts=group_timepoint_counts,
        animals_per_group=animals_per_group,
        scenario_detected=scenario_detected,
        scenario_final=scenario_final,
        scenario_override_used=scenario_override_used,
        resolved_injury_reference_timepoint=injury_ref_tp,
        resolved_reference_group=reference_group,
        resolved_injury_status_baseline_label=injury_status_baseline_label,
        resolved_injury_status_injured_label=injury_status_injured_label,
        warnings=warnings,
    )


def print_profile_summary(profile: DatasetProfile) -> None:
    """Summary, printed before any heavy computation begins."""
    print("=" * 60)
    print("DATASET PROFILE")
    print("=" * 60)
    print(f"  Animal key rows: {profile.n_rows}")
    print(f"  Treatment groups ({len(profile.groups)}): {profile.groups}")
    print(f"  Timepoints (days): {profile.timepoints}")
    print(f"  Baseline (day {profile.baseline_value}) present: {profile.has_baseline}")
    print(f"  Post-baseline data present: {profile.has_post_baseline}")
    print()
    print("  Animals per group:")
    for g, n in profile.animals_per_group.items():
        print(f"    {g}: {n}")
    print()
    print(f"  Scenario detected: {profile.scenario_detected}")
    if profile.scenario_override_used:
        print(f"  Scenario override applied: {profile.scenario_final}")
    print(f"  Resolved injury reference timepoint: {profile.resolved_injury_reference_timepoint}")
    print(f"  Resolved reference group: {profile.resolved_reference_group}")
    if profile.resolved_injury_status_baseline_label is not None:
        print(f"  Injury status resolved via injury_status_col: "
              f"baseline='{profile.resolved_injury_status_baseline_label}', "
              f"injured='{profile.resolved_injury_status_injured_label}'")
    print()
    if profile.warnings:
        print("  Warnings:")
        for w in profile.warnings:
            print(f"    - {w}")
    print("=" * 60)
    print()


def build_dataset_profile(key_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, DatasetProfile]:
    """
    Main entry point for this section.

    Takes the raw animal key DataFrame (as loaded from Excel) and the full
    parsed config dict. Returns (std_df, profile):
      - std_df:  the standardized key with STANDARD_COLS, for reuse by every
                 downstream stage (so schema mapping only happens once)
      - profile: the DatasetProfile described above
    """
    schema_cfg = config.get("schema_mapping", {})
    reference_cfg = config.get("reference", {})

    if key_df.empty:
        raise ConfigError("Animal key file has no rows.")

    baseline_value = schema_cfg.get("baseline_timepoint_value", 0)

    std_df = apply_schema_mapping(key_df, schema_cfg)
    profile = profile_dataset(std_df, schema_cfg, reference_cfg, baseline_value)
    return std_df, profile


# #############################################################################
# 2. SHARED HELPERS
# #############################################################################
# Logic used across multiple stages below (baseline/injured mask
# construction, the 0-100 rescaling formula, project-path resolution).

def _resolve_project_paths(config: dict) -> dict:
    """
    Resolve the common project/output paths every stage needs from config,
    and ensure the output directory exists.

    Returns a dict with:
      project_root, data_dir, output_dir
    (individual stages create their own subfolders under output_dir as
    needed, e.g. "extraction_cache", "recovery_scores").
    """
    paths_cfg = config['paths']
    # project_root may be left null in config.yaml, meaning "the folder this
    # file lives in", so a fresh copy runs without editing anything. Set it
    # explicitly only if the data lives elsewhere (e.g. a Drive folder).
    configured_root = paths_cfg.get('project_root')
    if configured_root:
        project_root = Path(configured_root)
    else:
        # This file may sit at the project root or in scripts/ -- both layouts
        # are in use here -- so resolve accordingly.
        _here = Path(__file__).resolve().parent
        project_root = _here.parent if _here.name == "scripts" else _here
    data_dir = project_root / paths_cfg.get('data_subdir', 'data')
    output_dir = project_root / paths_cfg.get('output_subdir', 'outputs')
    output_dir.mkdir(exist_ok=True, parents=True)

    return {
        'project_root': project_root,
        'data_dir': data_dir,
        'output_dir': output_dir,
    }


def _resolve_baseline_injured_masks(df: pd.DataFrame, baseline_value, injury_ref_tp,
                                     reference_group,
                                     injury_status_baseline_label=None,
                                     injury_status_injured_label=None,
                                     timepoint_col: str = 'timepoint_days',
                                     group_col: str = 'treatment_group',
                                     injury_status_col: str = 'injury_status') -> tuple:
    """
    Build the baseline and injured-reference boolean masks used throughout
    Stage 1 (Phase 1) and Stage 3 (recovery scores). Replaces logic that was
    previously copy-pasted in compute_cks, compute_mahalanobis_distance,
    compute_knn_baseline_distance, compute_ridge_recovery_score, and
    _check_metric_availability.

    Two ways to determine injury status, chosen automatically:
      - If injury_status_baseline_label is set (schema_mapping.
        injury_status_col was configured), injury status is read directly
        from df[injury_status_col] (independent of timepoint). This covers
        designs where injury status isn't a function of timepoint.
      - Otherwise (default), injury status is inferred from
        timepoint_days == baseline_value / == injury_ref_tp.
      - If no way to determine an injured reference (injury_ref_tp is None under
        the timepoint path, or injury_status_injured_label is None under the
        label path), injured_mask is all-False.

    Returns (baseline_mask, injured_mask), each a boolean Series aligned to
    df's index.
    """
    if injury_status_baseline_label is not None:
        baseline_mask = df[injury_status_col] == injury_status_baseline_label

        if injury_status_injured_label is None:
            injured_mask = pd.Series(False, index=df.index)
        else:
            injured_mask = df[injury_status_col] == injury_status_injured_label
            if reference_group is not None:
                injured_mask = injured_mask & (df[group_col] == reference_group)

        return baseline_mask, injured_mask

    baseline_mask = df[timepoint_col] == baseline_value

    if injury_ref_tp is None:
        injured_mask = pd.Series(False, index=df.index)
    else:
        injured_mask = df[timepoint_col] == injury_ref_tp
        if reference_group is not None:
            injured_mask = injured_mask & (df[group_col] == reference_group)

    return baseline_mask, injured_mask


def _rescale_to_0_100(scores: np.ndarray, baseline_mask: pd.Series, injured_mask: pd.Series) -> tuple:
    """
    Rescale a raw score array so the mean score of the injured-reference
    group maps to 0 and the mean score of the baseline group maps to 100.

    If injured_mask has no True values, or the baseline/injured means are
    numerically indistinguishable, rescaling is skipped and the raw scores
    are returned unchanged (normalized=False).

    Returns (scores, normalized).
    """
    scores = np.asarray(scores, dtype=float).copy()

    if injured_mask.sum() == 0:
        return scores, False

    centroid_baseline = scores[baseline_mask.values].mean()
    centroid_injured = scores[injured_mask.values].mean()
    max_distance = centroid_baseline - centroid_injured

    if abs(max_distance) > 1e-10:
        scores = 100 * (scores - centroid_injured) / max_distance
        return scores, True

    return scores, False


# #############################################################################
# 3. STAGE 1 - EXTRACTION
# #############################################################################
# Loads raw MotoRater Excel files, extracts hundreds of time-series features
# using TSFresh, then (if possible) filters down to the most statistically
# relevant ones.
#
# Output files created:
#   - outputs/extracted_features.csv
#   - outputs/extraction_cache/tsfresh_extracted_features.parquet
#   - outputs/extraction_cache/relevance_injury.parquet      (if computed)
#   - outputs/extraction_cache/relevance_treatment.parquet   (if computed)

def trim_flat_edges(x: np.ndarray) -> tuple:
    """
    Helper function to remove flatline sections at the beginning and end of a time series.
    Finds first and last points where the signal changes, keeps only that range.
    """
    x = np.asarray(x, dtype=float)
    diffs = np.diff(x)
    nonconst_idx = np.where(diffs != 0)[0]

    if len(nonconst_idx) == 0:
        return x, np.arange(len(x))

    start_idx = nonconst_idx[0]
    end_idx = nonconst_idx[-1] + 1
    kept_idx = np.arange(start_idx, end_idx + 1)

    return x[kept_idx], kept_idx


def select_features_by_relevance(relevance_injury, relevance_treatment,
                                 target_n_features=500,
                                 top_k_per_original=20):
    """
    Helper function to select features using TSFresh's combine_relevance_tables.
    Combines features relevant to differentiating both injury and treatment group, then selects top features.

    Parameters:
    -----------
    relevance_injury : Relevance table for injury (baseline vs injury reference timepoint)
    relevance_treatment : Relevance table for treatment (reference group vs other groups)
    target_n_features : Total number of features to select
    top_k_per_original : Maximum features to keep per original feature
    """
    # Normalize index for 'feature' column
    for df in [relevance_injury, relevance_treatment]:
        if df.index.name == 'feature':
            if 'feature' in df.columns:
                df.reset_index(drop=True, inplace=True)
            else:
                df.reset_index(inplace=True)

    print("  Combining relevance tables using TSFresh...")

    # Combine both relevance tables
    combined_relevance = combine_relevance_tables(
        [relevance_injury, relevance_treatment]
    )

    # Filter to significant features (passed FDR correction)
    combined_relevant = combined_relevance[combined_relevance['relevant'] == True].copy()

    print(f"  Combined relevance: {len(combined_relevant)} significant features")

    if len(combined_relevant) == 0:
        raise ValueError("No significant features found after filtering!")

    # Select top-N by smallest p-value
    n_to_select = min(target_n_features, len(combined_relevant))
    selected_combined = combined_relevant.nsmallest(n_to_select, 'p_value')
    intersection = selected_combined['feature'].tolist()

    print(f"    Selected top-{n_to_select} significant features")

    # Balance for diversity across original features
    if top_k_per_original is not None and len(intersection) > 0:
        print(f"  Balancing: limiting to top-{top_k_per_original} per original feature...")

        intersection_df = pd.DataFrame({'feature': intersection})
        intersection_df = intersection_df.merge(
            selected_combined[['feature', 'p_value']],
            on='feature',
            how='left'
        )

        # Extract original feature name (before __)
        intersection_df['original_feature'] = intersection_df['feature'].str.split('__').str[0]

        # Keep top-K per original feature
        # Rank within each original feature and take the top K. Equivalent to
        # nsmallest-per-group, but without groupby().apply(): as of pandas 3.0
        # that drops the grouping column, so the 'original_feature' lookup
        # below raised KeyError.
        rank_in_group = (intersection_df
                         .groupby('original_feature')['p_value']
                         .rank(method='first'))
        balanced = (intersection_df[rank_in_group <= top_k_per_original]
                    .sort_values(['original_feature', 'p_value'], kind='stable')
                    .reset_index(drop=True))
        selected = balanced['feature'].tolist()

        selected_original_features = set(balanced['original_feature'].unique())
        print(f"    After balancing: {len(selected)} features spanning {len(selected_original_features)} original features")
    else:
        selected = intersection

    # Print summary of original features
    if len(selected) > 0:
        all_original_features = set()
        all_original_features.update(relevance_injury['feature'].str.split('__').str[0].unique())
        all_original_features.update(relevance_treatment['feature'].str.split('__').str[0].unique())

        selected_original_features = set([f.split('__')[0] for f in selected])
        not_selected_original_features = all_original_features - selected_original_features

        print(f"\n  Original feature selection summary:")
        print(f"    Selected: {len(selected_original_features)} original features")
        print(f"      {sorted(selected_original_features)}")
        print(f"    Not selected: {len(not_selected_original_features)} original features")
        if len(not_selected_original_features) > 0:
            print(f"      {sorted(not_selected_original_features)}")

    return selected


# Filename parsing, driven by config instead of hardcoded positions.
# Field order is set by config['filename_parsing'].

_BASELINE_LABEL_ALIASES = {'presci', 'baseline'}


def _parse_filename(stem: str, delimiter: str, field_order: list, baseline_value):
    """Parse a MotoRater filename stem into its component fields."""
    parts = stem.split(delimiter)
    if len(parts) < len(field_order):
        return None

    fields = dict(zip(field_order, parts))

    timepoint_label = fields.get('timepoint_label', '')
    if timepoint_label.upper() in {a.upper() for a in _BASELINE_LABEL_ALIASES} or timepoint_label == '0':
        timepoint_days = baseline_value
    else:
        numbers = re.findall(r'\d+', timepoint_label)
        timepoint_days = int(numbers[0]) if numbers else baseline_value

    return {
        'date_code': fields.get('date_code', ''),
        'animal_id': fields.get('animal_id', ''),
        'timepoint_label': timepoint_label,
        'timepoint_days': timepoint_days,
        'run_number': fields.get('run_number', ''),
    }


def _normalize_filename(name: str) -> str:
    """Strip a .xlsx extension if present, for comparing key-file filenames to file stems."""
    return re.sub(r'\.xlsx$', '', str(name), flags=re.IGNORECASE)


def _numpy_rejects_degenerate_histograms() -> bool:
    """True if the installed numpy refuses to histogram a range narrower than
    float resolution (numpy >= 2.2). Older numpy -- including Colab's 2.0.x --
    silently bins the noise instead. Probed by behaviour, not version number."""
    try:
        np.histogram(np.array([125.0, 125.0 + 1e-13]), bins=10)
        return False
    except ValueError:
        return True


def _resolve_drop_degenerate(setting) -> bool:
    """Resolve extraction.drop_degenerate_series ("auto", true, false)."""
    rejects = _numpy_rejects_degenerate_histograms()
    if setting == 'auto':
        return rejects
    if setting is True:
        return True
    if setting is False:
        if rejects:
            raise ValueError(
                f"extraction.drop_degenerate_series is false, but numpy "
                f"{np.__version__} cannot process series that are constant to "
                f"within floating-point noise, so tsfresh would crash partway "
                f"through extraction. Set it to auto or true, or install "
                f"numpy < 2.2 (see requirements.txt)."
            )
        return False
    raise ValueError(
        f"extraction.drop_degenerate_series must be auto, true or false; got {setting!r}"
    )


def run_extraction(config: dict, std_df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """
    Runs the full Stage 1 workflow: load files, extract TSFresh features,
    (if possible) compute two-stage relevance, select features, save CSV.

    Parameters
    ----------
    config : parsed config dict (see config_template.yaml)
    std_df : standardized animal key from build_dataset_profile()
    profile : DatasetProfile from build_dataset_profile()

    Returns
    -------
    X_final : the extracted (and possibly selected) feature matrix with
              metadata columns, same as outputs/extracted_features.csv
    """
    extraction_cfg = config.get('extraction', {})
    fname_cfg = config.get('filename_parsing', {})

    rp = _resolve_project_paths(config)
    DATA_DIR = rp['data_dir']
    OUTPUT_DIR = rp['output_dir']
    EXTRACTION_DIR = OUTPUT_DIR / "extraction_cache"
    EXTRACTION_DIR.mkdir(exist_ok=True, parents=True)

    # Metadata columns produced by the schema-mapping section above.
    META_COLS = STANDARD_COLS

    n_jobs_cfg = extraction_cfg.get('n_jobs', 'auto')
    drop_degenerate = _resolve_drop_degenerate(
        extraction_cfg.get('drop_degenerate_series', 'auto'))
    chunk_size = extraction_cfg.get('chunk_size', 200)
    n_jobs = (os.cpu_count() or 2) if n_jobs_cfg == 'auto' else n_jobs_cfg
    fdr_level = extraction_cfg.get('fdr_level', 0.01)
    target_n_features = extraction_cfg.get('target_n_features', 500)
    top_k_per_original_feature = extraction_cfg.get('top_k_per_original_feature', 20)
    run_relevance_filtering = extraction_cfg.get('run_relevance_filtering', True)

    delimiter = fname_cfg.get('delimiter', '_')
    field_order = fname_cfg.get('field_order', ['date_code', 'animal_id', 'timepoint_label', 'run_number'])

    baseline_value = profile.baseline_value
    injury_ref_tp = profile.resolved_injury_reference_timepoint
    reference_group = profile.resolved_reference_group

    print("=" * 60)
    print("1. LOAD ALL FILES")
    print("=" * 60)
    print()

    excel_files = list(DATA_DIR.glob("*.xlsx"))
    print(f"Found {len(excel_files)} Excel files in {DATA_DIR}")
    print()

    # Build a lookup from normalized filename -> standardized key row, plus a
    # fallback lookup by (animal_id, timepoint_days), mirroring the original
    # "match by FileName, else match by Animal ID + Timepoint" fallback.
    key_by_filename = {
        _normalize_filename(fn): row
        for fn, row in zip(std_df['filename'], std_df.to_dict('records'))
    }
    key_by_id_tp = {}
    for row in std_df.to_dict('records'):
        key_by_id_tp.setdefault((str(row['animal_id']), row['timepoint_days']), row)

    print("=" * 60)
    print("2. FEATURE EXTRACTION WITH TSFRESH")
    print("=" * 60)
    print()

    extracted_features_file = EXTRACTION_DIR / "tsfresh_extracted_features.parquet"

    if extracted_features_file.exists():
        print(f"Found existing extracted features: {extracted_features_file}")
        print("  Loading extracted features (skipping extraction)...")
        X_extracted = pd.read_parquet(extracted_features_file)

        if X_extracted.index.name is None:
            X_extracted.index.name = 'id'

        metadata_df = X_extracted[META_COLS].copy()
        X_extracted = X_extracted.drop(columns=META_COLS)

        # The cache stores the metadata current when it was written,
        # treatment_group included, and a changed animal key does not
        # invalidate it -- so a corrected key would otherwise be silently
        # ignored in favour of stale groups. The tsfresh features themselves
        # don't depend on the key, so refresh the key-derived columns and keep
        # the expensive cache.
        refreshed = set()
        for idx, fname in metadata_df['filename'].items():
            match = key_by_filename.get(_normalize_filename(fname))
            if match is None:
                continue
            for col in ('treatment_group', 'injury_status'):
                if col in match and metadata_df.at[idx, col] != match[col]:
                    metadata_df.at[idx, col] = match[col]
                    refreshed.add(col)
            if 'treatment_group' in match:
                metadata_df.at[idx, 'condition'] = (
                    f"{match['treatment_group']}{metadata_df.at[idx, 'timepoint_days']}"
                )
        if refreshed:
            print(f"  NOTE: the animal key disagrees with the cached metadata; "
                  f"refreshed {sorted(refreshed)} from the current key.")

        print(f"  Loaded {X_extracted.shape[1]} features from {X_extracted.shape[0]} files")
        print()
    else:
        print("No saved features found, extracting now.")
        print(f"  Noise-only series (constant to float precision): "
              f"{'dropped' if drop_degenerate else 'kept'} "
              f"(drop_degenerate_series="
              f"{extraction_cfg.get('drop_degenerate_series', 'auto')}, "
              f"numpy {np.__version__})")
        n_degenerate = 0

        # Get feature names from first file
        sample_file = excel_files[0]
        sample_df = pd.read_excel(sample_file, sheet_name=1)
        sample_df.columns = (sample_df.columns.str.lower()
                            .str.replace(' ', '_')
                            .str.replace('(', '')
                            .str.replace(')', ''))

        feature_names = [col for col in sample_df.columns if col != 'time']
        print(f"Found {len(feature_names)} time series features")

        def _process_one_file(file_path):
            """Load and process a single Excel file. Returns (metadata_dict, per_feature_dfs) or None if unparseable."""
            name = file_path.stem
            parsed = _parse_filename(name, delimiter, field_order, baseline_value)
            if parsed is None:
                return None

            date_code = parsed['date_code']
            animal_id = parsed['animal_id']
            timepoint_str = parsed['timepoint_label']
            timepoint_days = parsed['timepoint_days']
            run_number = parsed['run_number']

            filename_base = _normalize_filename(file_path.stem)
            key_match = key_by_filename.get(filename_base)

            if key_match is not None:
                treatment_group = key_match['treatment_group']
                injury_status = key_match.get('injury_status')
            else:
                key_match2 = key_by_id_tp.get((animal_id, timepoint_days))
                if key_match2 is not None:
                    treatment_group = key_match2['treatment_group']
                    injury_status = key_match2.get('injury_status')
                else:
                    treatment_group = 'Unknown'
                    injury_status = None

            file_id = f"{animal_id}_{timepoint_days}_{run_number}"

            df = pd.read_excel(file_path, sheet_name=1)
            df.columns = (df.columns.str.lower()
                        .str.replace(' ', '_')
                        .str.replace('(', '')
                        .str.replace(')', ''))

            time_col = df['time'].values if 'time' in df.columns else np.arange(len(df))

            per_feature_dfs = {}
            n_degenerate_local = 0
            for feat_name in feature_names:
                if feat_name not in df.columns:
                    continue

                x = df[feat_name].values
                valid_mask = ~np.isnan(x)
                if not np.any(valid_mask):
                    continue

                x_clean = x[valid_mask]
                t_clean = time_col[valid_mask]

                x_trimmed, kept_idx = trim_flat_edges(np.round(x_clean, 15))

                if len(x_trimmed) == 0:
                    x_final = x_clean
                    t_final = t_clean
                else:
                    t_trimmed = t_clean[kept_idx]
                    if np.all(x_trimmed == 0):
                        x_final = x_clean
                        t_final = t_clean
                    else:
                        x_final = x_trimmed
                        t_final = t_trimmed

                if len(x_final) == 0 or len(np.unique(np.round(x_final, 15))) <= 1:
                    continue

                if np.isnan(x_final).any() or np.isinf(x_final).any():
                    continue

                # Skip series that are constant to within floating-point noise
                # (untracked/frozen markers). The np.round(..., 15) check above
                # is absolute, so it only catches this for values near 1.0 -- a
                # joint angle of ~125 deg wobbling by 1e-13 slips through, and
                # numpy >= 2.2 then refuses to histogram it, killing the whole
                # extraction inside tsfresh.
                scale = np.abs(x_final).max()
                if drop_degenerate and scale > 0 and np.ptp(x_final) <= scale * 1e-10:
                    n_degenerate_local += 1
                    continue

                per_feature_dfs[feat_name] = pd.DataFrame({
                    'id': file_id,
                    'time': t_final,
                    'value': x_final
                })

            metadata_entry = {
                'file_id': file_id,
                'animal_id': animal_id,
                'treatment_group': treatment_group,
                'timepoint_days': timepoint_days,
                'timepoint_label': timepoint_str,
                'run_number': run_number,
                'date_code': date_code,
                'filename': file_path.name,
                'condition': f"{treatment_group}{timepoint_days}",
                'injury_status': injury_status,
            }

            return metadata_entry, per_feature_dfs, n_degenerate_local

        # Files are read one chunk at a time (see the chunk loop below), so
        # only chunk_size trials are ever resident in memory.
        def _load_files(files, desc):
            """Read trials into tsfresh containers plus their metadata."""
            timeseries_dict = {feat_name: [] for feat_name in feature_names}
            metadata_list = []
            n_degenerate = 0

            # Process each file
            n_io_workers = extraction_cfg.get('n_io_workers', 'auto')
            if n_io_workers == 'auto':
                n_io_workers = min(32, (os.cpu_count() or 4) * 4)

            with ThreadPoolExecutor(max_workers=n_io_workers) as executor:
                futures = {executor.submit(_process_one_file, fp): fp for fp in files}
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    file_path = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"  ERROR processing {file_path.name}: {e}")
                        continue

                    if result is None:
                        continue

                    metadata_entry, per_feature_dfs, n_degen = result
                    n_degenerate += n_degen
                    metadata_list.append(metadata_entry)
                    for feat_name, feat_df in per_feature_dfs.items():
                        timeseries_dict[feat_name].append(feat_df)


            # Combine time series DataFrames for each feature kind to prepare for TSFresh extraction
            for feat_name in feature_names:
                if len(timeseries_dict[feat_name]) > 0:
                    combined_df = pd.concat(timeseries_dict[feat_name], ignore_index=True)
                    combined_df = combined_df.dropna()
                    if len(combined_df) > 0:
                        timeseries_dict[feat_name] = combined_df
                    else:
                        del timeseries_dict[feat_name]
                else:
                    if feat_name in timeseries_dict:
                        del timeseries_dict[feat_name]

            return timeseries_dict, metadata_list, n_degenerate

        def _extract(container):
            return extract_features(
                timeseries_container=container,
                column_id='id',
                column_sort='time',
                column_kind=None,
                column_value='value',
                default_fc_parameters=None,
                n_jobs=n_jobs,
                disable_progressbar=False
            )

        # Extraction runs in chunks, each saved as it completes, so an
        # interrupted run (a Colab disconnect) resumes from the last finished
        # chunk. Series are independent, so chunking cannot change a feature
        # value; imputation, which does look across trials, still happens once
        # over the combined matrix below. chunk_size: 0 disables chunking.
        print("Extracting TSFresh features (this will take a while)...")

        if not chunk_size or chunk_size >= len(excel_files):
            file_chunks = [excel_files]
        else:
            file_chunks = [excel_files[i:i + chunk_size]
                           for i in range(0, len(excel_files), chunk_size)]

        n_chunks = len(file_chunks)
        resumable = n_chunks > 1
        chunk_dir = EXTRACTION_DIR / "chunks"
        if resumable:
            chunk_dir.mkdir(exist_ok=True, parents=True)
            print(f"  {n_chunks} chunks of up to {chunk_size} files. Each is read,"
                  f" extracted and saved to {chunk_dir.name}/ before the next one"
                  f" starts, so an interrupted run resumes from there.")

        metadata_list = []
        n_degenerate = 0
        part_files = []
        in_memory = []

        for i, files in enumerate(file_chunks, 1):
            label = f"chunk {i}/{n_chunks}" if resumable else "all files"
            part_file = chunk_dir / f"chunk_{i:04d}.parquet" if resumable else None
            meta_file = chunk_dir / f"chunk_{i:04d}_meta.parquet" if resumable else None

            if part_file is not None and part_file.exists() and meta_file.exists():
                print(f"  {label}: already done, loading")
                metadata_list.extend(pd.read_parquet(meta_file).to_dict("records"))
                part_files.append(part_file)
                continue

            desc = f"Processing files ({label})" if resumable else "Processing files"
            timeseries_dict, chunk_meta, chunk_degenerate = _load_files(files, desc)
            metadata_list.extend(chunk_meta)
            n_degenerate += chunk_degenerate

            X_chunk = _extract(timeseries_dict)
            del timeseries_dict

            if part_file is not None:
                X_chunk.to_parquet(part_file)
                pd.DataFrame(chunk_meta).to_parquet(meta_file)
                part_files.append(part_file)
                del X_chunk
            else:
                in_memory.append(X_chunk)
            gc.collect()

        print(f"Processed {len(metadata_list)} files successfully")
        if n_degenerate:
            print(f"  Skipped {n_degenerate} series that were constant to within "
                  f"floating-point noise (untracked/frozen markers)")
        print()

        if part_files:
            X_extracted = pd.concat([pd.read_parquet(f) for f in part_files])
        else:
            X_extracted = in_memory[0]
        in_memory.clear()
        gc.collect()

        X_extracted = X_extracted.reindex(
            [m['file_id'] for m in metadata_list if m['file_id'] in X_extracted.index])

        print(f"  Extracted {X_extracted.shape[1]} features from {X_extracted.shape[0]} files")
        print()

        # Impute NaN values
        print("  Imputing NaN values...")
        X_extracted = impute(X_extracted)

        # Save extracted features with metadata
        print(f"  Saving to: {extracted_features_file}")
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df.set_index('file_id', inplace=True)

        metadata_df_deduped = metadata_df[~metadata_df.index.duplicated(keep='first')]

        X_extracted_with_meta = X_extracted.copy()
        for col in META_COLS:
            X_extracted_with_meta[col] = metadata_df_deduped.reindex(X_extracted_with_meta.index)[col].values

        X_extracted_with_meta.to_parquet(extracted_features_file, index=True)


        # The full matrix is saved, so the per-chunk resume files are redundant.

        chunk_dir = EXTRACTION_DIR / "chunks"

        if chunk_dir.is_dir():

            for f in chunk_dir.glob("chunk_*.parquet"):

                f.unlink()

            chunk_dir.rmdir()
        print(f"  Saved {X_extracted_with_meta.shape[0]} files x {X_extracted_with_meta.shape[1]} features")
        print()

        metadata_df = metadata_df_deduped

    # NOTE: TSFresh extraction can take 30-90 minutes depending on how many files you have.
    # The result is cached as a .parquet file in EXTRACTION_DIR
    # On every run, the cached file will be loaded instantly

    # Relevance filtering is conditional. Stage 1 (injury) needs both
    # baseline and injured data present; Stage 2 (treatment) needs at least
    # 2 treatment groups.
    relevance_injury = None
    relevance_treatment = None

    can_run_stage1 = profile.has_baseline and profile.has_injured_data
    can_run_stage2 = len(profile.groups) >= 2 and profile.has_injured_data

    if not run_relevance_filtering:
        print("extraction.run_relevance_filtering = false; skipping relevance filtering, keeping all extracted features.")
    else:
        print("=" * 60)
        print("3. CALCULATE TWO-STAGE RELEVANCE TABLES")
        print("=" * 60)
        print()
        _injury_desc = (f"baseline vs {injury_ref_tp}dpi" if profile.resolved_injury_status_baseline_label is None
                        else f"'{profile.resolved_injury_status_baseline_label}' vs '{profile.resolved_injury_status_injured_label}' (via injury_status_col)")
        print(f"  Stage 1: Injury relevance ({_injury_desc}); {'will run' if can_run_stage1 else 'SKIPPED (no baseline/injury-reference data)'}")
        print(f"  Stage 2: Treatment relevance (reference group vs others); {'will run' if can_run_stage2 else 'SKIPPED (fewer than 2 treatment groups present)'}")
        print()

        relevance_injury_file = EXTRACTION_DIR / "relevance_injury.parquet"
        relevance_treatment_file = EXTRACTION_DIR / "relevance_treatment.parquet"

        if can_run_stage1:
            if relevance_injury_file.exists():
                print("  Found cached injury relevance table...")
                relevance_injury = pd.read_parquet(relevance_injury_file)
            else:
                print("  Stage 1: Injury relevance...")
                metadata_df = metadata_df[~metadata_df.index.duplicated(keep='first')]
                metadata_df = metadata_df.reindex(X_extracted.index)

                baseline_mask, injured_mask = _resolve_baseline_injured_masks(
                    metadata_df, baseline_value, injury_ref_tp, None,
                    profile.resolved_injury_status_baseline_label,
                    profile.resolved_injury_status_injured_label
                )
                comparison_mask = baseline_mask | injured_mask

                X_injury = X_extracted.loc[comparison_mask]
                y_injury = pd.Series(
                    baseline_mask.loc[comparison_mask].astype(int),
                    index=metadata_df.index[comparison_mask]
                )

                print(f"    Files: {len(y_injury)} ({y_injury.sum()} baseline, {len(y_injury) - y_injury.sum()} injured)")

                relevance_injury = calculate_relevance_table(
                    X_injury,
                    y_injury,
                    ml_task='classification',
                    fdr_level=fdr_level,
                    n_jobs=n_jobs
                )

                print(f"    Found {(relevance_injury['relevant']==True).sum()} significant features")
                print()

                relevance_injury.to_parquet(relevance_injury_file)

        if can_run_stage2:
            if relevance_treatment_file.exists():
                print("  Found cached treatment relevance table...")
                relevance_treatment = pd.read_parquet(relevance_treatment_file)
            else:
                print("  Stage 2: Treatment relevance...")
                metadata_df = metadata_df[~metadata_df.index.duplicated(keep='first')]
                metadata_df = metadata_df.reindex(X_extracted.index)

                _, all_injured_mask = _resolve_baseline_injured_masks(
                    metadata_df, baseline_value, injury_ref_tp, None,
                    profile.resolved_injury_status_baseline_label,
                    profile.resolved_injury_status_injured_label
                )

                if injury_ref_tp is not None:
                    strict_post_injury_mask = all_injured_mask & (metadata_df['timepoint_days'] > injury_ref_tp)
                else:
                    strict_post_injury_mask = pd.Series(False, index=metadata_df.index)

                if strict_post_injury_mask.sum() > 0:
                    post_injury_mask = strict_post_injury_mask
                else:
                    post_injury_mask = all_injured_mask
                    print("    NOTE: no timepoints exist strictly after the injury reference; "
                          "comparing treatment groups across all injured data instead of "
                          "restricting to a later window.")

                X_treatment = X_extracted.loc[post_injury_mask]
                y_treatment_raw = metadata_df.loc[post_injury_mask, 'treatment_group']

                if reference_group is not None:
                    # Binary target: 0 = reference group (e.g. Vehicle), 1 = everything else
                    y_treatment = (y_treatment_raw != reference_group).astype(int)
                    multiclass = False
                    print(f"    Files: {len(y_treatment)} at post-injury timepoints")
                    print(f"    Groups: {y_treatment.value_counts().to_dict()} (0={reference_group}, 1=Other)")
                else:
                    # No single reference group configured/available;
                    # fall back to multiclass relevance across all groups present.
                    y_treatment = y_treatment_raw.astype('category').cat.codes
                    multiclass = True
                    print(f"    Files: {len(y_treatment)} at post-injury timepoints")
                    print(f"    Groups (multiclass): {y_treatment_raw.value_counts().to_dict()}")

                relevance_treatment = calculate_relevance_table(
                    X_treatment,
                    y_treatment,
                    ml_task='classification',
                    multiclass=multiclass,
                    fdr_level=fdr_level,
                    n_jobs=n_jobs
                )

                print(f"    Found {(relevance_treatment['relevant']==True).sum()} significant features")
                print()

                relevance_treatment.to_parquet(relevance_treatment_file)

    print("=" * 60)
    print("4. SELECT FEATURES BY RELEVANCE (IF COMPUTED)")
    print("=" * 60)
    print()

    if relevance_injury is not None and relevance_treatment is not None:
        selected_features = select_features_by_relevance(
            relevance_injury,
            relevance_treatment,
            target_n_features=target_n_features,
            top_k_per_original=top_k_per_original_feature
        )
    elif relevance_injury is not None or relevance_treatment is not None:
        only_table = relevance_injury if relevance_injury is not None else relevance_treatment
        which = "injury" if relevance_injury is not None else "treatment"
        print(f"  Only {which} relevance was computable; selecting from that table alone.")
        relevant = only_table[only_table['relevant'] == True].copy()
        n_to_select = min(target_n_features, len(relevant))
        selected_features = relevant.nsmallest(n_to_select, 'p_value')['feature'].tolist()
        print(f"  Selected {len(selected_features)} features")
    else:
        print("  No relevance tables computed; keeping all extracted features.")
        selected_features = list(X_extracted.columns)

    print()

    if len(selected_features) == 0:
        raise ValueError("No features selected! Try adjusting parameters.")

    print("=" * 60)
    print("5. CREATING FINAL EXTRACTED FEATURE CSV")
    print("=" * 60)

    # Create final feature matrix
    X_final = X_extracted[selected_features].copy()

    # Add metadata columns
    for col in META_COLS:
        X_final[col] = metadata_df.loc[X_final.index, col].values

    # Reorder: metadata first, then features
    feature_cols = [c for c in X_final.columns if c not in META_COLS]
    X_final = X_final[META_COLS + feature_cols]

    # Save to CSV
    output_file = OUTPUT_DIR / "extracted_features.csv"
    X_final.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    print()

    # Summary statistics
    print(f"Total files processed: {X_final.shape[0]}")
    print(f"Features selected: {len(feature_cols)}")
    print(f"Treatment groups: {X_final['treatment_group'].value_counts().to_dict()}")
    print(f"Timepoints: {sorted(X_final['timepoint_days'].unique())}")
    print()
    print("Output files:")
    print(f"  - {output_file}")
    print(f"  - {extracted_features_file}")

    return X_final


# #############################################################################
# 4. STAGE 2 - TRANSFORM & SELECT
# #############################################################################
# Takes the raw extracted features from Stage 1 and does:
#
#   Part A - Transformation:
#     Some features are heavily skewed (ex. most animals score near 0
#     but a few score very high). This applies a Yeo-Johnson transformation to skewed features, 
#     then standardizes everything so all features are on the same scale
#     (mean=0, stdev=1).
#
#   Part B - Feature selection (optional):
#     Selects for features whose recovery slope post-injury differs
#     significantly between treatment groups. Fits a linear regression per
#     animal x feature, then runs a one-way ANOVA to test if slopes differ
#     by group. Features with p < anova_p_value_threshold are kept.
#
# Output files created:
#   - outputs/transformed_selected_features.csv


def load_fixed_lambdas(config: dict):
    """
    Returns the fixed-lambda dict if reproducibility.use_fixed_lambdas is
    true in config, else None. skew_and_scale fits lambda fresh per-run when
    this returns None.
    """
    import json

    repro_cfg = config.get("reproducibility", {})
    if not repro_cfg.get("use_fixed_lambdas", False):
        return None

    rp = _resolve_project_paths(config)
    lambdas_path = rp["project_root"] / repro_cfg.get("fixed_lambdas_file", "fixed_lambdas.json")
    with open(lambdas_path) as f:
        return json.load(f)


def skew_and_scale(df: pd.DataFrame, skew_threshold: float = 1.0, meta_cols=None, fixed_lambdas: dict = None) -> pd.DataFrame:

    """
    1. For each of the features column, calculates how skewed the distribution is
    Meaning: Skewness > threshold then it will apply Yeo-Johnson transformation in order to normalize it.
    (because Yeo-Johnson works on both of positive and negative values)
     - ANOVA & linear regression assumes the roughly normal distributions
    2. Then standardizes all of the features so they will have a mean = 0 & stdev = 1
     - Standardization will prevent the features with large raw values from dominating the data & Stats

    Parameters:
    df (output from Stage 1) & skew_threshold
    fixed_lambdas: optional dict mapping column name -> frozen Yeo-Johnson lambda.
        When provided, a flagged column uses this fixed lambda instead of fitting
        one fresh, for cross-run/cross-environment reproducibility. Columns not
        present in fixed_lambdas still fit fresh.

    Returns:
    df_t  : same shape as input, but feature columns are now transformed
    """
    from scipy import stats
    from sklearn.preprocessing import StandardScaler

    if meta_cols is None:
        meta_cols = STANDARD_COLS
    feature_cols = [c for c in df.columns if c not in meta_cols]

    df_transformed = df.copy()
    skewed_count = 0

    # Step 1: Apply Yeo-Johnson if skewness exceeds threshold
    for col in feature_cols:
        skewness = df[col].skew()
        if abs(skewness) > skew_threshold:
            if fixed_lambdas is not None and col in fixed_lambdas:
                transformed = stats.yeojohnson(df[col], lmbda=fixed_lambdas[col])
            else:
                transformed, _ = stats.yeojohnson(df[col])
            df_transformed[col] = transformed
            skewed_count += 1

    # Step 2: Standardize all features
    scaler = StandardScaler()
    df_transformed[feature_cols] = scaler.fit_transform(df_transformed[feature_cols])

    print(f"  Applied skew adjustment to {skewed_count}/{len(feature_cols)} features")
    print(f"  Standardized all features globally")

    return df_transformed

def select_features_by_anova_slope(df: pd.DataFrame, p_value_threshold: float = 0.1,
                                    post_injury_tps=None, min_timepoints_per_animal: int = 3,
                                    meta_cols=None) -> pd.DataFrame:
    """
    Helper function to perform feature selection using recovery slope differences (ANOVA).

    Computes recovery slopes (injury reference timepoint -> last timepoint) for each animal and tests
    for differences between treatment groups using one-way ANOVA.
    """
    from scipy import stats
    from sklearn.linear_model import LinearRegression

    print(f"\nANOVA Slope Method (p < {p_value_threshold})")

    if meta_cols is None:
        meta_cols = STANDARD_COLS
    feature_cols = [col for col in df.columns if col not in meta_cols]

    # Step 1: Aggregate across runs (mean per animal x timepoint) so each animal only has one value per timepoint
    print("  Step 1: Aggregating across runs...")
    df_agg = df.groupby(['animal_id', 'treatment_group', 'timepoint_days'])[feature_cols].mean().reset_index()

    # Step 2: Compute slopes per animal per feature
    print("  Step 2: Computing recovery slopes per animal...")

    slope_results = []

    for feat in feature_cols:
        # Get data for this feature
        feat_data = df_agg[['animal_id', 'treatment_group', 'timepoint_days', feat]].copy()
        feat_data = feat_data[feat_data['timepoint_days'].isin(post_injury_tps)]
        feat_data = feat_data.dropna(subset=[feat])

        # Compute slope for each animal
        animal_slopes = []

        for animal_id in feat_data['animal_id'].unique():
            animal_data = feat_data[feat_data['animal_id'] == animal_id].sort_values('timepoint_days')

            if len(animal_data) < min_timepoints_per_animal:
                continue

            # Fit linear regression: feature_value ~ timepoint_days
            X = animal_data['timepoint_days'].values.reshape(-1, 1)
            y = animal_data[feat].values

            try:
                reg = LinearRegression()
                reg.fit(X, y)
                slope = reg.coef_[0]

                animal_slopes.append({
                    'animal_id': animal_id,
                    'treatment_group': animal_data['treatment_group'].iloc[0],
                    'slope': slope,
                    'feature': feat
                })
            except Exception:
                continue

        if len(animal_slopes) == 0:
            continue

        # Step 3: Test for slope differences between treatment groups (ANOVA)
        slopes_df = pd.DataFrame(animal_slopes)
        treatment_groups = slopes_df['treatment_group'].unique()

        if len(treatment_groups) < 2:
            continue

        # Prepare data for ANOVA
        groups = [slopes_df[slopes_df['treatment_group'] == tg]['slope'].values
                 for tg in treatment_groups]

        try:
            f_stat, p_value = stats.f_oneway(*groups)

            # Compute mean slopes per treatment
            mean_slopes = slopes_df.groupby('treatment_group')['slope'].mean()

            result_row = {
                'feature': feat,
                'f_statistic': f_stat,
                'p_value': p_value,
                'n_animals': len(slopes_df)
            }
            # Mean slope per group, built dynamically for however many
            # groups are present.
            for tg in treatment_groups:
                result_row[f'mean_slope_{tg}'] = mean_slopes.get(tg, np.nan)

            slope_results.append(result_row)
        except Exception:
            continue

    # Create results dataframe
    slope_stats_df = pd.DataFrame(slope_results)

    if len(slope_stats_df) == 0:
        raise ValueError("No features with valid slopes computed!")

    print(f"  Computed slopes for {len(slope_stats_df)} features")
    print(f"  Mean F-statistic: {slope_stats_df['f_statistic'].mean():.3f}")
    print(f"  Mean p-value: {slope_stats_df['p_value'].mean():.4f}")

    # Select features by p-value threshold
    selected = slope_stats_df[slope_stats_df['p_value'] < p_value_threshold].copy()
    selected = selected.sort_values('f_statistic', ascending=False)

    print(f"\n  Selected {len(selected)} features with p < {p_value_threshold}")
    print(f"  Selection rate: {100 * len(selected) / len(slope_stats_df):.1f}%")

    if len(selected) > 0:
        print(f"\n  Top 5 features by F-statistic:")
        for i, row in selected.head(5).iterrows():
            print(f"    {row['feature']}: F={row['f_statistic']:.2f}, p={row['p_value']:.4f}")

    return selected


def run_transform_select(config: dict, profile: DatasetProfile, df: pd.DataFrame = None, fixed_lambdas: dict = None) -> pd.DataFrame:
    """
    Runs the full Stage 2 workflow: load extracted features, apply skew
    adjustment + scaling, (if possible) select features by ANOVA slope test,
    save CSV.

    Parameters
    ----------
    config : parsed config dict (see config_template.yaml)
    profile : DatasetProfile from build_dataset_profile()
    df : extracted features DataFrame from run_extraction(). If None, loads
         from outputs/extracted_features.csv (lets this stage be re-run on
         its own without re-running extraction).

    Returns
    -------
    df_final : transformed (and possibly selected) feature matrix with
                metadata columns, same as outputs/transformed_selected_features.csv
    """
    ts_cfg = config.get('transform_select', {})
    ts_adv_cfg = config.get('transform_select_advanced', {})

    rp = _resolve_project_paths(config)
    OUTPUT_DIR = rp['output_dir']

    INPUT_FILE = OUTPUT_DIR / "extracted_features.csv"

    meta_cols = STANDARD_COLS

    skew_threshold = ts_adv_cfg.get('skew_threshold', 1.0)
    anova_p_value_threshold = ts_adv_cfg.get('anova_p_value_threshold', 0.1)
    min_timepoints_per_animal = ts_adv_cfg.get('min_timepoints_per_animal_for_slope', 3)
    run_feature_selection = ts_cfg.get('run_feature_selection', True)

    print("=" * 60)
    print("1. LOAD EXTRACTED FEATURES")
    print("=" * 60)
    print()

    if df is None:
        print(f"Loading data from: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} samples")

    feature_cols = [col for col in df.columns if col not in meta_cols]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Treatment groups: {df['treatment_group'].unique()}")
    print(f"  Timepoints: {sorted(df['timepoint_days'].unique())}")
    print()

    # Check for missing values just in case(should be none because handled by TSFresh)
    missing_rows = df[df[feature_cols].isna().any(axis=1)]

    print(f"Missing value check:")
    print(f"  Total rows with missing values: {len(missing_rows)}")

    if len(missing_rows) > 0:
        print("\nRows with missing values:")
        print(missing_rows[meta_cols + ['filename']])

    print()
    print("=" * 60)
    print("2. APPLY TRANSFORMATIONS (SKEW ADJUSTMENT AND SCALING)")
    print("=" * 60)

    df_scaled = skew_and_scale(df, skew_threshold=skew_threshold, meta_cols=meta_cols, fixed_lambdas=fixed_lambdas)
    print("\nTransformations complete")
    print()

    print("=" * 60)
    print("3. FEATURE SELECTION (ANOVA SLOPE METHOD)")
    print("=" * 60)

    # Feature selection is optional, and auto-skips if < 2
    # treatment groups are present or there's no real time axis to fit a
    # slope against (ANOVA-slope selection needs both).
    can_run_selection = len(profile.groups) >= 2 and profile.has_multiple_timepoints

    if not run_feature_selection:
        print("transform_select.run_feature_selection = false; keeping all transformed features.")
        selected_feature_names = feature_cols
    elif not can_run_selection:
        if not profile.has_multiple_timepoints:
            print("  SKIPPED: fewer than 2 distinct timepoints present in the data; ANOVA slope "
                  "selection needs multiple timepoints to compute a slope. Keeping all transformed features.")
        else:
            print(f"  SKIPPED: only {len(profile.groups)} treatment group(s) present "
                  f"({profile.groups}); ANOVA slope selection needs at least 2. "
                  "Keeping all transformed features.")
        selected_feature_names = feature_cols
    else:
        post_injury_tps = sorted(tp for tp in profile.timepoints if tp != profile.baseline_value)
        selected_features = select_features_by_anova_slope(
            df_scaled,
            p_value_threshold=anova_p_value_threshold,
            post_injury_tps=post_injury_tps,
            min_timepoints_per_animal=min_timepoints_per_animal,
            meta_cols=meta_cols
        )
        selected_feature_names = selected_features['feature'].tolist()

    print()
    print("=" * 60)
    print("4. CREATING FINAL FEATURE CSV")
    print("=" * 60)

    # Create final dataset with metadata + selected features
    df_final = df_scaled[meta_cols + selected_feature_names]

    # Save to CSV
    output_file = OUTPUT_DIR / "transformed_selected_features.csv"
    df_final.to_csv(output_file, index=False)

    print(f"Saved to: {output_file}")
    print()
    print(f"Total features analyzed: {len(feature_cols)}")
    print(f"Features selected: {len(selected_feature_names)}")
    print(f"Selection rate: {100 * len(selected_feature_names) / len(feature_cols):.1f}%")

    # Print original features selected
    if len(selected_feature_names) > 0:
        original_features = set([f.split('__')[0] for f in selected_feature_names])
        print(f"\nOriginal features used: {len(original_features)}")
        print(f"  {sorted(original_features)}")

    return df_final


# #############################################################################
# 5. STAGE 3 - RECOVERY SCORES
# #############################################################################
# Takes the transformed, selected features from Stage 2 and computes up to
# 4 recovery score metrics:
#
#   Where computable, scores are normalized to a 0-100 scale where:
#     ~0   = severely injured (like an animal at the injury reference timepoint)
#     ~100 = fully recovered (like a baseline animal)
#
#   1. CKS         : Composite Walking Score: projection onto injured->baseline line in n-D PC space
#   2. Mahalanobis : Distance to baseline distribution, accounting for feature correlations
#   3. kNN         : k-Nearest Neighbor distance to baseline animals (non-parametric)
#   4. Ridge       : Ridge regression trained to classify baseline(1) vs injured(0), with
#                    treatment group interaction terms
#
# Output files created (only for metrics actually computed):
#   - outputs/recovery_scores/recovery_scores.csv
#   - outputs/recovery_scores/feature_importance/pc_importance.csv
#   - outputs/recovery_scores/feature_importance/cks_importance.csv
#   - outputs/recovery_scores/feature_importance/mahalanobis_importance.csv
#   - outputs/recovery_scores/feature_importance/ridge_importance.csv
#   - outputs/recovery_scores/feature_importance/ridge_shap_full.csv

def compute_pca(df: pd.DataFrame, meta_cols: list, n_components: int = None) -> tuple:
    """
    Perform PCA on feature columns.
    """
    from sklearn.decomposition import PCA

    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    pca_model = PCA(n_components=n_components)
    pc_scores = pca_model.fit_transform(X)

    pc_cols = [f'PC{i+1}' for i in range(pc_scores.shape[1])]
    pc_df = pd.DataFrame(pc_scores, columns=pc_cols, index=df.index)

    for col in meta_cols:
        pc_df[col] = df[col].values
    pc_df = pc_df[meta_cols + pc_cols]

    explained_var = pca_model.explained_variance_ratio_

    return pca_model, pc_df, explained_var, feature_cols


def compute_cks(pc_df: pd.DataFrame, n_pcs: int, baseline_value, injury_ref_tp,
                 reference_group, pca_loadings: pd.DataFrame = None,
                 injury_status_baseline_label=None, injury_status_injured_label=None) -> tuple:
    """
    Compute Composite Walking Score (CKS) using PC1-PCn.

    Follows MATLAB logic exactly:
    1. Get baseline and injured-reference centroids in n-D PC space
    2. Compute direction vector: centroid_baseline - centroid_injured
    3. Project all points onto this line (scalar projection)
    4. Normalize: CKS% = (projection / max_distance) * 100
    """
    pc_cols = [f'PC{i+1}' for i in range(n_pcs)]
    available_pcs = [col for col in pc_cols if col in pc_df.columns]

    if len(available_pcs) < n_pcs:
        raise ValueError(f"Not enough PC columns available. Requested {n_pcs}, found {len(available_pcs)}")

    pc_cols = available_pcs[:n_pcs]

    baseline_mask, injured_mask = _resolve_baseline_injured_masks(
        pc_df, baseline_value, injury_ref_tp, reference_group,
        injury_status_baseline_label, injury_status_injured_label
    )

    baseline_data = pc_df.loc[baseline_mask, pc_cols].values
    injured_data = pc_df.loc[injured_mask, pc_cols].values

    if len(baseline_data) == 0:
        raise ValueError("No baseline data found")
    if len(injured_data) == 0:
        raise ValueError("No injured-reference data found")

    centroid_baseline = baseline_data.mean(axis=0)
    centroid_injured = injured_data.mean(axis=0)

    direction_vector = centroid_baseline - centroid_injured
    direction_norm = np.linalg.norm(direction_vector)

    if direction_norm < 1e-10:
        return pd.Series(0.0, index=pc_df.index), None

    direction_vector_normalized = direction_vector / direction_norm

    feature_importance = None
    if pca_loadings is not None:
        pc_weights = direction_vector_normalized
        available_pcs_imp = [pc for pc in pc_cols if pc in pca_loadings.index]

        if len(available_pcs_imp) == n_pcs:
            loadings_subset = pca_loadings.loc[available_pcs_imp]
            feature_importance = (loadings_subset.T * pc_weights).sum(axis=1)
            feature_importance = pd.Series(feature_importance, index=loadings_subset.columns)

    max_distance = direction_norm

    all_pc_data = pc_df[pc_cols].values
    cks_scores = np.zeros(len(pc_df))

    for i in range(len(pc_df)):
        point = all_pc_data[i]
        relative_vector = point - centroid_injured
        scalar_projection = np.dot(relative_vector, direction_vector_normalized)
        cks_scores[i] = (scalar_projection / max_distance) * 100

    mean_baseline = cks_scores[baseline_mask].mean()
    mean_injured = cks_scores[injured_mask].mean()

    if mean_baseline <= mean_injured:
        cks_scores = -cks_scores + 100

    return pd.Series(cks_scores, index=pc_df.index), feature_importance


def compute_mahalanobis_distance(pc_df: pd.DataFrame, n_pcs: int, baseline_value,
                                  injury_ref_tp, reference_group,
                                  pca_loadings: pd.DataFrame = None,
                                  injury_status_baseline_label=None, injury_status_injured_label=None) -> tuple:
    """
    Compute Mahalanobis distance to baseline distribution in PC space.
    Higher scores indicate better recovery (closer to baseline).

    Returns
    -------
    (mahal_scores, feature_importance, normalized)
    normalized=False means an injured reference wasn't available to rescale
    onto 0-100, and mahal_scores is the raw distance instead (still higher =
    closer to baseline, just not on the standard scale).
    """
    from scipy.linalg import inv

    pc_cols = [f'PC{i+1}' for i in range(n_pcs)]
    data = pc_df[pc_cols].values

    baseline_mask, injured_mask = _resolve_baseline_injured_masks(
        pc_df, baseline_value, injury_ref_tp, reference_group,
        injury_status_baseline_label, injury_status_injured_label
    )
    baseline_data = data[baseline_mask.values]

    baseline_mean = baseline_data.mean(axis=0)
    baseline_cov = np.cov(baseline_data.T)
    baseline_cov += np.eye(baseline_cov.shape[0]) * 1e-6

    try:
        baseline_cov_inv = inv(baseline_cov)
    except Exception:
        baseline_cov_inv = np.linalg.pinv(baseline_cov)

    feature_importance = None
    if pca_loadings is not None and all(pc in pca_loadings.index for pc in pc_cols):
        pc_weights = np.diag(baseline_cov_inv)
        loadings_subset = pca_loadings.loc[pc_cols]
        feature_importance = (loadings_subset.T ** 2 * pc_weights).sum(axis=1)

    mahal_distances = np.zeros(len(pc_df))

    for i in range(len(pc_df)):
        point = data[i]
        diff = point - baseline_mean
        mahal_distances[i] = np.sqrt(np.dot(np.dot(diff, baseline_cov_inv), diff))

    # Distance is already "higher = further from baseline"; flip sign so
    # higher = more similar to baseline, consistent with the other metrics,
    # before attempting any rescaling.
    mahal_scores = -mahal_distances

    mahal_scores, normalized = _rescale_to_0_100(mahal_scores, baseline_mask, injured_mask)

    return pd.Series(mahal_scores, index=pc_df.index), feature_importance, normalized


def compute_knn_baseline_distance(features_df: pd.DataFrame, meta_cols: list, k: int,
                                   baseline_value, injury_ref_tp, reference_group,
                                   injury_status_baseline_label=None, injury_status_injured_label=None) -> tuple:
    """
    Compute k-NN distance to baseline animals.
    Higher scores indicate better recovery (more similar to baseline).

    Returns (knn_scores, normalized); see compute_mahalanobis_distance for
    what `normalized` means.

    NOTE: no feature importance returned because uses entire feature vector,
    so each feature has equal importance.
    """
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    baseline_mask, injured_mask = _resolve_baseline_injured_masks(
        features_df, baseline_value, injury_ref_tp, reference_group,
        injury_status_baseline_label, injury_status_injured_label
    )
    baseline_data = features_df[baseline_mask].copy()
    baseline_avg = baseline_data.groupby('animal_id')[feature_cols].mean().values

    if len(baseline_avg) == 0:
        return pd.Series(0.0, index=features_df.index), False

    knn_scores = np.zeros(len(features_df))
    feature_matrix = features_df[feature_cols].values.astype(float)

    for i in range(len(features_df)):
        features = feature_matrix[i]

        diff = baseline_avg - features
        distances = np.sqrt((diff ** 2).sum(axis=1))

        k_nearest = np.partition(distances, min(k - 1, len(distances) - 1))[:k]
        mean_distance = k_nearest.mean()

        knn_scores[i] = 1 / (1 + mean_distance)

    knn_scores, normalized = _rescale_to_0_100(knn_scores, baseline_mask, injured_mask)

    return pd.Series(knn_scores, index=features_df.index), normalized


def compute_ridge_recovery_score(features_df: pd.DataFrame, meta_cols: list, alpha_grid: list,
                                  baseline_value, injury_ref_tp, reference_group,
                                  injury_status_baseline_label=None, injury_status_injured_label=None) -> tuple:
    """
    Compute recovery score using Ridge Regression, with treatment-group
    interaction terms when a single reference_group is resolved (generalizes
    the original is_vehicle/is_treated indicators to is_reference/is_other).
    If no single reference_group is available, trains without interaction
    terms instead (base features only).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import r2_score

    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    baseline_mask, injured_mask = _resolve_baseline_injured_masks(
        features_df, baseline_value, injury_ref_tp, reference_group,
        injury_status_baseline_label, injury_status_injured_label
    )
    baseline_data = features_df[baseline_mask].copy()
    injured_data = features_df[injured_mask].copy()

    baseline_avg = baseline_data.groupby('animal_id').agg(
        {**{feat: 'mean' for feat in feature_cols}, 'treatment_group': 'first'}
    ).reset_index()

    injured_avg = injured_data.groupby('animal_id').agg(
        {**{feat: 'mean' for feat in feature_cols}, 'treatment_group': 'first'}
    ).reset_index()

    use_interactions = reference_group is not None
    interaction_cols = []

    if use_interactions:
        for d in [baseline_avg, injured_avg]:
            d['is_reference'] = (d['treatment_group'] == reference_group).astype(float)
            d['is_other'] = (d['treatment_group'] != reference_group).astype(float)

        for feat in feature_cols:
            for d in [baseline_avg, injured_avg]:
                d[f'{feat}_x_reference'] = d[feat] * d['is_reference']
                d[f'{feat}_x_other'] = d[feat] * d['is_other']
            interaction_cols.extend([f'{feat}_x_reference', f'{feat}_x_other'])

        all_feature_cols = feature_cols + ['is_reference', 'is_other'] + interaction_cols
    else:
        print("    No single reference group resolved; training Ridge without treatment interaction terms.")
        all_feature_cols = feature_cols

    # Prepare training data
    X_train, y_train, animal_ids_train = [], [], []

    for _, row in baseline_avg.iterrows():
        X_train.append(row[all_feature_cols].values)
        y_train.append(1.0)
        animal_ids_train.append(row['animal_id'])

    for _, row in injured_avg.iterrows():
        X_train.append(row[all_feature_cols].values)
        y_train.append(0.0)
        animal_ids_train.append(row['animal_id'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    animal_ids_train = np.array(animal_ids_train)

    print(f"    Training: {len(baseline_avg)} baseline + {len(injured_avg)} injured-reference = {len(X_train)} samples")
    print(f"    Features: {len(feature_cols)} base + {len(interaction_cols)} interactions = {len(all_feature_cols)} total")

    unique_animals = np.unique(animal_ids_train)
    n_folds = min(5, len(unique_animals) // 2)
    skf = StratifiedKFold(n_splits=max(2, n_folds), shuffle=True, random_state=42)

    best_alpha = alpha_grid[0]
    best_cv_score = -np.inf

    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, random_state=42)
        cv_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            if len(np.unique(y_pred)) > 1:
                r2 = r2_score(y_val, y_pred)
                if not np.isnan(r2) and not np.isinf(r2):
                    cv_scores.append(r2)

        mean_cv_score = np.mean(cv_scores) if len(cv_scores) > 0 else -np.inf

        if mean_cv_score > best_cv_score:
            best_cv_score = mean_cv_score
            best_alpha = alpha

    model = Ridge(alpha=best_alpha, random_state=42)
    model.fit(X_train, y_train)

    print(f"    Best alpha: {best_alpha}, CV R2: {best_cv_score:.3f}")

    def _build_row_features(row):
        if use_interactions:
            treatment = row['treatment_group']
            is_reference = 1.0 if treatment == reference_group else 0.0
            is_other = 1.0 - is_reference

            feature_dict = {}
            for feat in feature_cols:
                feature_dict[feat] = row[feat]
                feature_dict[f'{feat}_x_reference'] = row[feat] * is_reference
                feature_dict[f'{feat}_x_other'] = row[feat] * is_other
            feature_dict['is_reference'] = is_reference
            feature_dict['is_other'] = is_other
            return np.array([feature_dict[col] for col in all_feature_cols]).reshape(1, -1)
        else:
            return row[feature_cols].values.reshape(1, -1).astype(float)

    scores = np.zeros(len(features_df))
    for pos, (idx, row) in enumerate(features_df.iterrows()):
        scores[pos] = model.predict(_build_row_features(row))[0]

    scores, _ = _rescale_to_0_100(scores, baseline_mask, injured_mask)

    # Compute SHAP values
    try:
        import shap

        print(f"    Computing SHAP values...")
        explainer = shap.LinearExplainer(model, X_train)

        all_shap_values = []
        for idx, row in features_df.iterrows():
            shap_vals = explainer.shap_values(_build_row_features(row))
            if isinstance(shap_vals, np.ndarray):
                all_shap_values.append(shap_vals.flatten() if shap_vals.ndim > 1 else shap_vals)
            else:
                all_shap_values.append(np.array([shap_vals]))

        all_shap_array = np.array(all_shap_values)
        shap_df_full = pd.DataFrame(all_shap_array, index=features_df.index, columns=all_feature_cols)

        shap_df = pd.DataFrame(index=features_df.index, columns=feature_cols)
        for feat in feature_cols:
            if use_interactions:
                shap_df[feat] = (
                    shap_df_full[feat].values
                    + shap_df_full[f'{feat}_x_reference'].values
                    + shap_df_full[f'{feat}_x_other'].values
                )
            else:
                shap_df[feat] = shap_df_full[feat].values

        # Unified importance, weighted by group sizes present
        group_sizes = features_df['treatment_group'].value_counts()
        weighted = pd.Series(0.0, index=feature_cols)
        for group_name, n in group_sizes.items():
            idx_mask = features_df['treatment_group'] == group_name
            weighted += shap_df.loc[idx_mask].abs().mean(axis=0) * n
        unified_shap_importance = weighted / group_sizes.sum()

        return pd.Series(scores, index=features_df.index), shap_df_full, unified_shap_importance

    except ImportError:
        print("    WARNING: SHAP not installed, skipping importance calculation")
        return pd.Series(scores, index=features_df.index), None, None


def _check_metric_availability(features_df: pd.DataFrame, profile: DatasetProfile,
                                n_pcs: int, baseline_min_n: int, injured_reference_min_n: int) -> tuple:
    """
    Each metric's requirement is checked once, up front, against the resolved reference
    values and actual sample counts in features_df.
    """
    baseline_value = profile.baseline_value
    injury_ref_tp = profile.resolved_injury_reference_timepoint
    reference_group = profile.resolved_reference_group

    baseline_mask, injured_mask = _resolve_baseline_injured_masks(
        features_df, baseline_value, injury_ref_tp, reference_group,
        profile.resolved_injury_status_baseline_label,
        profile.resolved_injury_status_injured_label
    )
    baseline_n = features_df.loc[baseline_mask, 'animal_id'].nunique()
    injured_reference_n = features_df.loc[injured_mask, 'animal_id'].nunique()

    availability = {}
    reasons = {}

    availability['kNN'] = baseline_n >= baseline_min_n
    if not availability['kNN']:
        reasons['kNN'] = f"needs >= {baseline_min_n} baseline animals, found {baseline_n}"

    mahal_min_n = max(baseline_min_n, n_pcs + 1)
    availability['Mahalanobis'] = baseline_n >= mahal_min_n
    if not availability['Mahalanobis']:
        reasons['Mahalanobis'] = f"needs >= {mahal_min_n} baseline animals (n_pcs + 1), found {baseline_n}"

    cks_ok = (baseline_n >= baseline_min_n) and (injury_ref_tp is not None) and (injured_reference_n >= injured_reference_min_n)
    availability['CKS'] = cks_ok
    if not cks_ok:
        if injury_ref_tp is None:
            reasons['CKS'] = "no injured reference timepoint available in the data"
        else:
            reasons['CKS'] = (f"needs >= {baseline_min_n} baseline and >= {injured_reference_min_n} "
                               f"injured-reference animals; found {baseline_n} / {injured_reference_n}")

    ridge_ok = (baseline_n >= injured_reference_min_n) and (injury_ref_tp is not None) and (injured_reference_n >= injured_reference_min_n)
    availability['Ridge'] = ridge_ok
    if not ridge_ok:
        if injury_ref_tp is None:
            reasons['Ridge'] = "no injured reference timepoint available in the data"
        else:
            reasons['Ridge'] = (f"needs >= {injured_reference_min_n} animals per class for cross-validation; "
                                 f"found {baseline_n} baseline / {injured_reference_n} injured-reference")

    counts = {
        'baseline_n': baseline_n,
        'injured_reference_n': injured_reference_n,
        'injury_ref_tp': injury_ref_tp,
        'reference_group': reference_group,
    }
    return availability, reasons, counts


def run_recovery_scores(config: dict, profile: DatasetProfile, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Runs the full Stage 3 workflow: load transformed features, check which
    metrics are computable given the data, run those, save recovery scores +
    feature importance files.
    """
    rs_cfg = config.get('recovery_scores', {})

    rp = _resolve_project_paths(config)
    OUTPUT_DIR = rp['output_dir']
    RECOVERY_DIR = OUTPUT_DIR / "recovery_scores"
    FEAT_IMPORTANCE_DIR = RECOVERY_DIR / "feature_importance"
    RECOVERY_DIR.mkdir(exist_ok=True, parents=True)
    FEAT_IMPORTANCE_DIR.mkdir(exist_ok=True, parents=True)

    INPUT_FILE = OUTPUT_DIR / "transformed_selected_features.csv"
    meta_cols = STANDARD_COLS

    n_pcs = rs_cfg.get('n_pcs', 10)
    metrics_requested = rs_cfg.get('metrics_requested', ['kNN', 'Mahalanobis', 'CKS', 'Ridge'])
    knn_k = rs_cfg.get('knn', {}).get('k', 6)
    ridge_alpha_grid = rs_cfg.get('ridge', {}).get(
        'alpha_grid', [1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    )
    reqs_cfg = rs_cfg.get('metric_requirements', {})
    baseline_min_n = reqs_cfg.get('baseline_min_n', 3)
    injured_reference_min_n = reqs_cfg.get('injured_reference_min_n', 3)

    print("=" * 60)
    print("1. LOAD TRANSFORMED, SELECTED FEATURES")
    print("=" * 60)
    print()

    if df is None:
        print(f"Loading data from: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
    features_df = df

    print(f"  Loaded {features_df.shape[0]} samples")

    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    print(f"  Features: {len(feature_cols)}")
    print(f"  Treatment groups: {features_df['treatment_group'].unique()}")
    print(f"  Timepoints: {sorted(features_df['timepoint_days'].unique())}")
    print()

    result_df = features_df[meta_cols].copy()

    # Check metric availability up front, print a clear summary before
    # attempting anything.
    availability, reasons, counts = _check_metric_availability(
        features_df, profile, n_pcs, baseline_min_n, injured_reference_min_n
    )

    print("=" * 60)
    print("METRIC AVAILABILITY CHECK")
    print("=" * 60)
    print(f"  Baseline animals: {counts['baseline_n']}")
    print(f"  Injured-reference animals (tp={counts['injury_ref_tp']}, group={counts['reference_group']}): "
          f"{counts['injured_reference_n']}")
    print()
    for metric in ['kNN', 'Mahalanobis', 'CKS', 'Ridge']:
        requested = metric in metrics_requested
        ok = availability[metric] and requested
        status = "WILL RUN" if ok else ("NOT REQUESTED" if not requested else f"SKIPPED; {reasons[metric]}")
        print(f"  {metric}: {status}")
    print()

    metrics_to_run = [m for m in metrics_requested if availability.get(m, False)]

    # PCA; only needed for CKS / Mahalanobis
    pc_df = None
    pc_loadings_df = None
    if 'CKS' in metrics_to_run or 'Mahalanobis' in metrics_to_run:
        print("=" * 60)
        print("Running PCA (needed for CKS and/or Mahalanobis)")
        print("=" * 60)

        pca_model, pc_df, explained_var, feature_cols = compute_pca(features_df, meta_cols, n_components=None)

        n_pcs_available = len([c for c in pc_df.columns if c.startswith('PC')])
        print(f"  Computed {n_pcs_available} principal components")
        print(f"  PC1 explains {explained_var[0]:.1%} of variance")
        print(f"  PC1-PC{min(10, n_pcs_available)} explain {np.sum(explained_var[:10]):.1%} of variance")
        print()

        n_pcs_to_save = min(n_pcs, pca_model.n_components_)
        pc_loadings_df = pd.DataFrame(
            pca_model.components_[:n_pcs_to_save],
            columns=feature_cols,
            index=[f'PC{i+1}' for i in range(n_pcs_to_save)]
        )
        pca_loadings_file = FEAT_IMPORTANCE_DIR / "pc_importance.csv"
        pc_loadings_df.to_csv(pca_loadings_file, index=True)
        print(f"Saved PCA loadings to: {pca_loadings_file}")
        print()

    baseline_value = profile.baseline_value
    injury_ref_tp = profile.resolved_injury_reference_timepoint
    reference_group = profile.resolved_reference_group

    # CKS
    if 'CKS' in metrics_to_run:
        print("=" * 60)
        print("Metric: CKS Score")
        print("=" * 60)
        try:
            cks_scores, cks_importance = compute_cks(
                pc_df, n_pcs=n_pcs, baseline_value=baseline_value,
                injury_ref_tp=injury_ref_tp, reference_group=reference_group,
                pca_loadings=pc_loadings_df,
                injury_status_baseline_label=profile.resolved_injury_status_baseline_label,
                injury_status_injured_label=profile.resolved_injury_status_injured_label
            )
            result_df['CKS'] = cks_scores
            if cks_importance is not None:
                cks_file = FEAT_IMPORTANCE_DIR / "cks_importance.csv"
                cks_importance.to_csv(cks_file, index=True)
                print(f"Saved feature importance to: {cks_file}")
        except Exception as e:
            print(f"  SKIPPED CKS; unexpected error: {e}")
        print()

    # Mahalanobis
    if 'Mahalanobis' in metrics_to_run:
        print("=" * 60)
        print("Metric: Mahalanobis Distance")
        print("=" * 60)
        try:
            mahal_scores, mahal_importance, mahal_normalized = compute_mahalanobis_distance(
                pc_df, n_pcs=n_pcs, baseline_value=baseline_value,
                injury_ref_tp=injury_ref_tp, reference_group=reference_group,
                pca_loadings=pc_loadings_df,
                injury_status_baseline_label=profile.resolved_injury_status_baseline_label,
                injury_status_injured_label=profile.resolved_injury_status_injured_label
            )
            result_df['Mahalanobis'] = mahal_scores
            if not mahal_normalized:
                print("  NOTE: no injured reference available; Mahalanobis is a raw distance "
                      "(higher = more similar to baseline), not rescaled to 0-100.")
            if mahal_importance is not None:
                mahal_file = FEAT_IMPORTANCE_DIR / "mahalanobis_importance.csv"
                mahal_importance.to_csv(mahal_file, index=True)
                print(f"Saved feature importance to: {mahal_file}")
        except Exception as e:
            print(f"  SKIPPED Mahalanobis; unexpected error: {e}")
        print()

    # kNN
    if 'kNN' in metrics_to_run:
        print("=" * 60)
        print("Metric: kNN Baseline Distance")
        print("=" * 60)
        try:
            knn_scores, knn_normalized = compute_knn_baseline_distance(
                features_df, meta_cols, k=knn_k, baseline_value=baseline_value,
                injury_ref_tp=injury_ref_tp, reference_group=reference_group,
                injury_status_baseline_label=profile.resolved_injury_status_baseline_label,
                injury_status_injured_label=profile.resolved_injury_status_injured_label
            )
            result_df['kNN_Baseline'] = knn_scores
            print(f"kNN distance computed (k={knn_k})")
            if not knn_normalized:
                print("  NOTE: no injured reference available; kNN_Baseline is a raw similarity "
                      "score (higher = more similar to baseline), not rescaled to 0-100.")
        except Exception as e:
            print(f"  SKIPPED kNN; unexpected error: {e}")
        print()

    # Ridge
    if 'Ridge' in metrics_to_run:
        print("=" * 60)
        print("Metric: Ridge Regression Score")
        print("=" * 60)
        try:
            ridge_scores, ridge_shap_full, ridge_shap_importance = compute_ridge_recovery_score(
                features_df, meta_cols, alpha_grid=ridge_alpha_grid,
                baseline_value=baseline_value, injury_ref_tp=injury_ref_tp,
                reference_group=reference_group,
                injury_status_baseline_label=profile.resolved_injury_status_baseline_label,
                injury_status_injured_label=profile.resolved_injury_status_injured_label
            )
            result_df['Ridge'] = ridge_scores

            if ridge_shap_full is not None:
                ridge_full_file = FEAT_IMPORTANCE_DIR / "ridge_shap_full.csv"
                ridge_shap_full.to_csv(ridge_full_file, index=True)
                print(f"Saved full SHAP values to: {ridge_full_file.name}")

                ridge_unified_file = FEAT_IMPORTANCE_DIR / "ridge_importance.csv"
                ridge_shap_importance.to_csv(ridge_unified_file, index=True)
                print(f"Saved unified importance to: {ridge_unified_file.name}")
        except Exception as e:
            print(f"  SKIPPED Ridge; unexpected error: {e}")
        print()

    print("=" * 60)
    print("SAVING ALL RECOVERY SCORES")
    print("=" * 60)

    score_cols = [c for c in ['CKS', 'Mahalanobis', 'kNN_Baseline', 'Ridge'] if c in result_df.columns]
    result_df = result_df[meta_cols + score_cols]

    output_file = RECOVERY_DIR / "recovery_scores.csv"
    result_df.to_csv(output_file, index=False)

    print(f"Saved recovery scores to: {output_file}")
    print(f"  Shape: {result_df.shape[0]} samples x {len(score_cols)} metrics computed: {score_cols}")
    print()
    print(f"Output files:")
    print(f"  - {output_file}")

    return result_df


# #############################################################################
# 6. STAGE 4 - VISUALIZE & ANALYZE SCORES
# #############################################################################
# Takes the recovery scores computed in Stage 3 and:
#
#   Part A - Visualization:
#     Plots recovery curves showing how each treatment group changes over
#     time for each metric. Also generates bar plots showing scores at each
#     timepoint side by side.
#
#   Part B - Statistics:
#     For each metric, runs:
#       1. Mixed-effects model - does each group show significant
#          recovery trend over time?
#       2. Permutation test - does each non-reference group improve more
#          than the reference group from the injury reference timepoint to
#          the final timepoint?
#       3. Cohen's d - effect size for each non-reference group vs. the
#          reference group.
#     All stats saved to an excel file.
#
# Output files created (only for metrics actually present):
#   - outputs/recovery_score_analysis/recovery_curves_<n>.png
#   - outputs/recovery_score_analysis/recovery_bars_<metric>.png
#   - outputs/recovery_score_analysis/recovery_statistics.xlsx

_DEFAULT_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def _resolve_group_colors(configured_colors: dict, groups: list) -> dict:
    """Fill in colors for any group not explicitly configured, from a default palette."""
    colors = dict(configured_colors or {})
    used = set(colors.values())
    remaining = [c for c in _DEFAULT_PALETTE if c not in used]
    palette_iter = iter(remaining)
    for g in groups:
        if g not in colors:
            colors[g] = next(palette_iter, 'gray')
    return colors


def _get_improvements(animal_tp_avg: pd.DataFrame, treatment: str, metric_name: str,
                       tp_start, tp_end) -> list:
    """Per-animal change in metric_name from tp_start to tp_end, for one treatment group."""
    improvements = []
    subset = animal_tp_avg[animal_tp_avg['treatment_group'] == treatment]

    for animal_id in subset['animal_id'].unique():
        animal_data = subset[subset['animal_id'] == animal_id]
        animal_data = animal_data[animal_data['timepoint_days'].isin([tp_start, tp_end])]

        if len(animal_data) < 2:
            continue

        val_start = animal_data[animal_data['timepoint_days'] == tp_start][metric_name].values
        val_end = animal_data[animal_data['timepoint_days'] == tp_end][metric_name].values

        if len(val_start) > 0 and len(val_end) > 0:
            improvements.append(val_end[0] - val_start[0])

    return improvements


def compute_metric_stats(df: pd.DataFrame, metric_name: str, injury_ref_tp, reference_group) -> dict:
    """
    Helper function to compute recovery statistics for a metric.

    Tests performed:
    1. If each treatment group shows recovery over time (slope test)
    2. If total recovery amount (injury reference timepoint -> last timepoint)
       differs between reference_group and every other group
    3. Cohen's d effect size for each of those comparisons
    """
    from scipy import stats
    from scipy.stats import f_oneway
    import statsmodels.formula.api as smf

    # Average each animal's scores at each timepoint
    animal_tp_avg = df.groupby(['animal_id', 'treatment_group', 'timepoint_days'])[metric_name].mean().reset_index()

    # Focus on recovery period (injury reference timepoint onwards)
    recovery_data = animal_tp_avg[animal_tp_avg['timepoint_days'] >= injury_ref_tp].copy()

    results = {}
    treatment_groups = sorted(recovery_data['treatment_group'].unique())

    if len(treatment_groups) == 0:
        return results

    # Statistical reference for the mixed-effects model contrast: use the
    # resolved reference_group if it's present, else fall back to the first
    # group alphabetically.
    model_reference = reference_group if reference_group in treatment_groups else treatment_groups[0]

    # TEST 1: Recovery trends using mixed-effects model
    # Model: metric ~ timepoint * treatment + (1|animal)
    if len(treatment_groups) >= 2:
        model = smf.mixedlm(
            f"{metric_name} ~ timepoint_days * C(treatment_group, Treatment(reference='{model_reference}'))",
            data=recovery_data,
            groups=recovery_data["animal_id"]
        )
        model_results = model.fit(reml=False)

        df_resid = model_results.df_resid
        params = model_results.params
        cov = model_results.cov_params()

        # Reference group slope
        ref_slope = params['timepoint_days']
        ref_se = np.sqrt(cov.loc['timepoint_days', 'timepoint_days'])
        ref_t = ref_slope / ref_se
        ref_p = model_results.pvalues['timepoint_days']

        # Convert t-statistic to correlation-like value
        ref_corr = ref_t / np.sqrt(ref_t**2 + df_resid)

        results[f'{model_reference}_corr'] = ref_corr
        results[f'{model_reference}_p'] = ref_p

        # Other treatment slopes
        for treatment in treatment_groups:
            if treatment == model_reference:
                continue

            interaction_term = f"timepoint_days:C(treatment_group, Treatment(reference='{model_reference}'))[T.{treatment}]"

            if interaction_term in params.index:
                treatment_slope = ref_slope + params[interaction_term]

                var_ref = cov.loc['timepoint_days', 'timepoint_days']
                var_interaction = cov.loc[interaction_term, interaction_term]
                cov_term = cov.loc['timepoint_days', interaction_term]
                treatment_se = np.sqrt(var_ref + var_interaction + 2 * cov_term)

                treatment_t = treatment_slope / treatment_se
                treatment_p = 2 * (1 - stats.t.cdf(abs(treatment_t), df_resid))

                treatment_corr = treatment_t / np.sqrt(treatment_t**2 + df_resid)

                results[f'{treatment}_corr'] = treatment_corr
                results[f'{treatment}_p'] = treatment_p
            else:
                results[f'{treatment}_corr'] = ref_corr
                results[f'{treatment}_p'] = ref_p
    else:
        # Only one group present - fit the trend alone, no treatment contrast possible.
        model = smf.mixedlm(f"{metric_name} ~ timepoint_days", data=recovery_data,
                             groups=recovery_data["animal_id"])
        model_results = model.fit(reml=False)
        df_resid = model_results.df_resid
        slope = model_results.params['timepoint_days']
        se = np.sqrt(model_results.cov_params().loc['timepoint_days', 'timepoint_days'])
        t = slope / se
        p = model_results.pvalues['timepoint_days']
        corr = t / np.sqrt(t**2 + df_resid)
        results[f'{treatment_groups[0]}_corr'] = corr
        results[f'{treatment_groups[0]}_p'] = p

    # TEST 2 + 3: reference_group vs every other group (improvement, injury ref -> last tp)
    last_tp = recovery_data['timepoint_days'].max()

    if reference_group is not None and reference_group in treatment_groups:
        ref_improvements = _get_improvements(animal_tp_avg, reference_group, metric_name, injury_ref_tp, last_tp)

        for treatment in treatment_groups:
            if treatment == reference_group:
                continue

            treat_improvements = _get_improvements(animal_tp_avg, treatment, metric_name, injury_ref_tp, last_tp)

            if len(ref_improvements) > 0 and len(treat_improvements) > 0:
                # Permutation test (one-tailed: treatment > reference)
                observed_diff = np.mean(treat_improvements) - np.mean(ref_improvements)
                all_improvements = np.concatenate([ref_improvements, treat_improvements])
                n_ref = len(ref_improvements)

                np.random.seed(2025)
                shuffled_diffs = []
                for _ in range(10000):
                    shuffled = np.random.permutation(all_improvements)
                    fake_diff = np.mean(shuffled[n_ref:]) - np.mean(shuffled[:n_ref])
                    shuffled_diffs.append(fake_diff)

                p_perm = max(np.mean(np.array(shuffled_diffs) >= observed_diff), 0.0001)

                # ANOVA (two-tailed)
                f_stat, p_anova = f_oneway(ref_improvements, treat_improvements)

                # Cohen's d
                mean_r = np.mean(ref_improvements)
                mean_t = np.mean(treat_improvements)
                var_r = np.var(ref_improvements, ddof=1)
                var_t = np.var(treat_improvements, ddof=1)
                pooled_std = np.sqrt((var_r + var_t) / 2)
                cohens_d = (mean_t - mean_r) / pooled_std if pooled_std > 0 else np.nan

                results[f'{reference_group}_vs_{treatment}_p_permutation'] = p_perm
                results[f'{reference_group}_vs_{treatment}_p_anova'] = p_anova
                results[f'{reference_group}_vs_{treatment}_cohens_d'] = cohens_d
            else:
                results[f'{reference_group}_vs_{treatment}_p_permutation'] = np.nan
                results[f'{reference_group}_vs_{treatment}_p_anova'] = np.nan
                results[f'{reference_group}_vs_{treatment}_cohens_d'] = np.nan

    return results


def plot_recovery_curves(df: pd.DataFrame, metrics: list, group_colors: dict,
                          injury_ref_tp, reference_group) -> tuple:
    """
    Plot recovery curves for specified metrics with statistics overlaid.

    Returns
    -------
    tuple : (fig, all_stats)
    """
    import matplotlib.pyplot as plt

    available_metrics = [m for m in metrics if m in df.columns]
    n_metrics = len(available_metrics)

    if n_metrics == 0:
        print("ERROR: No metrics found")
        return None, []

    animal_tp_avg = df.groupby(['animal_id', 'treatment_group', 'timepoint_days'])[available_metrics].mean().reset_index()

    timepoints = sorted(df['timepoint_days'].unique())
    treatment_groups = sorted(df['treatment_group'].unique())

    # Grid sizing adapts to n_metrics instead of assuming exactly 4 or 6
    n_cols = 2 if n_metrics > 1 else 1
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16 if n_cols > 1 else 8, 4 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    all_stats = []

    for idx, metric_name in enumerate(available_metrics):
        ax = axes[idx]

        for treatment in treatment_groups:
            treatment_data = animal_tp_avg[animal_tp_avg['treatment_group'] == treatment]

            mean_vals = []
            sem_vals = []

            for tp in timepoints:
                tp_data = treatment_data[treatment_data['timepoint_days'] == tp][metric_name]
                if len(tp_data) > 0:
                    mean_vals.append(tp_data.mean())
                    sem_vals.append(tp_data.std() / np.sqrt(len(tp_data)))
                else:
                    mean_vals.append(np.nan)
                    sem_vals.append(np.nan)

            ax.errorbar(timepoints, mean_vals, yerr=sem_vals,
                       marker='o', linewidth=2, markersize=6,
                       capsize=4, label=treatment, color=group_colors.get(treatment, 'gray'),
                       alpha=0.8)

        stats_dict = compute_metric_stats(df, metric_name, injury_ref_tp, reference_group)

        stats_row = {'Metric': metric_name}
        stats_lines = []

        for treatment in treatment_groups:
            corr = stats_dict.get(f'{treatment}_corr', np.nan)
            p_val = stats_dict.get(f'{treatment}_p', np.nan)

            if not (isinstance(corr, float) and np.isnan(corr)):
                sig = '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                stats_lines.append(f"{treatment}: r={corr:.3f}{sig}")
                stats_row[f'{treatment}_recovery_trend'] = f"{corr:.4f}{sig}"

        if reference_group is not None:
            for treatment in treatment_groups:
                if treatment == reference_group:
                    continue

                p_perm = stats_dict.get(f'{reference_group}_vs_{treatment}_p_permutation', np.nan)
                cohens_d = stats_dict.get(f'{reference_group}_vs_{treatment}_cohens_d', np.nan)
                p_anova = stats_dict.get(f'{reference_group}_vs_{treatment}_p_anova', np.nan)

                if not np.isnan(p_perm):
                    sig = '***' if p_perm < 0.001 else '**' if p_perm < 0.01 else '*' if p_perm < 0.05 else '†' if p_perm < 0.1 else ''
                    if not np.isnan(cohens_d):
                        stats_lines.append(f"{reference_group} vs {treatment}: p={p_perm:.4f}{sig}, d={cohens_d:.2f}")
                    stats_row[f'{reference_group}_vs_{treatment}_p_permutation'] = f"{p_perm:.4f}{sig}"

                if not np.isnan(p_anova):
                    sig2 = '**' if p_anova < 0.01 else '*' if p_anova < 0.05 else ''
                    stats_row[f'{reference_group}_vs_{treatment}_p_anova'] = f"{p_anova:.4f}{sig2}"

                stats_row[f'{reference_group}_vs_{treatment}_cohens_d'] = cohens_d

        all_stats.append(stats_row)

        if stats_lines:
            stats_text = '\n'.join(stats_lines)
            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_ylabel('Recovery Score', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(timepoints)

        if idx == 0:
            ax.legend(title='Treatment', fontsize=9, title_fontsize=10, loc='lower right')

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    for idx in range(max(0, n_metrics - n_cols), n_metrics):
        axes[idx].set_xlabel('Days Post-Injury', fontsize=12, fontweight='bold')

    fig.suptitle("Recovery Metrics and Significance", fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    return fig, all_stats


def plot_recovery_bars(df: pd.DataFrame, metrics: list, group_colors: dict) -> tuple:
    """
    Plot recovery bars for specified metrics with statistics overlaid.

    Returns
    -------
    tuple : (figs, all_plots_df)
    """
    import matplotlib.pyplot as plt

    all_plot_dfs = {}
    figs = {}

    timepoints = sorted(df['timepoint_days'].unique())
    treatment_groups = sorted(df['treatment_group'].unique())

    for metric in metrics:
        if metric not in df.columns:
            print(f"  WARNING: Metric '{metric}' not found in recovery scores, skipping...")
            continue

        print(f"  Plotting {metric}...")

        plot_data = []

        for treatment in treatment_groups:
            for tp in timepoints:
                mask = (df['treatment_group'] == treatment) & (df['timepoint_days'] == tp)
                values = df.loc[mask, f'{metric}'].values
                values = values[~np.isnan(values)]

                if len(values) > 0:
                    mean_val = np.mean(values)
                    sem_val = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
                    plot_data.append({
                        'treatment': treatment,
                        'timepoint': tp,
                        'mean': mean_val,
                        'sem': sem_val,
                        'n': len(values)
                    })

        plot_df = pd.DataFrame(plot_data)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(timepoints))
        width = 0.25
        n_treatments = len(treatment_groups)

        for i, treatment in enumerate(treatment_groups):
            treatment_data = plot_df[plot_df['treatment'] == treatment]

            means = []
            sems = []
            for tp in timepoints:
                tp_data = treatment_data[treatment_data['timepoint'] == tp]
                if len(tp_data) > 0:
                    means.append(tp_data['mean'].values[0])
                    sems.append(tp_data['sem'].values[0])
                else:
                    means.append(0)
                    sems.append(0)

            offset = (i - n_treatments / 2 + 0.5) * width

            ax.bar(x + offset, means, width, label=treatment,
                   color=group_colors.get(treatment, '#808080'),
                   alpha=0.8, edgecolor='black', linewidth=1)

            ax.errorbar(x + offset, means, yerr=sems, fmt='none',
                       color='black', capsize=5, capthick=1.5, linewidth=1.5)

        y_min = min(plot_df['mean'] - plot_df['sem'])
        y_max = max(plot_df['mean'] + plot_df['sem'])

        ax.set_xlabel('Timepoint (days post-injury)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric} (-100 to 100 scale)', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Recovery Scores by Treatment Group',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(tp)}' for tp in timepoints], fontsize=11)
        ax.set_ylim(y_min - 10, y_max + 10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend(title='Treatment Group', fontsize=10, title_fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        figs[metric] = fig
        all_plot_dfs[metric] = plot_df

    return figs, all_plot_dfs


def run_visualize_scores(config: dict, profile: DatasetProfile, df: pd.DataFrame = None) -> None:
    """
    Runs the full Stage 4 workflow: load recovery scores, plot recovery
    curves + bar plots, run statistics, save everything to disk.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10

    viz_cfg = config.get('visualize_scores', {})

    rp = _resolve_project_paths(config)
    OUTPUT_DIR = rp['output_dir']
    RECOVERY_DIR = OUTPUT_DIR / "recovery_scores"
    ANALYSIS_DIR = OUTPUT_DIR / "recovery_score_analysis"
    RECOVERY_DIR.mkdir(exist_ok=True, parents=True)
    ANALYSIS_DIR.mkdir(exist_ok=True, parents=True)

    INPUT_FILE = RECOVERY_DIR / "recovery_scores.csv"
    configured_metrics = viz_cfg.get('metrics_to_plot', ['CKS', 'Mahalanobis', 'kNN_Baseline', 'Ridge'])
    configured_colors = viz_cfg.get('group_colors', {})

    print("=" * 60)
    print("RECOVERY SCORE VISUALIZATION AND ANALYSIS")
    print("=" * 60)
    print()

    if df is None:
        print(f"Loading data from: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)

    available_metrics = [m for m in configured_metrics if m in df.columns]
    print(f"  Metrics available: {available_metrics}")

    if len(available_metrics) == 0:
        print("  No recovery score metrics found in the input file; nothing to visualize. Skipping Stage 4.")
        return

    injury_ref_tp = profile.resolved_injury_reference_timepoint
    if injury_ref_tp is None or not profile.has_multiple_timepoints:
        print("  No injury reference timepoint or fewer than 2 distinct timepoints present in "
              "the data; recovery curves and recovery statistics require a real time axis and "
              "are being skipped.")
        return

    treatment_groups = sorted(df['treatment_group'].unique())
    group_colors = _resolve_group_colors(configured_colors, treatment_groups)
    reference_group = profile.resolved_reference_group

    print()
    print("=" * 60)
    print("Generating Recovery Curves")
    print("=" * 60)

    fig_main, stats_main = plot_recovery_curves(df, available_metrics, group_colors, injury_ref_tp, reference_group)

    if fig_main is not None:
        output_file_main = ANALYSIS_DIR / f"recovery_curves_{len(available_metrics)}.png"
        fig_main.savefig(output_file_main, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file_main}")

    print()
    print("=" * 60)
    print("Exporting Statistics")
    print("=" * 60)

    stats_df = pd.DataFrame(stats_main)

    # Column order built dynamically from whatever groups are present,
    # instead of a hardcoded Vehicle/LoDose/HiDose list.
    col_order = ['Metric'] + [f'{g}_recovery_trend' for g in treatment_groups]
    if reference_group is not None:
        for g in treatment_groups:
            if g == reference_group:
                continue
            col_order += [
                f'{reference_group}_vs_{g}_p_permutation',
                f'{reference_group}_vs_{g}_p_anova',
                f'{reference_group}_vs_{g}_cohens_d',
            ]
    col_order = [c for c in col_order if c in stats_df.columns]
    stats_df = stats_df[col_order]

    excel_file = ANALYSIS_DIR / "recovery_statistics.xlsx"
    stats_df.to_excel(excel_file, index=False)
    print(f"Saved: {excel_file}")

    print("\nStatistical Results:")
    print(stats_df.to_string(index=False))

    print()
    print("=" * 60)
    print("Generating Recovery Bar Plots")
    print("=" * 60)

    figs_bars, bars_data = plot_recovery_bars(df, available_metrics, group_colors)

    for metric, fig in figs_bars.items():
        out = ANALYSIS_DIR / f"recovery_bars_{metric}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")


# #############################################################################
# 7. STAGE 5 - VISUALIZE & ANALYZE FEATURE IMPORTANCE
# #############################################################################
# Takes the feature importance files produced by Stage 3 and answers: 
# which of the original gait signals has the greatest
# significance regarding recovery scores?
#
#   Part A - Consensus bar chart:
#     Shows top N features, with each bar colored by which metric
#     contributed the most. Features are color-coded by original signal type.
#
#   Part B - Heatmap by original features:
#     Aggregates TSFresh sub-features back up to original signal. Heatmap
#     shows which original signals matter per metric.
#
#   Part C - TSFresh feature glossary:
#     Prints a reference table explaining what each TSFresh feature name
#     means.
#
# Output files created:
#   - outputs/feature_imp_analysis/feature_importance.csv
#   - outputs/feature_imp_analysis/consensus_importance.png
#   - outputs/feature_imp_analysis/consensus_heatmap_by_original.png

def get_original_feature_name(feat_name: str) -> str:
    """Extract original feature name (before first '__')."""
    return feat_name.split('__')[0] if '__' in feat_name else feat_name


def create_interpretable_label(feat_name: str, max_length: int = 60) -> str:
    """Create readable label: 'shoulder_angle_right__mean' -> 'Shoulder Angle Right | Mean'"""
    parts = feat_name.split('__')

    if len(parts) == 1:
        return parts[0].replace('_', ' ').title()

    original = parts[0].replace('_', ' ').title()
    tsfresh = parts[1].replace('_', ' ').title()
    label = f"{original} | {tsfresh}"

    if len(label) > max_length:
        label = label[:max_length - 3] + '...'

    return label


def select_top_features_balanced(importance_series: pd.Series,
                                 top_n: int = 30,
                                 top_k_per_original: int = 3) -> pd.Series:
    """
    Select top features while limiting per original feature to prevent domination.
    """
    original_groups = {}
    for feat_name, importance in importance_series.items():
        original = get_original_feature_name(feat_name)
        if original not in original_groups:
            original_groups[original] = []
        original_groups[original].append((feat_name, importance))

    selected_features = []
    for original_feat, features in original_groups.items():
        features_sorted = sorted(features, key=lambda x: abs(x[1]), reverse=True)
        top_k = features_sorted[:top_k_per_original]
        selected_features.extend(top_k)

    selected_series = pd.Series({feat: imp for feat, imp in selected_features})
    top_n_series = selected_series.abs().nlargest(top_n)

    return selected_series.loc[top_n_series.index]


def get_root_color_key(feat_name: str) -> str:
    """Root color key = first two words of the original feature name."""
    root = get_original_feature_name(feat_name)
    words = root.replace('_', ' ').title().split()
    return ' '.join(words[:2])


# TSFresh feature glossary

TSFRESH_GLOSSARY = [
    ('abs_energy', 'Energy (sum of squared values)'),
    ('absolute_maximum', 'Max absolute value'),
    ('absolute_sum_of_changes', 'Total variation (sum of abs changes)'),
    ('agg_autocorrelation', 'Autocorrelation statistics'),
    ('agg_linear_trend', 'Linear trend in aggregated chunks'),
    ('approximate_entropy', 'Approximate entropy (regularity measure)'),
    ('ar_coefficient', 'Autoregressive coefficient (predictability)'),
    ('augmented_dickey_fuller', 'Unit root test (stationarity)'),
    ('autocorrelation', 'Autocorrelation at lag'),
    ('benford_correlation', 'Benford correlation (anomaly detection)'),
    ('binned_entropy', 'Entropy of binned distribution'),
    ('c3', 'Non-linearity measure'),
    ('change_quantiles', 'Stats in quantile range (ql to qh)'),
    ('cid_ce', 'Complexity estimate (peaks/valleys)'),
    ('count_above', 'Percentage above threshold'),
    ('count_above_mean', 'Count above mean'),
    ('count_below', 'Percentage below threshold'),
    ('count_below_mean', 'Count below mean'),
    ('cwt_coefficients', 'Wavelet transform coefficients'),
    ('energy_ratio_by_chunks', 'Energy ratio per chunk'),
    ('fft_aggregated', 'FFT spectrum stats (centroid, variance, skew, kurtosis)'),
    ('fft_coefficient', 'FFT coefficient (frequency component)'),
    ('first_location_of_maximum', 'First max location'),
    ('first_location_of_minimum', 'First min location'),
    ('fourier_entropy', 'Spectral entropy (power spectrum)'),
    ('friedrich_coefficients', 'Langevin model coefficients (dynamics)'),
    ('has_duplicate', 'Has duplicate values'),
    ('has_duplicate_max', 'Max appears multiple times'),
    ('has_duplicate_min', 'Min appears multiple times'),
    ('index_mass_quantile', 'Index where q% of mass lies'),
    ('kurtosis', 'Kurtosis (tail heaviness)'),
    ('large_standard_deviation', 'Large std dev indicator'),
    ('last_location_of_maximum', 'Last max location'),
    ('last_location_of_minimum', 'Last min location'),
    ('lempel_ziv_complexity', 'LZ complexity (compressibility)'),
    ('linear_trend', 'Linear trend slope'),
    ('linear_trend_timewise', 'Timewise linear trend'),
    ('longest_strike_above_mean', 'Longest sequence above mean'),
    ('longest_strike_below_mean', 'Longest sequence below mean'),
    ('matrix_profile', 'Matrix profile (subsequence similarity)'),
    ('max_langevin_fixed_point', 'Langevin fixed point (dynamics)'),
    ('mean_abs_change', 'Mean absolute change'),
    ('mean_change', 'Mean change'),
    ('mean_n_absolute_max', 'Mean of n largest values'),
    ('mean_second_derivative_central', 'Mean 2nd derivative (acceleration)'),
    ('number_crossing_m', 'Number of crossings of m'),
    ('number_cwt_peaks', 'Wavelet peaks'),
    ('partial_autocorrelation', 'Partial autocorrelation'),
    ('percentage_of_reoccurring_datapoints_to_all_datapoints', 'Reoccurring points %'),
    ('percentage_of_reoccurring_values_to_all_values', 'Reoccurring values %'),
    ('permutation_entropy', 'Permutation entropy (complexity/regularity)'),
    ('quantile', 'Quantile value'),
    ('query_similarity_count', 'Similar subsequence count'),
    ('range_count', 'Count in range'),
    ('ratio_beyond_r_sigma', 'Ratio beyond r*std (outliers)'),
    ('ratio_value_number_to_time_series_length', 'Uniqueness ratio'),
    ('root_mean_square', 'RMS'),
    ('sample_entropy', 'Sample entropy (regularity)'),
    ('skewness', 'Skewness'),
    ('spkt_welch_density', 'Power spectral density (frequency content)'),
    ('sum_of_reoccurring_data_points', 'Sum reoccurring points'),
    ('sum_of_reoccurring_values', 'Sum reoccurring values'),
    ('symmetry_looking', 'Symmetry indicator'),
    ('time_reversal_asymmetry_statistic', 'Time reversal asymmetry'),
    ('value_count', 'Value count'),
    ('variance_larger_than_standard_deviation', 'Variance > std'),
    ('variation_coefficient', 'CV (std/mean)'),
]


def print_glossary() -> None:
    """Print the TSFresh glossary as a formatted reference table."""
    print(f"{'Feature':<50} | Description")
    print("-" * 100)
    for feature, desc in TSFRESH_GLOSSARY:
        print(f"{feature:<50} | {desc}")


def run_feature_importance(config: dict) -> None:
    """
    Runs the full Stage 5 workflow: load whichever per-metric feature
    importance files Stage 3 produced, build a consensus ranking, plot the
    bar chart + heatmap, print the TSFresh glossary.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10

    fi_cfg = config.get('feature_importance', {})

    rp = _resolve_project_paths(config)
    OUTPUT_DIR_ROOT = rp['output_dir']
    BASE_DIR = OUTPUT_DIR_ROOT / "recovery_scores" / "feature_importance"
    OUTPUT_DIR = OUTPUT_DIR_ROOT / "feature_imp_analysis"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TOP_N = fi_cfg.get('top_n_features', 15)
    TOP_K_PER_ORIGINAL = fi_cfg.get('top_k_per_original_feature', 3)

    CKS_IMPORTANCE_FILE = BASE_DIR / "cks_importance.csv"
    MAHAL_IMPORTANCE_FILE = BASE_DIR / "mahalanobis_importance.csv"
    RIDGE_IMPORTANCE_FILE = BASE_DIR / "ridge_importance.csv"

    print("=" * 60)
    print("LOADING ALL FEATURE IMPORTANCE DATA")
    print("=" * 60)
    print()

    all_importance = {}

    if CKS_IMPORTANCE_FILE.exists():
        cks_imp = pd.read_csv(CKS_IMPORTANCE_FILE, index_col=0).squeeze()
        cks_abs = cks_imp.abs()
        all_importance['CKS'] = cks_abs / cks_abs.max() if cks_abs.max() > 0 else cks_abs
        print(f"  Loaded CKS: {len(cks_imp)} features")
    else:
        print(f"  CKS importance file not found (metric likely skipped in Stage 3); omitting from consensus.")

    if MAHAL_IMPORTANCE_FILE.exists():
        mahal_imp = pd.read_csv(MAHAL_IMPORTANCE_FILE, index_col=0).squeeze()
        mahal_abs = mahal_imp.abs()
        all_importance['Mahalanobis'] = mahal_abs / mahal_abs.max() if mahal_abs.max() > 0 else mahal_abs
        print(f"  Loaded Mahalanobis: {len(mahal_imp)} features")
    else:
        print(f"  Mahalanobis importance file not found (metric likely skipped in Stage 3); omitting from consensus.")

    if RIDGE_IMPORTANCE_FILE.exists():
        ridge_imp = pd.read_csv(RIDGE_IMPORTANCE_FILE, index_col=0).squeeze()
        ridge_abs = ridge_imp.abs()
        all_importance['Ridge'] = ridge_abs / ridge_abs.max() if ridge_abs.max() > 0 else ridge_abs
        print(f"  Loaded Ridge: {len(ridge_imp)} features")
    else:
        print(f"  Ridge importance file not found (metric likely skipped in Stage 3); omitting from consensus.")

    print()

    # If no importance files exist at all (e.g. only kNN was computable for
    # this dataset, and kNN never produces one), print why and stop instead
    # of raising.
    if len(all_importance) == 0:
        print("No feature importance data available; CKS, Mahalanobis, and Ridge all appear to have "
              "been skipped in Stage 3 for this dataset (kNN does not produce a feature importance "
              "file, by design). Skipping Stage 5.")
        return

    print("=" * 60)
    print("CREATING CONSENSUS IMPORTANCE")
    print("=" * 60)
    print()

    all_features = set(f for imp in all_importance.values() for f in imp.index)
    consensus_df = pd.DataFrame(index=sorted(all_features), columns=all_importance.keys())

    for method, importance in all_importance.items():
        consensus_df[method] = importance.reindex(consensus_df.index).fillna(0)

    consensus_df['Total_Contribution'] = consensus_df.sum(axis=1)

    feat_imp_file = OUTPUT_DIR / "feature_importance.csv"
    consensus_df.index.name = "TSFresh_Feature"
    consensus_df.to_csv(feat_imp_file, index=True)
    print(f"Saved feature importance to: {feat_imp_file}")
    print()

    print(f"Total features: {len(consensus_df)}")
    print()

    print(f"Limiting to top {TOP_K_PER_ORIGINAL} per original feature...")
    selected_features = []
    original_groups = {}

    for feat in consensus_df.index:
        original = get_original_feature_name(feat)
        original_groups.setdefault(original, []).append(feat)

    for feats in original_groups.values():
        feats_sorted = sorted(feats, key=lambda f: consensus_df.loc[f, 'Total_Contribution'], reverse=True)
        selected_features.extend(feats_sorted[:TOP_K_PER_ORIGINAL])

    top_consensus = consensus_df.loc[selected_features].sort_values('Total_Contribution', ascending=False).head(TOP_N)

    print(f"Selected features: {len(selected_features)}")
    print(f"\nUsing top {TOP_N} for visualization:")
    print(f"  Original features: {len(set([get_original_feature_name(f) for f in top_consensus.index]))}")
    print()

    print("=" * 60)
    print("CONSENSUS IMPORTANCE - STACKED BAR CHART")
    print("=" * 60)
    print()

    top_consensus_plot = top_consensus.iloc[::-1]
    methods_ordered = [m for m in ['CKS', 'Mahalanobis', 'Ridge'] if m in top_consensus_plot.columns]
    y_pos = np.arange(len(top_consensus_plot))

    method_colors = {'CKS': '#ff7f0e', 'Mahalanobis': '#2ca02c', 'Ridge': '#1f77b4'}

    color_keys = [get_root_color_key(f) for f in top_consensus_plot.index]
    unique_keys = list(dict.fromkeys(color_keys))

    cmap = plt.cm.Pastel1
    root_colors = {key: mcolors.to_hex(cmap(i % cmap.N)) for i, key in enumerate(unique_keys)}

    fig, ax = plt.subplots(figsize=(16, 8))

    bottom = np.zeros(len(top_consensus_plot))
    for method in methods_ordered:
        vals = top_consensus_plot[method].values
        ax.barh(
            y_pos,
            vals,
            left=bottom,
            color=method_colors.get(method, '#808080'),
            edgecolor='white',
            linewidth=0.5,
            alpha=0.85,
            label=method
        )
        bottom += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels([''] * len(y_pos))

    for i, feat_name in enumerate(top_consensus_plot.index):
        full_label = create_interpretable_label(feat_name)
        parts = full_label.split('|')
        tsfresh_text = parts[1].strip() if len(parts) > 1 else ""

        color_key = get_root_color_key(feat_name)
        color = root_colors[color_key]

        ax.text(
            -0.01 * top_consensus_plot[methods_ordered].sum(axis=1).max(),
            y_pos[i],
            full_label,
            fontsize=14,
            ha='right',
            va='center',
            color='black',
            zorder=11,
            bbox=dict(facecolor=color, alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
        )

        ax.text(
            -0.02, y_pos[i],
            f"| {tsfresh_text}",
            fontsize=14,
            ha='right',
            va='center',
            color='black',
            backgroundcolor='white',
            zorder=12
        )

    ax.set_xlabel('Total Normalized Importance', fontsize=18, fontweight='bold')
    ax.set_title(
        "Consensus Importance of Top Features Across Methods",
        fontsize=20,
        fontweight='bold',
        pad=20
    )
    ax.legend(title='Method', loc='lower right', fontsize=14, title_fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.subplots_adjust(left=0.28)
    plt.tight_layout()

    output_file = OUTPUT_DIR / 'consensus_importance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    print()
    print("=" * 60)
    print("CONSENSUS HEATMAP - BY ORIGINAL FEATURE")
    print("=" * 60)
    print()

    method_cols = list(all_importance.keys())

    original_df = consensus_df[method_cols].copy()
    original_df['Original'] = [get_original_feature_name(f) for f in original_df.index]
    original_df = original_df.groupby('Original').mean()

    original_df_norm = original_df / original_df.max()

    original_df_norm['Total'] = original_df_norm.mean(axis=1)
    original_df_norm_sorted = original_df_norm.sort_values('Total', ascending=False)
    heatmap_data = original_df_norm_sorted[method_cols]

    plt.figure(figsize=(10, max(6, len(heatmap_data) * 0.35)))
    sns.set(style="white")
    ax = sns.heatmap(
        heatmap_data,
        cmap='OrRd',
        cbar_kws={'label': '\nNormalized Importance (0-1)'},
        linewidths=0.5,
        linecolor='gray',
        annot=False,
        fmt=".2f"
    )

    ax.set_xlabel("\nMethod", fontsize=15, fontweight='bold')
    ax.set_ylabel("Original Feature\n", fontsize=15, fontweight='bold')

    ax.set_title(
        'Average Contribution by Original Feature',
        fontsize=20, fontweight='bold', pad=15
    )

    ax.set_xticklabels(method_cols, fontsize=14, fontweight='normal')
    ax.set_yticklabels([name.replace('_', ' ').title() for name in heatmap_data.index], fontsize=14)

    plt.tight_layout()

    output_file = OUTPUT_DIR / 'consensus_heatmap_by_original.png'
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"Saved: {output_file}")
    print(f"  Showing {len(heatmap_data)} original features")
    print()

    print("=" * 60)
    print("TSFRESH FEATURE GLOSSARY")
    print("=" * 60)
    print_glossary()
