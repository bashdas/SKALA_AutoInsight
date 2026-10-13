"""Analyze raw feature influence for both project targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from src.tracking import init_run


TARGETS = ("Target_1", "Target_2")
NON_FEATURE_COLUMNS = ("Time_step", *TARGETS)


def feature_group(feature: str) -> str:
    """Map a feature name to its source group."""
    return feature.split("_", maxsplit=1)[0]


def chronological_split(
    data: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically after validating the time key."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if data["Time_step"].duplicated().any():
        raise ValueError("Time_step contains duplicate values")

    ordered = data.sort_values("Time_step").reset_index(drop=True)
    split_index = int(len(ordered) * train_ratio)
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def univariate_auc_strength(
    train_values: pd.Series,
    valid_values: pd.Series,
    valid_target: pd.Series,
) -> tuple[float, float]:
    """Return directional AUC and direction-independent predictive strength."""
    median = train_values.median()
    scores = valid_values.fillna(median)
    auc = roc_auc_score(valid_target, scores)
    return float(auc), float(abs(auc - 0.5) * 2)


def analyze_target(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    *,
    n_estimators: int,
    permutation_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, float]:
    """Calculate complementary influence measures for one target."""
    feature_columns = [
        column for column in train.columns if column not in NON_FEATURE_COLUMNS
    ]
    x_train = train[feature_columns].replace([np.inf, -np.inf], np.nan)
    x_valid = valid[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = train[target].astype(int)
    y_valid = valid[target].astype(int)

    medians = x_train.median()
    x_train = x_train.fillna(medians).fillna(0)
    x_valid = x_valid.fillna(medians).fillna(0)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    valid_probability = model.predict_proba(x_valid)[:, 1]
    valid_auc = float(roc_auc_score(y_valid, valid_probability))

    permutation = permutation_importance(
        model,
        x_valid,
        y_valid,
        scoring="roc_auc",
        n_repeats=permutation_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    rows: list[dict[str, float | str]] = []
    for index, feature in enumerate(feature_columns):
        spearman = train[feature].corr(y_train, method="spearman")
        auc, auc_strength = univariate_auc_strength(
            train[feature], valid[feature], y_valid
        )
        rows.append(
            {
                "target": target,
                "feature": feature,
                "group": feature_group(feature),
                "spearman": float(0 if pd.isna(spearman) else spearman),
                "abs_spearman": float(0 if pd.isna(spearman) else abs(spearman)),
                "univariate_valid_auc": auc,
                "univariate_auc_strength": auc_strength,
                "rf_impurity_importance": float(model.feature_importances_[index]),
                "permutation_importance_mean": float(
                    permutation.importances_mean[index]
                ),
                "permutation_importance_std": float(
                    permutation.importances_std[index]
                ),
            }
        )

    result = pd.DataFrame(rows)
    rank_columns = (
        "abs_spearman",
        "univariate_auc_strength",
        "rf_impurity_importance",
        "permutation_importance_mean",
    )
    for column in rank_columns:
        result[f"rank_{column}"] = result[column].rank(
            ascending=False, method="average", pct=True
        )
    result["combined_rank_score"] = 1 - result[
        [f"rank_{column}" for column in rank_columns]
    ].mean(axis=1)
    result = result.sort_values(
        ["combined_rank_score", "permutation_importance_mean"], ascending=False
    ).reset_index(drop=True)
    result.insert(0, "combined_rank", np.arange(1, len(result) + 1))
    return result, valid_auc


def summarize_groups(feature_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual feature influence by data-source prefix."""
    prepared = feature_results.assign(
        positive_permutation_importance=feature_results[
            "permutation_importance_mean"
        ].clip(lower=0)
    )
    return (
        prepared.groupby(["target", "group"], as_index=False)
        .agg(
            feature_count=("feature", "size"),
            mean_abs_spearman=("abs_spearman", "mean"),
            mean_auc_strength=("univariate_auc_strength", "mean"),
            total_rf_importance=("rf_impurity_importance", "sum"),
            mean_rf_importance=("rf_impurity_importance", "mean"),
            total_permutation_importance=("permutation_importance_mean", "sum"),
            mean_permutation_importance=("permutation_importance_mean", "mean"),
            positive_permutation_importance=(
                "positive_permutation_importance",
                "sum",
            ),
            mean_combined_rank_score=("combined_rank_score", "mean"),
        )
        .sort_values(
            ["target", "total_permutation_importance"], ascending=[True, False]
        )
        .reset_index(drop=True)
    )


def log_results(
    results: pd.DataFrame,
    groups: pd.DataFrame,
    target_auc: dict[str, float],
    config: dict[str, int | float | str],
    mode: str,
) -> None:
    """Log metrics and complete result tables to W&B."""
    run = init_run(
        name="raw-feature-influence",
        job_type="feature-analysis",
        config=config,
        tags=["feature-importance", "random-forest", "time-split"],
        mode=mode,
    )
    for target, auc in target_auc.items():
        run.summary[f"{target}/valid_auc"] = auc
        top = results[results["target"] == target].head(20)
        top_table = wandb.Table(
            dataframe=top[["feature", "combined_rank_score"]]
        )
        run.log(
            {
                f"{target}/top_features": wandb.plot.bar(
                    top_table,
                    "feature",
                    "combined_rank_score",
                    title=f"{target} Top 20 Feature Influence",
                )
            }
        )

    run.log(
        {
            "feature_influence/all_features": wandb.Table(dataframe=results),
            "feature_influence/groups": wandb.Table(dataframe=groups),
        }
    )
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/analysis")
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data, encoding="cp949")
    missing_columns = set(NON_FEATURE_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    train, valid = chronological_split(data, args.train_ratio)
    all_results: list[pd.DataFrame] = []
    target_auc: dict[str, float] = {}
    for target in TARGETS:
        result, auc = analyze_target(
            train,
            valid,
            target,
            n_estimators=args.n_estimators,
            permutation_repeats=args.permutation_repeats,
            random_state=args.random_state,
        )
        all_results.append(result)
        target_auc[target] = auc

    results = pd.concat(all_results, ignore_index=True)
    groups = summarize_groups(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "raw_feature_influence.csv", index=False)
    groups.to_csv(args.output_dir / "feature_group_influence.csv", index=False)

    config: dict[str, int | float | str] = {
        "data": str(args.data),
        "rows": len(data),
        "features": len(data.columns) - len(NON_FEATURE_COLUMNS),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "train_ratio": args.train_ratio,
        "model": "RandomForestClassifier",
        "n_estimators": args.n_estimators,
        "permutation_repeats": args.permutation_repeats,
        "random_state": args.random_state,
    }
    log_results(results, groups, target_auc, config, args.wandb_mode)

    print("Validation AUC:")
    for target, auc in target_auc.items():
        print(f"  {target}: {auc:.6f}")
    for target in TARGETS:
        print(f"\n{target} top 10:")
        print(
            results.loc[
                results["target"] == target,
                ["combined_rank", "feature", "group", "combined_rank_score"],
            ]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
