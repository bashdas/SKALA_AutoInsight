"""Compare the Target_1 baseline and FI logistic model on identical splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.target1_logistic_experiment import make_financial_features, make_model
from src.tracking import init_run


TARGET = "Target_1"
NON_FEATURE_COLUMNS = ("Time_step", "Target_1", "Target_2")


def make_baseline_features(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the baseline's raw plus one-day-change feature set."""
    raw = data.drop(columns=list(NON_FEATURE_COLUMNS)).copy()
    numeric_columns = raw.select_dtypes(include=np.number).columns
    daily_change = raw[numeric_columns].pct_change(fill_method=None)
    daily_change = daily_change.replace([np.inf, -np.inf], np.nan).add_suffix("_chg")
    return pd.concat([raw, daily_change], axis=1)


def make_baseline_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_pair(
    baseline_features: pd.DataFrame,
    candidate_features: pd.DataFrame,
    target: pd.Series,
    *,
    splits: int,
    gap: int,
    c_value: float,
    select_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    splitter = TimeSeriesSplit(n_splits=splits, gap=gap)
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(baseline_features), start=1
    ):
        baseline = make_baseline_model()
        candidate = make_model(
            len(candidate_features.columns), c_value=c_value, select_k=select_k
        )
        baseline.fit(baseline_features.iloc[train_index], target.iloc[train_index])
        candidate.fit(candidate_features.iloc[train_index], target.iloc[train_index])
        baseline_auc = float(
            roc_auc_score(
                target.iloc[valid_index],
                baseline.predict_proba(baseline_features.iloc[valid_index])[:, 1],
            )
        )
        candidate_auc = float(
            roc_auc_score(
                target.iloc[valid_index],
                candidate.predict_proba(candidate_features.iloc[valid_index])[:, 1],
            )
        )
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "baseline_auc": baseline_auc,
                "candidate_auc": candidate_auc,
                "auc_delta": candidate_auc - baseline_auc,
            }
        )
    return pd.DataFrame(rows)


def evaluate_holdout(
    baseline_features: pd.DataFrame,
    candidate_features: pd.DataFrame,
    target: pd.Series,
    *,
    gap: int,
    c_value: float,
    select_k: int,
) -> dict[str, float]:
    split_index = int(len(target) * 0.7)
    train_end = split_index - gap
    train_index = np.arange(train_end)
    valid_index = np.arange(split_index, len(target))
    baseline = make_baseline_model()
    candidate = make_model(
        len(candidate_features.columns), c_value=c_value, select_k=select_k
    )
    baseline.fit(baseline_features.iloc[train_index], target.iloc[train_index])
    candidate.fit(candidate_features.iloc[train_index], target.iloc[train_index])
    baseline_auc = float(
        roc_auc_score(
            target.iloc[valid_index],
            baseline.predict_proba(baseline_features.iloc[valid_index])[:, 1],
        )
    )
    candidate_auc = float(
        roc_auc_score(
            target.iloc[valid_index],
            candidate.predict_proba(candidate_features.iloc[valid_index])[:, 1],
        )
    )
    return {
        "baseline_auc": baseline_auc,
        "candidate_auc": candidate_auc,
        "auc_delta": candidate_auc - baseline_auc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--select-k", type=int, default=50)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data, encoding="cp949").sort_values(
        "Time_step", ignore_index=True
    )
    baseline_features = make_baseline_features(data).iloc[1:].reset_index(drop=True)
    candidate_features = make_financial_features(data).iloc[1:].reset_index(drop=True)
    target = data[TARGET].astype(int).iloc[1:].reset_index(drop=True)

    folds = evaluate_pair(
        baseline_features,
        candidate_features,
        target,
        splits=args.splits,
        gap=args.gap,
        c_value=args.c,
        select_k=args.select_k,
    )
    holdout = evaluate_holdout(
        baseline_features,
        candidate_features,
        target,
        gap=args.gap,
        c_value=args.c,
        select_k=args.select_k,
    )
    summary = {
        "baseline_cv_mean_auc": float(folds["baseline_auc"].mean()),
        "baseline_cv_std_auc": float(folds["baseline_auc"].std(ddof=0)),
        "candidate_cv_mean_auc": float(folds["candidate_auc"].mean()),
        "candidate_cv_std_auc": float(folds["candidate_auc"].std(ddof=0)),
        "cv_mean_auc_delta": float(folds["auc_delta"].mean()),
        "candidate_fold_wins": int((folds["auc_delta"] > 0).sum()),
        "holdout_baseline_auc": holdout["baseline_auc"],
        "holdout_candidate_auc": holdout["candidate_auc"],
        "holdout_auc_delta": holdout["auc_delta"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = args.output_dir / "target1_model_comparison_folds.csv"
    summary_path = args.output_dir / "target1_model_comparison_summary.json"
    folds.to_csv(folds_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run = init_run(
        name="target1-baseline-vs-fi-logistic",
        job_type="model-comparison",
        tags=["target-1", "baseline-comparison", "time-series", "gap-3"],
        mode=args.wandb_mode,
        config={
            "rows": len(target),
            "cv_splits": args.splits,
            "cv_gap": args.gap,
            "candidate_C": args.c,
            "candidate_select_k": args.select_k,
            "baseline_feature_count": len(baseline_features.columns),
            "candidate_feature_count": len(candidate_features.columns),
        },
    )
    for row in folds.to_dict(orient="records"):
        run.log(
            {
                "fold": row["fold"],
                "baseline/auc": row["baseline_auc"],
                "candidate/auc": row["candidate_auc"],
                "comparison/auc_delta": row["auc_delta"],
            }
        )
    run.summary.update(summary)
    run.log({"comparison/folds": wandb.Table(dataframe=folds)})
    if args.wandb_mode != "disabled":
        artifact = wandb.Artifact(
            name=f"{run.id}-target1-model-comparison",
            type="evaluation",
            metadata=summary,
        )
        artifact.add_file(str(folds_path))
        artifact.add_file(str(summary_path))
        run.log_artifact(artifact)
    run.finish()

    print(folds.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
