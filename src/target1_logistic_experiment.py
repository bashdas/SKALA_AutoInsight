"""Run the Target_1 financial-feature logistic-regression experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.tracking import init_run


TARGET = "Target_1"
CHANGE_WINDOWS = (1, 3, 5, 10)
ROLLING_WINDOWS = (3, 5, 10)


def make_financial_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create past-only level, change, momentum, and rolling FI features."""
    fi_columns = [column for column in data.columns if column.startswith("FI_")]
    financial = data[fi_columns].astype(float)
    blocks = [financial]

    for window in CHANGE_WINDOWS:
        blocks.append(financial.pct_change(window, fill_method=None).add_suffix(f"_ret_{window}d"))
        blocks.append(financial.diff(window).add_suffix(f"_diff_{window}d"))

    for window in ROLLING_WINDOWS:
        past = financial.shift(1)
        rolling_mean = past.rolling(window, min_periods=2).mean()
        rolling_std = past.rolling(window, min_periods=2).std()
        blocks.extend(
            [
                rolling_mean.add_suffix(f"_mean_{window}d"),
                rolling_std.add_suffix(f"_std_{window}d"),
                (financial - rolling_mean).add_suffix(f"_momentum_{window}d"),
            ]
        )

    return pd.concat(blocks, axis=1).replace([np.inf, -np.inf], np.nan)


def make_model(feature_count: int, *, c_value: float, select_k: int) -> Pipeline:
    """Build a fold-safe preprocessing and logistic-regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=min(select_k, feature_count))),
            (
                "model",
                LogisticRegression(C=c_value, max_iter=5_000, random_state=42),
            ),
        ]
    )


def evaluate_time_series(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    c_value: float,
    select_k: int,
    splits: int,
    gap: int,
) -> tuple[list[float], pd.DataFrame]:
    """Evaluate chronological folds and return fold scores and predictions."""
    rows: list[pd.DataFrame] = []
    scores: list[float] = []
    splitter = TimeSeriesSplit(n_splits=splits, gap=gap)
    for fold, (train_index, valid_index) in enumerate(splitter.split(features), start=1):
        model = make_model(len(features.columns), c_value=c_value, select_k=select_k)
        model.fit(features.iloc[train_index], target.iloc[train_index])
        probability = model.predict_proba(features.iloc[valid_index])[:, 1]
        auc = float(roc_auc_score(target.iloc[valid_index], probability))
        scores.append(auc)
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "row_index": valid_index,
                    "actual": target.iloc[valid_index].to_numpy(),
                    "probability": probability,
                }
            )
        )
    return scores, pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/test_public.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--select-k", type=int, default=50)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train, encoding="cp949").sort_values("Time_step")
    test = pd.read_csv(args.test, encoding="cp949").sort_values("Time_step")
    train = pd.concat(
        [train.copy(), pd.Series("train", index=train.index, name="_dataset")],
        axis=1,
    )
    test = pd.concat(
        [test.copy(), pd.Series("test", index=test.index, name="_dataset")],
        axis=1,
    )
    combined = pd.concat([train, test], ignore_index=True).sort_values(
        "Time_step", ignore_index=True
    )
    all_features = make_financial_features(combined)
    train_mask = combined["_dataset"].eq("train")
    features = all_features.loc[train_mask].reset_index(drop=True)
    test_features = all_features.loc[~train_mask].reset_index(drop=True)
    target = combined.loc[train_mask, TARGET].astype(int).reset_index(drop=True)

    run = init_run(
        name="target1-fi-logistic-rolling",
        job_type="training",
        tags=["target-1", "logistic-regression", "financial-features", "time-series"],
        mode=args.wandb_mode,
        config={
            "target": TARGET,
            "model": "LogisticRegression",
            "C": args.c,
            "selected_features": args.select_k,
            "input_feature_count": len(features.columns),
            "change_windows": list(CHANGE_WINDOWS),
            "rolling_windows": list(ROLLING_WINDOWS),
            "cv_splits": args.splits,
            "cv_gap": args.gap,
        },
    )

    fold_scores, oof_predictions = evaluate_time_series(
        features,
        target,
        c_value=args.c,
        select_k=args.select_k,
        splits=args.splits,
        gap=args.gap,
    )
    for fold, auc in enumerate(fold_scores, start=1):
        run.log({"cv/fold": fold, "cv/auc": auc})

    holdout_index = int(len(features) * 0.7)
    holdout_model = make_model(
        len(features.columns), c_value=args.c, select_k=args.select_k
    )
    holdout_model.fit(features.iloc[:holdout_index], target.iloc[:holdout_index])
    holdout_probability = holdout_model.predict_proba(features.iloc[holdout_index:])[:, 1]
    holdout_auc = float(
        roc_auc_score(target.iloc[holdout_index:], holdout_probability)
    )

    final_model = make_model(
        len(features.columns), c_value=args.c, select_k=args.select_k
    )
    final_model.fit(features, target)
    test_probability = final_model.predict_proba(test_features)[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "target1_fi_logistic_metrics.json"
    prediction_path = args.output_dir / "target1_fi_logistic_public_predictions.csv"
    metrics = {
        "holdout_auc": holdout_auc,
        "cv_mean_auc": float(np.mean(fold_scores)),
        "cv_std_auc": float(np.std(fold_scores)),
        "cv_fold_auc": fold_scores,
    }
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "Time_step": test["Time_step"].to_numpy(),
            "pred_1": test_probability,
        }
    ).to_csv(prediction_path, index=False)

    run.log(
        {
            "holdout/auc": holdout_auc,
            "cv/mean_auc": metrics["cv_mean_auc"],
            "cv/std_auc": metrics["cv_std_auc"],
            "cv/oof_predictions": wandb.Table(dataframe=oof_predictions),
        }
    )
    if args.wandb_mode != "disabled":
        artifact = wandb.Artifact(
            name=f"{run.id}-target1-predictions",
            type="predictions",
            metadata=metrics,
        )
        artifact.add_file(str(result_path))
        artifact.add_file(str(prediction_path))
        run.log_artifact(artifact)
    run.finish()

    print(json.dumps(metrics, indent=2))
    print(f"Predictions: {prediction_path}")


if __name__ == "__main__":
    main()
