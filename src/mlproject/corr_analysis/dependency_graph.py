"""
Functions for evaluating dependencies between feature sets and target variables using RandomForestRegressor and K-Fold cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

DEFAULT_REGRESSION_METRICS = {
    "R2": make_scorer(r2_score),
    "MAE": make_scorer(lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)),
    "RMSE": make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    ),
    "MAPE": make_scorer(
        lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred)
    ),
}


def evaluate_feature_set_to_feature(
    X_source: pd.DataFrame,
    Y_targets: pd.DataFrame,
    n_splits: int = 5,
    scoring: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train RandomForestRegressor models to predict each column in Y_targets from X_source
    using K-Fold cross-validation.

    Parameters:
        X_source (pd.DataFrame): Feature matrix.
        Y_targets (pd.DataFrame): DataFrame of target features.
        n_splits (int): Number of folds for CV.
        scoring (dict): Dictionary of scoring functions for cross_validate.

    Returns:
        metrics_df (pd.DataFrame): Per-fold metrics for each target feature.
        summary_df (pd.DataFrame): Mean ± std metrics for each target feature.
    """
    if scoring is None:
        scoring = DEFAULT_REGRESSION_METRICS

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_all = []

    for col in Y_targets.columns:
        model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

        results = cross_validate(
            model,
            X_source,
            Y_targets[col],
            cv=kf,
            scoring=scoring,
            return_train_score=False,
        )

        for fold_idx in range(n_splits):
            fold_metrics = {
                metric: results[f"test_{metric}"][fold_idx] for metric in scoring
            }
            fold_metrics["Target_Feature"] = col
            fold_metrics["Fold"] = fold_idx + 1
            metrics_all.append(fold_metrics)

    metrics_df = pd.DataFrame(metrics_all)

    # Group by target and compute mean and std
    summary_df = (
        metrics_df.drop(columns=["Fold"]).groupby("Target_Feature").agg(["mean", "std"])
    )

    # Flatten MultiIndex for convenience
    summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]

    # Add a "mean ± std" column for each metric
    for metric in scoring.keys():
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        summary_df[f"{metric}_mean±std"] = (
            summary_df[mean_col].round(4).astype(str)
            + " ± "
            + summary_df[std_col].round(4).astype(str)
        )

    return metrics_df, summary_df


def evaluate_features_to_target(
    X_source: pd.DataFrame,
    y_target: pd.Series | np.ndarray,
    n_splits: int = 5,
    scoring: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict a target from features using KFold cross-validation.

    Parameters:
        X_source (pd.DataFrame): Feature matrix.
        y_target (pd.Series or np.array): Target values.
        n_splits (int): Number of CV splits.
        scoring (dict): Dictionary of scoring functions for cross_validate.

    Returns:
        metrics_df (pd.DataFrame): Per-fold metrics.
        summary_df (pd.DataFrame): Mean ± std of metrics across folds.
    """
    if scoring is None:
        scoring = DEFAULT_REGRESSION_METRICS

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = cross_validate(
        model, X_source, y_target, cv=cv, scoring=scoring, return_train_score=False
    )

    # Convert results to DataFrame
    metrics_df = pd.DataFrame(
        {k.replace("test_", ""): v for k, v in results.items() if k.startswith("test_")}
    )
    metrics_df["Fold"] = range(1, n_splits + 1)

    summary_df = metrics_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summary_df["mean ± std"] = (
        summary_df["mean"].round(4).astype(str)
        + " ± "
        + summary_df["std"].round(4).astype(str)
    )

    return metrics_df, summary_df


def evaluate_feature_set_relationships(
    X_lob: pd.DataFrame,
    X_matminer: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model=None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    scoring: dict | None = None,
) -> pd.DataFrame:
    """
    Computes cross-validated regression metrics (mean ± std) between two feature sets and a target.

    Evaluates four relationships:
        1. X_lob → y
        2. X_matminer → y
        3. X_lob → X_matminer
        4. X_matminer → X_lob

    Parameters
    ----------
    X_lob : pd.DataFrame or np.ndarray
        Feature matrix for the first feature group (e.g. Lobster).
    X_matminer : pd.DataFrame or np.ndarray
        Feature matrix for the second feature group (e.g. Matminer).
    y : pd.Series or np.ndarray
        Target variable.
    model : estimator, optional
        Base regressor (default: RandomForestRegressor).
    n_splits : int, optional
        Number of CV splits (default: 5).
    random_state : int, optional
        Random seed (default: 42).
    n_jobs : int, optional
        Number of parallel jobs (default: -1).
    scoring : dict, optional
        Dict of scoring functions (name -> scorer). Default: R², MAE, RMSE, MAPE.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with mean and std for each metric and relationship.
    """

    # --- Default model ---
    if model is None:
        model = RandomForestRegressor(
            n_estimators=500, random_state=random_state, n_jobs=n_jobs
        )

    # --- Default metrics ---
    if scoring is None:
        scoring = DEFAULT_REGRESSION_METRICS

    # --- Helper: compute CV metrics ---
    def cv_metrics(model, X, y, cv):
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        out = {}
        for key in scoring.keys():
            scores = cv_results[f"test_{key}"]
            mean, std = np.mean(scores), np.std(scores)
            out[f"{key} Mean"] = mean
            out[f"{key} Std"] = std
        return out

    # --- Compute relationships ---
    results = []

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 1. X_lob → y
    res = cv_metrics(model, X_lob, y, cv)
    res.update({"From": "Lobster Features", "To": f"{y.name}"})
    results.append(res)

    # 2. X_matminer → y
    res = cv_metrics(model, X_matminer, y, cv)
    res.update({"From": "Matminer Features", "To": f"{y.name}"})
    results.append(res)

    # 3. X_lob → X_matminer (multioutput)
    res = cv_metrics(model, X_lob, X_matminer, cv)
    res.update({"From": "Lobster Features", "To": "Matminer Features"})
    results.append(res)

    # 4. X_matminer → X_lob (multioutput)
    res = cv_metrics(model, X_matminer, X_lob, cv)
    res.update({"From": "Matminer Features", "To": "Lobster Features"})
    results.append(res)

    # --- Combine and reorder columns ---
    df = pd.DataFrame(results)[
        ["From", "To"] + [col for col in results[0] if col not in ["From", "To"]]
    ]
    return df
