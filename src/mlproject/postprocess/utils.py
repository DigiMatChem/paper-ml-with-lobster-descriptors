"""
Misc postprocessing utility functions
"""

import os
import pickle
import numpy as np
import pandas as pd
from mlproject.postprocess.t_test import corrected_resampled_ttest


def mean_absolute_percentage_error(y_true, y_pred, threshold=1e-5) -> float:
    """
    Compute mean absolute percentage error, masked

    Masking is for when y_true is zero (causing a
    divide by zero error) or when y_true is very small
    (causing a massive skewing in the absolute percentage
    error).

    **Note: THIS WILL IGNORE ALL ENTRIES WHERE y_true's
    MAGNITUDE IS less than the threshold, hence the
    MAPE score is not representative of all
    entries if the truth array contains entries with
    magnitude very close to 0.**

    Parameters
    ----------
    y_true : np.ndarray
        A 1-D array of true values
    y_pred : np.ndarray
        A 1-D array of predicted values
    threshold : float
        Entries with magnitude below this
        value will be ignored in the output.

    Returns
    -------
    float
        Mean absolute percentage error, masked
    """
    y_true = np.asarray(y_true)
    mask = np.abs(y_true) > threshold
    y_pred = np.asarray(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.mean(np.fabs((y_true - y_pred) / y_true))


def calculate_cohens_d_av(baseline_scores, new_scores):
    """
    Calculate Cohen's d_av between two sets of scores.

    Parameters
    ----------
    baseline_scores : list or np.ndarray
        A list or array of baseline scores.
    new_scores : list or np.ndarray
        A list or array of new scores.

    Returns
    -------
    d_av : float
        Cohen's d_av value.
    """
    # Convert to numpy arrays if not already

    s1 = np.array(baseline_scores)
    s2 = np.array(new_scores)

    # 1. Calculate Means
    mu1, mu2 = np.mean(s1), np.mean(s2)

    # 2. Calculate Standard Deviations
    sd1, sd2 = np.std(s1, ddof=1), np.std(s2, ddof=1)

    # 3. Calculate Mean Difference
    mean_diff = mu1 - mu2

    # 4. Calculate Average Standard Deviation
    sd_avg = (sd1 + sd2) / 2

    d_av = mean_diff / sd_avg

    return d_av


def calculate_relative_percentage_improvement(baseline_scores, new_scores):
    """
    Calculate the relative percentage improvement from baseline to new scores.

    Parameters
    ----------
    baseline_scores : list or np.ndarray
        A list or array of baseline scores.
    new_scores : list or np.ndarray
        A list or array of new scores.

    Returns
    -------
    float
        Relative percentage improvement.
    """
    baseline_mean = np.mean(baseline_scores)
    new_mean = np.mean(new_scores)

    improvement = baseline_mean - new_mean
    relative_improvement = (improvement / baseline_mean) * 100

    return relative_improvement


def caclulate_percent_folds_improved(baseline_scores, new_scores):
    """
    Calculate the percentage of folds that showed improvement from baseline to new scores.

    Parameters
    ----------
    baseline_scores : list or np.ndarray
        A list or array of baseline scores.
    new_scores : list or np.ndarray
        A list or array of new scores.

    Returns
    -------
    float
        Percentage of folds improved.
    """
    baseline_scores = np.array(baseline_scores)
    new_scores = np.array(new_scores)

    improved_folds = np.sum(new_scores < baseline_scores)
    total_folds = len(baseline_scores)

    percent_improved = (improved_folds / total_folds) * 100

    return percent_improved


def load_cv_results(
    models_dir: str,
    model_type: str,
    target_name: str,
    feat_set_type: str,
    n_folds: int,
    collect_sizes: bool = False,
):
    """
    Load cross-validation results and aggregate test MAE errors.

    Parameters
    ----------
    models_dir : str
        Base directory containing model results.
    model_type : str
        Model name/prefix (e.g., 'rf', 'modnet').
    target_name : str
        Target property name.
    feat_set_type : str
        Subfolder suffix (e.g., 'matminer', 'matminer_lob').
    n_folds : int
        Number of CV folds.
    collect_sizes : bool, optional
        If True, also return train/test set sizes per fold.

    Returns
    -------
    mean_test_errors : list of float
        Mean test error for each fold.
    fold_test_errors : list of np.ndarray
        Raw test errors for each fold.
    n_train_list : list of int (optional)
        Number of training samples per fold.
    n_test_list : list of int (optional)
        Number of test samples per fold.
    """
    mean_test_errors = []
    fold_test_errors = []
    n_train_list = []
    n_test_list = []

    base_path = os.path.join(models_dir, f"{model_type}_{target_name}_{feat_set_type}")

    for i in range(1, n_folds + 1):
        results_path = os.path.join(base_path, f"{i}_results.pkl")
        with open(results_path, "rb") as f:
            res = pickle.load(f)

        test_errors = np.asarray(res["test_errors"])
        mean_test_errors.append(np.mean(test_errors))
        fold_test_errors.append(test_errors)

        if collect_sizes:
            n_train_list.append(np.asarray(res["train_errors"]).size)
            n_test_list.append(test_errors.size)

    if collect_sizes:
        return mean_test_errors, fold_test_errors, n_train_list, n_test_list

    return mean_test_errors, fold_test_errors


def get_ttest_summary_df(
    target_name,
    models_dir,
    num_folds: int = 10,
    model_type: str = "rf",
    feature_set_types: list[str] = ["matminer", "matminer_lob"],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """
    Get t-test model summary dataframe including effect size and relative improvement.

    Parameters
    ----------
    target_name : str
        Target property name.
    models_dir : str
        Base directory containing model results.
    num_folds : int, optional
        Number of CV folds. Default is 10.
    model_type : str, optional
        Model name/prefix (e.g., 'rf', 'modnet'). Default is 'rf'.
    feature_set_types : list of str, optional
        List of feature set variants to compare. Default is ['matminer', 'matminer_lob']

    Returns
    -------
    summary_df : pd.DataFrame
        Summary dataframe with t-test results, effect size, and relative improvement.
    """

    (
        matminer_mean_test_errors,
        _matminer_fold_test_errors,
        matminer_n_train_list,
        matminer_n_test_list,
    ) = load_cv_results(
        models_dir,
        model_type,
        target_name,
        feature_set_types[0],
        num_folds,
        collect_sizes=True,
    )

    (
        matminer_lob_mean_test_errors,
        _matminer_lob_fold_test_errors,
        matminer_lob_n_train_list,
        matminer_lob_n_test_list,
    ) = load_cv_results(
        models_dir,
        model_type,
        target_name,
        feature_set_types[1],
        num_folds,
        collect_sizes=True,
    )

    t_test_results = corrected_resampled_ttest(
        matminer_mean_test_errors,
        matminer_lob_mean_test_errors,
        matminer_n_train_list,
        matminer_n_test_list,
        alpha=0.05,
        alternative=alternative,
    )

    d_av = calculate_cohens_d_av(
        matminer_mean_test_errors, matminer_lob_mean_test_errors
    )

    rel_improvement = calculate_relative_percentage_improvement(
        matminer_mean_test_errors, matminer_lob_mean_test_errors
    )

    percent_folds_improved = caclulate_percent_folds_improved(
        matminer_mean_test_errors, matminer_lob_mean_test_errors
    )
    summary_dict = {
        "t_stat": t_test_results["t_stat"],
        "df": t_test_results["df"],
        "critical_value": t_test_results["critical_value"],
        "p_value": t_test_results["p_value"],
        "significance_stars": significance_stars(t_test_results["p_value"]),
        "r_bar": t_test_results["r_bar"],
        "d_av": d_av,
        "rel_improvement": rel_improvement,
        "percent_folds_improved": percent_folds_improved,
    }

    return pd.DataFrame(summary_dict, index=[target_name])


def significance_stars(p: float) -> str:
    """
    Return significance stars based on p-value

    Parameters
    ----------
    p : float
        P-value from statistical test.
    Returns
    -------
    str
        Significance stars: '***' for p<0.001, '**' for p<0.01, '*' for p<0.05, '' otherwise.
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""
