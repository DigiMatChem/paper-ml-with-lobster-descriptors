"""
Misc utility functions
"""

import numpy as np


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
