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
