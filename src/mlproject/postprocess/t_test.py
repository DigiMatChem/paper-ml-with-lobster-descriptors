"""
Function implementing Nadeau & Bengio corrected resampled paired t-test with varying fold sizes.
"""

from scipy.stats import t
from statistics import stdev, mean, sqrt


def corrected_resampled_ttest(
    scores_model_a: list[float],
    scores_model_b: list[float],
    n_train_list: list[int],
    n_test_list: list[int],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict:
    """
    Nadeau & Bengio corrected resampled paired t-test with varying fold sizes.

    This is the same idea as the standard corrected resampled t-test, but instead
    of a single n_test/n_train ratio, it uses the *average* ratio across splits:

        r_bar = (1/m) * sum_i (n_test_i / n_train_i)

    Then:

        Var(d_bar) ≈ (1/m + r_bar) * s^2

    Parameters
    ----------
    scores_model_a : list[float]
        List of test errors (e.g., MAE) for model A across splits.
    scores_model_b : list[float]
        List of test errors (e.g., MAE) for model B across splits.
    n_train_list : list[int]
        List of training set sizes for each split.
    n_test_list : list[int]
        List of test set sizes for each split.
    alpha : float, optional
        Significance level for the test (default is 0.05).
    alternative : str, optional
        The alternative hypothesis to test. Options are "two-sided", "greater", or "less

    Returns
    ----------
    dict
       A dict that consists of the t-statistic, degrees of freedom, critical value, p-value, and average test/train ratio across splits with following keys

        * t_stat: The t-statistic value for the test
        * df: degrees of freedom
        * critical_value: Critical value for the given alpha and alternative hypothesis
        * p_value: p-value as per the specified alternative hypothesis
        * r_bar: average test/train ratio across splits

    References
    -----------
    Nadeau, C., Bengio, Y. Inference for the Generalization Error. Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366

    """
    if len(scores_model_a) != len(scores_model_b):
        raise ValueError("scores_model_a and scores_model_b must have the same length.")
    if len(scores_model_a) != len(n_train_list) or len(scores_model_a) != len(
        n_test_list
    ):
        raise ValueError("Fold size lists must match the number of splits.")

    m = len(scores_model_a)
    if m < 2:
        raise ValueError("Need at least 2 splits.")

    diffs = [a - b for a, b in zip(scores_model_a, scores_model_b)]
    d_bar = mean(diffs)
    sd = stdev(diffs)

    # average test/train ratio across splits
    ratios = [nt / ntr for nt, ntr in zip(n_test_list, n_train_list)]
    r_bar = sum(ratios) / m

    if sd == 0.0:
        t_stat = 0.0 if d_bar == 0.0 else float("inf") * (1.0 if d_bar > 0 else -1.0)
        df = m - 1
        if alternative == "two-sided":
            critical_value = t.ppf(1.0 - alpha / 2.0, df)
            p_value = 1.0 if d_bar == 0.0 else 0.0
        elif alternative == "greater":
            critical_value = t.ppf(1.0 - alpha, df)
            p_value = 1.0 if d_bar <= 0.0 else 0.0
        else:
            critical_value = t.ppf(alpha, df)
            p_value = 1.0 if d_bar >= 0.0 else 0.0
        return {
            "t_stat": t_stat,
            "df": df,
            "critical_value": critical_value,
            "p_value": p_value,
            "r_bar": r_bar,
        }

    se = sqrt((1.0 / m) + r_bar) * sd
    t_stat = d_bar / se
    df = m - 1

    if alternative == "two-sided":
        critical_value = t.ppf(1.0 - alpha / 2.0, df)
        p_value = 2.0 * t.sf(abs(t_stat), df)
    elif alternative == "greater":
        critical_value = t.ppf(1.0 - alpha, df)
        p_value = t.sf(t_stat, df)
    else:
        critical_value = t.ppf(alpha, df)
        p_value = t.cdf(t_stat, df)

    return {
        "t_stat": t_stat,
        "df": df,
        "critical_value": critical_value,
        "p_value": p_value,
        "r_bar": r_bar,
    }
