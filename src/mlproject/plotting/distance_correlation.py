"""
Functions for plotting distance correlation heatmaps
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


def summarize_pvalue_significance(
    pvals: pd.DataFrame,
    significance_func=significance_stars,
) -> str:
    """
    Summarize p-value significance across a matrix for plot title.

    If all cells have the same significance level, returns a summary
    like 'all cells: p < 0.01'. Otherwise, returns the legend mapping
    for significance stars.

    Parameters
    ----------
    pvals : pd.DataFrame
        Matrix of p-values.
    significance_func : callable, optional
        Function mapping a p-value to a significance string

    Returns
    -------
    str
        A summary of p-value significance.
    """
    star_matrix = pvals.applymap(significance_func)
    unique_stars = star_matrix.stack().unique()

    if len(unique_stars) == 1:
        uniform_star = unique_stars[0]

        if uniform_star == "***":
            return "all cells: p < 0.001"
        elif uniform_star == "**":
            return "all cells: p < 0.01"
        elif uniform_star == "*":
            return "all cells: p < 0.05"
        else:
            return "all cells: p ≥ 0.05 (not significant)"

    return "all cells: * p<0.05, ** p<0.01, *** p<0.001"


def plot_distance_correlation_heatmap(
    mat: pd.DataFrame,
    pvals: pd.DataFrame,
    std_mat: pd.DataFrame | None = None,
    title: str = "Distance Correlation Heatmap",
    cmap: str = "Blues",
    figsize: tuple = (12, 11),
    show_values: bool = True,
) -> plt.Figure:
    """
    Plot a heatmap of distance correlations with standard deviation and significance annotations.

    Parameters
    ----------
    mat : pd.DataFrame
        Symmetric matrix of distance correlations.
    pvals : pd.DataFrame
        Symmetric matrix of permutation-test p-values (same shape as mat).
    std_mat : pd.DataFrame, optional
        Symmetric matrix of std of distance correlations (from CV).
    title : str, optional
        Title of the plot.
    cmap : str, optional
        Colormap for heatmap.
    show_values : bool, optional
        If True, annotates each cell with correlation + significance stars.
    """

    # Build annotated matrix for display
    annot = mat.copy().astype(str)
    for i in mat.index:
        # get star strings for this row
        row_stars = pvals.loc[i].apply(significance_stars)

        # check if all star strings are identical
        row_has_variation = row_stars.nunique() > 1

        for j in mat.columns:
            if pd.notnull(mat.loc[i, j]):
                star = row_stars.loc[j] if row_has_variation else ""
                if std_mat is not None:
                    annot.loc[i, j] = (
                        f"{mat.loc[i, j]:.2f}±{std_mat.loc[i, j]:.2f}{star}"
                    )
                else:
                    annot.loc[i, j] = f"{mat.loc[i, j]:.2f}{star}"

    # Create mask for upper triangle
    mask = np.triu(
        np.ones_like(mat, dtype=bool), k=1
    )  # k=1 excludes diagonal from mask

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        mat.astype(float),
        mask=mask,
        annot=annot if show_values else False,
        fmt="",
        cmap=cmap,
        vmin=round(mat.min(axis=None), 1) - 0.1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Distance correlation"},
        annot_kws={"fontsize": 14},
        ax=ax,
    )

    # Set axis tick label sizes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)

    # Set colorbar font sizes
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Distance correlation", fontsize=14)

    # get p-value significance summary for title
    significance_summary = summarize_pvalue_significance(pvals)

    ax.set_title(
        f"{title}, ± = std across bootstrapped runs\n(Distance covariance independence test for {significance_summary})",
        fontsize=14,
    )
    fig.tight_layout()

    return fig
