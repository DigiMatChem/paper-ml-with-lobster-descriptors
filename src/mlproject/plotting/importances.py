"""
Functions for plotting feature importances
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mlproject.utils.misc import split_features


def plot_feature_importance(
    feat_imp_df: pd.DataFrame,
    target_name: str,
    figsize: tuple = (14, 10),
    title_fontsize: int = 18,
    tick_label_fontsize: int = 14,
    n_feats: int = 20,
    lob_color: str = "#a6cee3",
    default_color: str = "#fdbf6f",
    importance_type: str = "Permutation",
    model_name: str = "MODNet",
    include_err_bars: bool = False,
) -> plt.Figure:
    """
    Plot feature importances from a DataFrame.

    Parameters
    ----------
    feat_imp_df : pd.DataFrame
        DataFrame with feature importances. Must contain 'mean' and 'std' columns.
    target_name : str
        Name of the target variable.
    figsize : tuple
        Figure size.
    title_fontsize : int
        Font size for the title.
    tick_label_fontsize : int
        Font size for the tick labels.
    n_feats : int
        Number of top features to plot.
    lob_color : str
        Color for lobster features.
    default_color : str
        Color for other features.
    importance_type : str
        Type of feature importance (e.g., "Permutation", "SHAP").
    model_name : str
        Name of the model.
    include_err_bars : bool
        Whether to include error bars.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # get top n feats
    top_feats = feat_imp_df.sort_values("mean", ascending=False).head(n_feats)

    # split_features
    lob_feat, matminer_feat = split_features(top_feats.index.tolist())

    # assign colors based on feature group
    colors = [
        lob_color if idx in lob_feat else default_color for idx in top_feats.index
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        top_feats.index,
        top_feats["mean"],
        xerr=top_feats["std"] if include_err_bars else None,
        color=colors,
        ecolor="black",
        edgecolor="black",
        linewidth=0.1,
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        f"{importance_type} mean feature importance", fontsize=tick_label_fontsize
    )
    ax.set_title(
        f"{model_name} {importance_type} feature importance — {target_name}",
        fontsize=title_fontsize,
    )
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    plt.tight_layout()
    # legend handles
    lob_patch = mpatches.Patch(
        facecolor=lob_color, edgecolor="black", linewidth=0.1, label="Lobster Features"
    )
    mat_patch = mpatches.Patch(
        facecolor=default_color,
        edgecolor="black",
        linewidth=0.1,
        label="Matminer Features",
    )
    ax.legend(
        handles=[lob_patch, mat_patch], fontsize=tick_label_fontsize, loc="lower right"
    )

    return fig
