import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42


def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def plot_distance_correlation_heatmap(mat, pvals, title="Distance Correlation Heatmap",
                                      cmap="Blues", figsize=(12, 11), show_values=True):
    """
    Plot a heatmap of distance correlations with significance annotations.

    Parameters
    ----------
    mat : pd.DataFrame
        Symmetric matrix of distance correlations.
    pvals : pd.DataFrame
        Symmetric matrix of permutation-test p-values (same shape as mat).
    title : str, optional
        Title of the plot.
    cmap : str, optional
        Colormap for heatmap.
    show_values : bool, optional
        If True, annotates each cell with correlation + significance stars.
    """

    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    # Build annotated matrix for display
    annot = mat.copy().astype(str)
    for i in mat.index:
        for j in mat.columns:
            if pd.notnull(mat.loc[i, j]):
                star = significance_stars(pvals.loc[i, j])
                annot.loc[i, j] = f"{mat.loc[i, j]:.2f}{star}"

    # Create mask for upper triangle
    #mask = np.triu(np.ones_like(mat, dtype=bool))
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)  # k=1 excludes diagonal from mask

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        mat.astype(float),
        mask=mask,
        annot=annot if show_values else False,
        fmt="",
        cmap=cmap,
        vmin=0, vmax=1,
        square=True,
        cbar_kws={'label': 'Distance correlation'},
        annot_kws={'fontsize': 14}, 
        ax=ax
    )

    # Set axis tick label sizes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)

    # Set colorbar font sizes
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Distance correlation', fontsize=14)
    
    ax.set_title(f"{title}\n(Distance covariance independence test : * p<0.05, ** p<0.01, *** p<0.001)", fontsize=14)
    fig.tight_layout()

    return fig
