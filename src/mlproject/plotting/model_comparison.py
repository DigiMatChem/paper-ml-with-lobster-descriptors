"""
Functions for visualizing model performance comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_errors(
    error_lists: list[list[float] | np.ndarray, list[float] | np.ndarray],
    labels: list[str],
    plot_type: str = "boxplot",
    bins: int = 40,
    figsize: tuple = (10, 10),
    target: str | None = None,
    target_unit: str | None = None,
    summary_ttest_df: pd.DataFrame | None = None,
    model_type: str = "rf",
    show_stats_in_title: bool = False,
):
    """
    Function to plot error distributions using boxplots, histograms, or fold-wise comparisons.

    Parameters
    ----------
    error_lists : list of list or np.array
        Error values (e.g., per-fold MAE).
    labels : list of str
        Feature set names (e.g., 'matminer', 'matminer_lob').
    plot_type : str
        'boxplot', 'hist', or 'fold_comparison'.
    bins : int
        Histogram bins.
    figsize : tuple
        Figure size.
    target : str, optional
        Target property name (e.g., 'max_pfc').
    summary_ttest_df : pandas.DataFrame, optional
        Single-row paired t-test results dataframe.
    show_stats_in_title : bool
        Whether to add t-test stats to the boxplot title.
    model_type: str
        Type of model for which errors are plotted (e.g., RF, MODNet).
    """

    if plot_type == "boxplot":
        fig = plt.figure(figsize=figsize)
        plt.boxplot(error_lists, tick_labels=labels)

        plt.xlabel("Descriptor set", fontsize=14)
        plt.ylabel(
            rf"mean MAE / fold [{target_unit}]" if target_unit else "mean MAE / fold",
            fontsize=14,
        )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # ---- Title construction ----
        title_parts = []

        if target is not None:
            title_parts.append(f"{model_type} – {target}")
        else:
            title_parts.append(f"{model_type}")

        if show_stats_in_title and summary_ttest_df is not None:
            row = summary_ttest_df.iloc[0]

            stats_line = (
                f"Paired t-test: "
                f"(p={row['p_value']:.3g}, "
                f"t={row['t_stat']:.3g})"
            )

            effect_line = (
                f"% Improvement: {row['rel_improvement']:.3g}, "
                f"d_av: {row['d_av']:.3g}"
            )

            title = f"{title_parts[0]}\n" f"{stats_line}\n" f"{effect_line}"
        else:
            title = title_parts[0]

        plt.title(title, fontsize=18)
        plt.tight_layout()

    elif plot_type == "hist":
        fig = plt.figure(figsize=figsize)

        colors = ["#fdbf6f", "#a6cee3", "#b2df8a", "#ff7f00", "#cab2d6"]

        for i, arr in enumerate(error_lists):
            arr = np.asarray(arr)
            color = colors[i % len(colors)]

            plt.hist(arr, bins=bins, alpha=0.5, color=color, label=labels[i])

            mean_val = np.mean(arr)
            plt.axvline(
                mean_val,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"{labels[i]} mean={mean_val:.3g}",
            )

        plt.xlabel(
            rf"mean MAE / fold [{target_unit}]" if target_unit else "mean MAE / fold",
            fontsize=14,
        )
        plt.ylabel("Frequency (log scale)", fontsize=14)
        plt.yscale("log")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(fontsize=14)

        hist_title = f"{model_type}"
        if target is not None:
            hist_title = f"{hist_title} – {target}"

        plt.title(hist_title, fontsize=18)
        plt.tight_layout()

    elif plot_type == "fold_comparison":
        if len(error_lists) != 2:
            raise ValueError("Fold comparison requires exactly two error arrays.")

        a1, a2 = error_lists
        name1, name2 = labels

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        x = [f"{i}" for i in range(1, len(a1) + 1)]

        axes[0].plot(x, a1, marker="o", label=name1, color="#fdbf6f")
        axes[0].plot(x, a2, marker="o", label=name2, color="#a6cee3")
        axes[0].set_ylabel(
            rf"mean MAE / fold [{target_unit}]" if target_unit else "mean MAE / fold",
            fontsize=14,
        )
        axes[0].legend(fontsize=14)
        axes[0].tick_params(axis="both", labelsize=14)

        title = f" {model_type} Fold-wise Comparison"
        if target is not None:
            title += f" – {target}"

        axes[0].set_title(title, fontsize=18)

        diffs = np.asarray(a1) - np.asarray(a2)
        diff_mean = np.mean(diffs)

        axes[1].bar(x, diffs, alpha=0.7)
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].axhline(
            diff_mean,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"mean diff={diff_mean:.3g}",
        )

        axes[1].set_ylabel(
            (
                f"mean MAE diff / fold [{target_unit}]"
                if target_unit
                else "mean MAE diff / fold"
            ),
            fontsize=14,
        )
        axes[1].set_xlabel("Fold", fontsize=14)
        axes[1].legend(fontsize=14)
        axes[1].tick_params(axis="both", labelsize=14)

        plt.tight_layout()

    else:
        raise ValueError("plot_type must be 'boxplot', 'hist', or 'fold_comparison'")

    return fig
