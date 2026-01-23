import pandas as pd
from mlproject.plotting.model_comparison import plot_errors

def test_plot_errors_boxplot():
    errors1 = [0.1, 0.2, 0.15, 0.18, 0.22]
    errors2 = [0.12, 0.25, 0.14, 0.2, 0.3]
    error_lists = [errors1, errors2]
    labels = ["Feature Set 1", "Feature Set 2"]

    fig = plot_errors(
        error_lists=error_lists,
        labels=labels,
        plot_type="boxplot",
        target="test_property",
        target_unit="eV",
        model_type="RF",
        show_stats_in_title=True,
        summary_ttest_df=pd.DataFrame(
            {
                "p_value": [0.03],
                "t_stat": [2.5],
                "rel_improvement": [15.0],
                "d_av": [0.05],
            }
        ),
    )
    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Descriptor set"
    assert ax.get_ylabel() == r"mean MAE / fold [eV]"
    print(ax.get_title())
    assert "RF – test_property" in ax.get_title()
    assert "Paired t-test" in ax.get_title()
    assert "% Improvement:" in ax.get_title()
    assert "p=" in ax.get_title()
    assert "t=" in ax.get_title()
    assert "d_av:" in ax.get_title()

def test_plot_errors_hist():
    errors1 = [0.1, 0.2, 0.15, 0.18, 0.22]
    errors2 = [0.12, 0.25, 0.14, 0.2, 0.3]
    error_lists = [errors1, errors2]
    labels = ["Feature Set 1", "Feature Set 2"]

    fig = plot_errors(
        error_lists=error_lists,
        labels=labels,
        plot_type="hist",
        target="test_property",
        target_unit="eV",
        model_type="RF",
    )
    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == r"mean MAE / fold [eV]"
    assert ax.get_ylabel() == "Frequency (log scale)"
    assert ax.get_yscale() == "log"
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("Feature Set 1 mean=" in text for text in legend_texts)
    assert any("Feature Set 2 mean=" in text for text in legend_texts)

def test_plot_errors_fold_comparison():
    errors1 = [0.1, 0.2, 0.15, 0.18, 0.22]
    errors2 = [0.12, 0.25, 0.14, 0.2, 0.3]

    fig = plot_errors(
        error_lists=[errors1, errors2],
        labels=["Feature Set 1", "Feature Set 2"],
        plot_type="fold_comparison",
        model_type="rf",
        target="test_property",
        target_unit="eV",
    )
    assert fig is not None
    assert len(fig.axes) == 2
    ax1, ax2 = fig.axes
    assert ax1.get_ylabel() == r"mean MAE / fold [eV]"
    assert ax2.get_ylabel() == "mean MAE diff / fold [eV]"
    assert ax2.get_xlabel() == "Fold"
    legend_texts_ax1 = [text.get_text() for text in ax1.get_legend().get_texts()]
    assert "Feature Set 1" in legend_texts_ax1
    assert "Feature Set 2" in legend_texts_ax1
    legend_texts_ax2 = [text.get_text() for text in ax2.get_legend().get_texts()]
    assert any("mean diff=" in text for text in legend_texts_ax2)
    # assert tick labels
    tick_labels = [tick.get_text() for tick in ax2.get_xticklabels()]
    assert tick_labels == ["1", "2", "3", "4", "5"]
    # assert title contains correct info
    assert fig.axes[0].get_title() == " rf Fold-wise Comparison – test_property"
