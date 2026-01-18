import matplotlib.pyplot as plt
import pandas as pd
from mlproject.plotting.distance_correlation import plot_distance_correlation_heatmap


def test_plot_distance_correlation_heatmap(test_data_dir):

    # Read data used for plotting distance correlation heatmap
    mean_mat = pd.read_json(
        test_data_dir / "corr_analysis" / "dcor_data" / "dcor_matrix.json"
    )
    mean_pvals = pd.read_json(
        test_data_dir / "corr_analysis" / "dcor_data" / "pvals_matrix.json"
    )
    std_mat = pd.read_json(
        test_data_dir / "corr_analysis" / "dcor_data" / "dcor_std_matrix.json"
    )

    target_name = "last_phdos_peak"

    fig = plot_distance_correlation_heatmap(
        mat=mean_mat,
        pvals=mean_pvals,
        std_mat=std_mat,
        title=f"Distance correlation: {target_name}",
        cmap="Blues",
        show_values=True,
    )

    assert isinstance(fig, plt.Figure)
    assert (
        fig.axes[0].get_title()
        == f"Distance correlation: {target_name}, ± = std across bootstrapped runs\n(Distance covariance independence test for all cells: p < 0.01)"
    )
    assert fig.axes[0].get_xticklabels()[0].get_text() == mean_mat.columns[0]
    assert fig.axes[0].get_yticklabels()[0].get_text() == mean_mat.index[0]
