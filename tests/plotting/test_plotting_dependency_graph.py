import pandas as pd
from mlproject.plotting.dependency_graph import (
    plot_dependency_graph_from_df,
    plot_feature_learnability,
)
import os
import numpy as np


def test_plot_dependency_graph_from_df(test_data_dir, tmp_path):

    results = pd.read_json(
        test_data_dir
        / "corr_analysis"
        / "dependency_graph_data"
        / "multioutput_regression_summary.json"
    )
    summary_a2t = pd.read_json(
        test_data_dir / "corr_analysis" / "dependency_graph_data" / "summary_a2t.json"
    )

    r2_combined_feat = np.round(summary_a2t.loc["R2", "mean"], 5)

    target_name = "max_pfc"

    plot_dependency_graph_from_df(
        results_df=results,
        feature1_name="Lobster Features",
        feature2_name="Matminer Features",
        target_name=f"{target_name}",
        metric="R2 Mean",  # or "MAE Mean", "RMSE Mean", etc.
        node_colors={
            "Lobster Features": "#a6cee3",
            "Matminer Features": "#fdbf6f",
            f"{target_name}": "#99d8c9",
        },
        title=f"(Matminer+Lobster) Features = {r2_combined_feat}",
        save_path=f"{tmp_path}/{target_name}_feat_metrics.png",
    )

    assert os.path.isfile(f"{tmp_path}/{target_name}_feat_metrics.png")


def test_plot_feature_learnability(test_data_dir, tmp_path):

    summary_m2l = pd.read_json(
        test_data_dir / "corr_analysis" / "dependency_graph_data" / "summary_m2l.json"
    )

    target_name = "max_pfc"

    plot_feature_learnability(
        results=summary_m2l,
        title="Lobster Feature Learnability from Matminer Features",
        save_path=f"{tmp_path}/{target_name}_m2l_learnability.png",
        n_feats=20,
    )

    assert os.path.isfile(f"{tmp_path}/{target_name}_m2l_learnability.png")
