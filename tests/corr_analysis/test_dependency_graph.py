import os
import numpy as np
from mlproject.data.preprocessing import get_dataset
from mlproject.training.feature_selection import get_relevant_features
from mlproject.corr_analysis.dependency_graph import (
    evaluate_feature_set_relationships,
    evaluate_feature_set_to_feature,
    evaluate_features_to_target,
)
from mlproject.utils.misc import split_features


def test_dependency_graph_data_generation(tmp_path, data_dir, num_jobs):
    data_parent_dir = data_dir

    target_name = "max_pfc"

    target, all_feat = get_dataset(
        target_name=target_name,
        feat_type="matminer_lob",
        data_parent_dir=data_parent_dir,
    )

    lob_feats, matminer_feats = split_features(feats=all_feat.columns)

    X_all = all_feat.dropna(axis=1)
    X_lob_all = all_feat.loc[:, lob_feats].dropna(axis=1)
    X_matminer_all = all_feat.loc[:, matminer_feats].dropna(axis=1)
    y = target.iloc[:, 0]

    X_l_pipe, X_lob = get_relevant_features(
        X_train=X_lob_all,
        y_train=y,
        grootcv_n_iter=1,
        grootcv_nfolds=2,
        **{"all_rel_feats__n_jobs": num_jobs},
    )
    X_m_pipe, X_matminer = get_relevant_features(
        X_train=X_matminer_all,
        y_train=y,
        grootcv_n_iter=1,
        grootcv_nfolds=2,
        **{"all_rel_feats__n_jobs": num_jobs},
    )
    X_a_pip, X_all_fil = get_relevant_features(
        X_train=X_all,
        y_train=y,
        grootcv_n_iter=1,
        grootcv_nfolds=2,
        **{"all_rel_feats__n_jobs": num_jobs},
    )

    os.makedirs(f"{tmp_path}/{target_name}", exist_ok=True)
    os.chdir(f"{tmp_path}/{target_name}")

    # Matminer to lobster
    metrics_m2l, summary_m2l = evaluate_feature_set_to_feature(
        X_matminer, X_lob, n_splits=2
    )
    metrics_m2l.to_json("metrics_m2l.json")
    summary_m2l.to_json("summary_m2l.json")

    # Lobster to matminer
    metrics_l2m, summary_l2m = evaluate_feature_set_to_feature(
        X_lob, X_matminer, n_splits=2
    )
    metrics_l2m.to_json("metrics_l2m.json")
    summary_l2m.to_json("summary_l2m.json")

    # Lobster to target
    metrics_l2t_df, summary_l2t = evaluate_features_to_target(X_lob, y, n_splits=2)
    metrics_l2t_df.to_json("metrics_l2t.json")
    summary_l2t.to_json("summary_l2t.json")

    # Matminer to target
    metrics_m2t_df, summary_m2t = evaluate_features_to_target(X_matminer, y, n_splits=2)
    metrics_m2t_df.to_json("metrics_m2t.json")
    summary_m2t.to_json("summary_m2t.json")

    # all_Feats to target
    metrics_a2t_df, summary_a2t = evaluate_features_to_target(X_all_fil, y, n_splits=2)
    metrics_a2t_df.to_json("metrics_a2t.json")
    summary_a2t.to_json("summary_a2t.json")

    r2_combined_feat = np.round(summary_a2t.loc["R2", "mean"], 5)

    # --------------------------------------------------
    # Now runs with mutioutput rf regression and final dependency graph

    # Compute relationships
    results = evaluate_feature_set_relationships(X_lob, X_matminer, y, n_splits=2)
    results.to_json("multioutput_regression_summary.json")

    # Assert data saved
    assert os.path.exists(f"{tmp_path}/{target_name}/metrics_m2l.json")
    assert os.path.exists(f"{tmp_path}/{target_name}/metrics_l2m.json")
    assert os.path.exists(f"{tmp_path}/{target_name}/metrics_l2t.json")
    assert os.path.exists(f"{tmp_path}/{target_name}/metrics_m2t.json")
    assert os.path.exists(f"{tmp_path}/{target_name}/metrics_a2t.json")
    assert os.path.exists(
        f"{tmp_path}/{target_name}/multioutput_regression_summary.json"
    )
