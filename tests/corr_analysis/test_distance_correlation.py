import os
from sklearn.preprocessing import StandardScaler
from mlproject.data.preprocessing import get_dataset
from mlproject.training.feature_selection import get_relevant_features
from mlproject.corr_analysis.distance_correlation import (
    evaluate_distance_correlation_matrix_bootstrap,
)
from mlproject.utils.misc import split_features


def test_evaluate_distance_correlation_matrix_bootstrap(tmp_path, data_dir):

    data_parent_dir = data_dir

    print(f"Using data directory: {data_parent_dir}")

    target_name = "last_phdos_peak"
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
        **{"all_rel_feats__n_jobs": 4},
    )
    X_m_pipe, X_matminer = get_relevant_features(
        X_train=X_matminer_all,
        y_train=y,
        grootcv_n_iter=1,
        grootcv_nfolds=2,
        **{"all_rel_feats__n_jobs": 4},
    )
    X_a_pip, X_all_fil = get_relevant_features(
        X_train=X_all,
        y_train=y,
        grootcv_n_iter=1,
        grootcv_nfolds=2,
        **{"all_rel_feats__n_jobs": 4},
    )

    X1 = X_lob.values
    X2 = X_matminer.values
    X1X2 = X_all_fil.values
    Z = y.values.reshape(-1, 1)

    scaler = StandardScaler()

    X1s = scaler.fit_transform(X1)
    X2s = scaler.fit_transform(X2)
    X1s_X2s = scaler.fit_transform(X1X2)
    ys = scaler.fit_transform(Z)

    # Define feature sets
    sets = {
        "LOBSTER": X1s,
        "MATMINER": X2s,
        "LOBSTER+MATMINER": X1s_X2s,
        f"{target_name}": ys,
    }

    os.makedirs(f"{tmp_path}/{target_name}", exist_ok=True)

    mean_mat, std_mat, mean_pvals = evaluate_distance_correlation_matrix_bootstrap(
        sets=sets, num_resamples=100, num_bootstrap=5
    )
    mean_mat.to_json(f"{tmp_path}/{target_name}/dcor_matrix.json")
    mean_pvals.to_json(f"{tmp_path}/{target_name}/pvals_matrix.json")
    std_mat.to_json(f"{tmp_path}/{target_name}/dcor_std_matrix.json")

    # Assert data saved
    assert os.path.exists(f"{tmp_path}/{target_name}/dcor_matrix.json")
    assert os.path.exists(f"{tmp_path}/{target_name}/pvals_matrix.json")
    assert os.path.exists(f"{tmp_path}/{target_name}/dcor_std_matrix.json")
