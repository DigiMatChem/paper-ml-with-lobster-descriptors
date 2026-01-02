import pandas as pd
from mlproject.data.featurizer import (
    get_matminer_feats,
    get_matminer_site_feats,
    get_lobster_feats,
    get_lobster_site_feat,
)


def test_get_matminer_feats(data_dir, num_jobs):
    print(num_jobs)
    data_parent_dir = data_dir

    structures_file = data_parent_dir / "structures" / "structures.json.gz"

    structures_df = pd.read_json(structures_file).head(5)

    matminer_feats = get_matminer_feats(structures_df=structures_df, n_jobs=num_jobs)

    assert isinstance(matminer_feats, pd.DataFrame)
    assert matminer_feats.shape[0] == structures_df.shape[0]


def test_get_matminer_site_feats(data_dir, num_jobs):
    data_parent_dir = data_dir

    structures_file = data_parent_dir / "structures" / "structures_w_site_index_msd.json.gz"

    structures_df = pd.read_json(structures_file).head(5)
    
    matminer_site_feats = get_matminer_site_feats(
        structures_df=structures_df, n_jobs=num_jobs
    )

    assert isinstance(matminer_site_feats, pd.DataFrame)
    assert matminer_site_feats.shape[0] == structures_df.shape[0]


def test_get_lobster_feats(test_data_dir, num_jobs):

    data_parent_dir = test_data_dir

    path_to_lobster_calcs = data_parent_dir / "lobster_calcs"

    lobster_feats = get_lobster_feats(
        path_to_lobster_calcs=path_to_lobster_calcs, n_jobs=num_jobs
    )

    assert isinstance(lobster_feats, pd.DataFrame)
    assert sorted(lobster_feats.index) == sorted(["mp-463", "mp-1000", "mp-2176"])
    assert lobster_feats.shape[0] == 3  # number of lobster calc folders
    assert lobster_feats.shape[1] == 135  # number of features extracted


def test_get_lobster_site_feat(test_data_dir):
    data_parent_dir = test_data_dir

    path_to_lobster_calc = data_parent_dir / "lobster_calcs" / "mp-2176"

    lobster_site_feats = get_lobster_site_feat(
        path_to_lobster_calc=path_to_lobster_calc,
        site_index=1,
    )

    expected_cols = [
        "site_bwdf_mean",
        "site_bwdf_std",
        "site_bwdf_min",
        "site_bwdf_max",
        "site_bwdf_sum",
        "site_bwdf_skew",
        "site_bwdf_kurtosis",
        "site_asi",
        "charge_loew",
        "charge_mull",
    ]

    assert isinstance(lobster_site_feats, pd.DataFrame)
    assert lobster_site_feats.index == "mp-2176_1"
    assert all(col in lobster_site_feats.columns for col in expected_cols)
