from mlproject.data.preprocessing import get_dataset


def test_get_dataset(data_dir):
    data_parent_dir = data_dir

    target_name = "max_pfc"

    target_df, feature_df = get_dataset(
        target_name=target_name,
        feat_type="matminer",
        data_parent_dir=data_parent_dir,
    )

    assert target_df.shape[0] == feature_df.shape[0]
    assert target_df.columns[0] == target_name
    assert target_df.index.equals(feature_df.index)

    # check that only matminer features are present
    assert not any(col.startswith("asi_") for col in feature_df.columns)
    assert not any(col.startswith("bwdf_") for col in feature_df.columns)
    assert not any(col.startswith("ICOHP_") for col in feature_df.columns)

    # check if units are correctly applied
    assert any("Average_bond_length (A)" in c for c in feature_df.columns)
    assert any("Average_bond_angle (rad)" in c for c in feature_df.columns)
    assert any("GSbandgap (eV)" in c for c in feature_df.columns)
    assert any("LUMO_energy (eV)" in c for c in feature_df.columns)
    assert any("CovalentRadius (pm)" in c for c in feature_df.columns)

    # check for lobster features when feat_type is matminer_lob
    _, feature_df_lob = get_dataset(
        target_name=target_name,
        feat_type="matminer_lob",
        data_parent_dir=data_parent_dir,
    )
    assert any(col.startswith("asi_") for col in feature_df_lob.columns)
    assert any(col.startswith("bwdf_") for col in feature_df_lob.columns)
    assert any(col.startswith("ICOHP_") for col in feature_df_lob.columns)

    # check if units are correctly applied for lobster features
    assert any("w_ICOHP (eV)" in c for c in feature_df_lob.columns)
    assert any("asi_std (eV)" in c for c in feature_df_lob.columns)
    assert any("ICOHP_mean_min (eV)" in c for c in feature_df_lob.columns)
    assert any("bwdf_sum (eV)" in c for c in feature_df_lob.columns)
    assert any("w_ICOHP (eV)" in c for c in feature_df_lob.columns)
    assert any("Madelung_Mull (eV)" in c for c in feature_df_lob.columns)
    assert any("center_COHP (eV)" in c for c in feature_df_lob.columns)
    assert any("dist_at_neg_bwdf2 (A)" in c for c in feature_df_lob.columns)
