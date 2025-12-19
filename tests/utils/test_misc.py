from mlproject.utils.misc import split_features


def test_split_features():
    feats = [
        "AtomicOrbitals_LUMO_energy (eV)",
        "CrystalNNFingerprint_mean_wt_CN_1",
        "asi_sum (eV)",
        "AGNIFingerPrint_std_dev_AGNI_dir_y_eta_8.00e-01",
        "site_bwdf_sum_std (eV)",
        "BondOrientationParameter_mean_BOOP_Q_l_2",
        "Icohp_mean_std (eV)",
        "ValenceOrbital_avg_s_valence_electrons",
        "bwdf_sum (eV)",
        "StructuralHeterogeneity_range_neighbor_distance_variation",
        "AtomicOrbitals_gap_AO (eV)",
        "ElementProperty_MagpieData_maximum_MeltingT (K)",
        "CrystalNNFingerprint_std_dev_wt_CN_2",
        "pair_bwdf_sum_mean (eV)",
        "ElementProperty_MagpieData_maximum_NUnfilled",
        "OPSiteFingerprint_mean_sgl_bd_CN_1",
        "AGNIFingerPrint_std_dev_AGNI_dir_z_eta_8.00e-01",
        "OxidationStates_range_oxidation_state",
        "AtomicOrbitals_LUMO_element",
        "GeneralizedRDF_std_dev_Gaussian_center_0.0_width_1.0",
        "EIN_ICOHP",
        "dist_at_neg_bwdf0 (A)",
        "asi_max (eV)",
        "StructuralHeterogeneity_mean_neighbor_distance_variation",
        "bwdf_at_dist0 (eV)",
        "pair_bwdf_min_mean (eV)",
        "BondOrientationParameter_std_dev_BOOP_Q_l_2",
        "StructuralHeterogeneity_maximum_neighbor_distance_variation",
        "ElementProperty_MagpieData_range_NpUnfilled",
        "MaximumPackingEfficiency_max_packing_efficiency",
        "AGNIFingerPrint_std_dev_AGNI_dir_x_eta_8.00e-01",
        "AGNIFingerPrint_mean_AGNI_eta_8.00e-01",
    ]

    expected_lob_feats = [
        "asi_sum (eV)",
        "site_bwdf_sum_std (eV)",
        "Icohp_mean_std (eV)",
        "bwdf_sum (eV)",
        "pair_bwdf_sum_mean (eV)",
        "EIN_ICOHP",
        "dist_at_neg_bwdf0 (A)",
        "asi_max (eV)",
        "bwdf_at_dist0 (eV)",
        "pair_bwdf_min_mean (eV)",
    ]
    expected_matminer_feats = [
        "AtomicOrbitals_LUMO_energy (eV)",
        "CrystalNNFingerprint_mean_wt_CN_1",
        "AGNIFingerPrint_std_dev_AGNI_dir_y_eta_8.00e-01",
        "BondOrientationParameter_mean_BOOP_Q_l_2",
        "ValenceOrbital_avg_s_valence_electrons",
        "StructuralHeterogeneity_range_neighbor_distance_variation",
        "AtomicOrbitals_gap_AO (eV)",
        "ElementProperty_MagpieData_maximum_MeltingT (K)",
        "CrystalNNFingerprint_std_dev_wt_CN_2",
        "ElementProperty_MagpieData_maximum_NUnfilled",
        "OPSiteFingerprint_mean_sgl_bd_CN_1",
        "AGNIFingerPrint_std_dev_AGNI_dir_z_eta_8.00e-01",
        "OxidationStates_range_oxidation_state",
        "AtomicOrbitals_LUMO_element",
        "GeneralizedRDF_std_dev_Gaussian_center_0.0_width_1.0",
        "StructuralHeterogeneity_mean_neighbor_distance_variation",
        "BondOrientationParameter_std_dev_BOOP_Q_l_2",
        "StructuralHeterogeneity_maximum_neighbor_distance_variation",
        "ElementProperty_MagpieData_range_NpUnfilled",
        "MaximumPackingEfficiency_max_packing_efficiency",
        "AGNIFingerPrint_std_dev_AGNI_dir_x_eta_8.00e-01",
        "AGNIFingerPrint_mean_AGNI_eta_8.00e-01",
    ]

    lob_feats, matminer_feats = split_features(feats)

    assert set(lob_feats) == set(expected_lob_feats)
    assert set(matminer_feats) == set(expected_matminer_feats)
