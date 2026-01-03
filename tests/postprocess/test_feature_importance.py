import pandas as pd
from mlproject.postprocess.feature_importances import (
    get_arfs_mean_feature_importances,
    get_modnet_pfi_shap_summary,
    get_rf_pfi_shap_summary,
)


def test_get_arfs_mean_feature_importances(test_data_dir):

    models_parent_dir = test_data_dir / "dummy_models"
    arfs_summary = get_arfs_mean_feature_importances(
        models_parent_dir=models_parent_dir, target_name="max_pfc"
    )

    assert isinstance(arfs_summary, pd.DataFrame)
    assert "mean" in arfs_summary.columns
    assert "std" in arfs_summary.columns


def test_get_modnet_pfi_shap_summary(test_data_dir):

    models_parent_dir = test_data_dir / "dummy_models"

    data_path = test_data_dir / "dummy_data"

    pfi_summary, shap_summary = get_modnet_pfi_shap_summary(
        models_parent_dir=models_parent_dir,
        target_name="last_phdos_peak",
        data_parent_dir=data_path,
        n_repeats=1,
    )

    assert isinstance(pfi_summary, pd.DataFrame)
    assert isinstance(shap_summary, pd.DataFrame)
    assert "mean" in pfi_summary.columns
    assert "mean" in shap_summary.columns
    assert "std" in pfi_summary.columns
    assert "std" in shap_summary.columns


def test_get_rf_pfi_shap_summary(test_data_dir):

    models_parent_dir = test_data_dir / "dummy_models"

    data_path = test_data_dir / "dummy_data"

    pfi_summary, shap_summary = get_rf_pfi_shap_summary(
        models_parent_dir=models_parent_dir,
        target_name="max_pfc",
        data_parent_dir=data_path,
        n_repeats=1,
    )
    assert isinstance(pfi_summary, pd.DataFrame)
    assert isinstance(shap_summary, pd.DataFrame)
    assert "mean" in pfi_summary.columns
    assert "mean" in shap_summary.columns
    assert "std" in pfi_summary.columns
    assert "std" in shap_summary.columns
