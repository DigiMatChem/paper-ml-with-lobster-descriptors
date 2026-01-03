import matplotlib.pyplot as plt
from mlproject.postprocess.feature_importances import get_rf_pfi_shap_summary
from mlproject.plotting.importances import plot_feature_importance


def test_plot_feature_importance(test_data_dir):

    models_parent_dir = test_data_dir / "dummy_models"

    data_path = test_data_dir / "dummy_data"

    pfi_summary, shap_summary = get_rf_pfi_shap_summary(
        models_parent_dir=models_parent_dir,
        target_name="max_pfc",
        data_parent_dir=data_path,
        n_repeats=1,
    )

    fig1 = plot_feature_importance(
        pfi_summary,
        target_name="max_pfc",
        model_name="RF",
        importance_type="PFI",
        n_feats=20,
    )
    fig2 = plot_feature_importance(
        shap_summary,
        target_name="max_pfc",
        model_name="RF",
        importance_type="SHAP",
        n_feats=20,
    )

    assert isinstance(fig1, plt.Figure)
    assert isinstance(fig2, plt.Figure)
    assert fig1.axes[0].get_title() == "RF PFI feature importance — max_pfc"
    assert fig2.axes[0].get_title() == "RF SHAP feature importance — max_pfc"
    assert len(fig1.axes[0].patches) > 0
    assert len(fig2.axes[0].patches) > 0
    assert fig1.axes[0].get_xlabel() == "PFI mean feature importance"
    assert fig2.axes[0].get_xlabel() == "SHAP mean feature importance"
