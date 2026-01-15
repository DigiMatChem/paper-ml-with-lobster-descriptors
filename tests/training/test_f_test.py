from mlproject.data.preprocessing import get_dataset
from mlproject.training.f_test import paired_ftest_5x2cv_feat_sets


def test_paired_ftest_5x2cv_feat_sets(data_dir, num_jobs):
    # get dataset
    y, X = get_dataset(
        target_name="last_phdos_peak",
        feat_type="matminer_lob",
        data_parent_dir=data_dir,
    )

    f_test_result = paired_ftest_5x2cv_feat_sets(
        model_type="rf",
        X=X.head(500),
        y=y.head(500),
        target_name="last_phdos_peak",
        grootcv_n_iter=2,
        num_jobs=num_jobs,
        random_seed=42,
        **{
            "n_jobs": num_jobs,
        },
    )

    assert "f_stat" in f_test_result
    assert "p_value" in f_test_result
    assert "diffs" in f_test_result
    assert "results_mae" in f_test_result
    assert len(f_test_result["results_mae"]["baseline"]) == 10
    assert len(f_test_result["results_mae"]["extended"]) == 10
    assert f_test_result["p_value"] >= 0.05
