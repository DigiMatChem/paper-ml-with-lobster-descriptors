from mlproject.postprocess.t_test import corrected_resampled_ttest


def test_corrected_resampled_ttest():

    # dummy data
    scores_model_1 = [
        0.1858696709403464,
        0.184928844624044,
        0.18349294544720243,
        0.19960707061407953,
        0.1918362640984189,
        0.1877179569833559,
        0.18251424919819753,
        0.1908434589416977,
        0.18629456756105275,
        0.18986189635596692,
    ]
    scores_model_2 = [
        0.1772242019548604,
        0.18635416172522143,
        0.17762492154335296,
        0.19673879837849365,
        0.1819929567135369,
        0.18025149487690942,
        0.17837063323953517,
        0.18272411531685057,
        0.18468349614045518,
        0.18333895830760177,
    ]
    n_train_list = [2945, 2945, 2945, 2946, 2946, 2946, 2946, 2946, 2946, 2946]
    n_test_list = [328, 328, 328, 327, 327, 327, 327, 327, 327, 327]
    alpha = 0.05

    result = corrected_resampled_ttest(
        scores_model_1, scores_model_2, n_train_list, n_test_list, alpha
    )

    assert isinstance(result["t_stat"], float)
    assert isinstance(result["df"], int)
    assert isinstance(result["critical_value"], float)
    assert isinstance(result["p_value"], float)
    assert isinstance(result["r_bar"], float)

    assert result["df"] == 9
    assert result["p_value"] <= 0.01
