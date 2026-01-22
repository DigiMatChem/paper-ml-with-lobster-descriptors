from mlproject.postprocess.utils import (
    calculate_cohens_d_av,
    calculate_relative_percentage_improvement,
    caclulate_percent_folds_improved,
    get_ttest_summary_df,
)


def test_calculate_cohens_d_av():

    baseline_scores = [
        45.69406373677308,
        53.08880265167338,
        53.2732383408494,
        42.874530859766644,
        56.23445517542763,
        40.70766535017603,
        51.934091584118065,
        46.88566517662434,
        52.764641252208,
        47.21200111406419,
    ]

    new_scores = [
        44.31534899337208,
        47.673849784136245,
        51.57863104119994,
        41.09489112006519,
        51.57121873257883,
        40.196283462137664,
        48.04626169966864,
        47.22656051148069,
        50.88908187669428,
        45.76111532596996,
    ]

    d_av = calculate_cohens_d_av(baseline_scores, new_scores)

    # Expected value
    expected_d_av = 0.487

    assert abs(d_av - expected_d_av) < 1e-3


def test_calculate_relative_percentage_improvement():
    baseline_scores = [
        45.69406373677308,
        53.08880265167338,
        53.2732383408494,
        42.874530859766644,
        56.23445517542763,
        40.70766535017603,
        51.934091584118065,
        46.88566517662434,
        52.764641252208,
        47.21200111406419,
    ]

    new_scores = [
        44.31534899337208,
        47.673849784136245,
        51.57863104119994,
        41.09489112006519,
        51.57121873257883,
        40.196283462137664,
        48.04626169966864,
        47.22656051148069,
        50.88908187669428,
        45.76111532596996,
    ]

    rpi = calculate_relative_percentage_improvement(baseline_scores, new_scores)

    # Manually calculated value
    expected_rpi = 4.548

    assert abs(rpi - expected_rpi) < 1e-3


def test_caclulate_percent_folds_improved():
    baseline_scores = [
        45.69406373677308,
        53.08880265167338,
        53.2732383408494,
        42.874530859766644,
        56.23445517542763,
        40.70766535017603,
        51.934091584118065,
        46.88566517662434,
        52.764641252208,
        47.21200111406419,
    ]

    new_scores = [
        44.31534899337208,
        47.673849784136245,
        51.57863104119994,
        41.09489112006519,
        51.57121873257883,
        40.196283462137664,
        48.04626169966864,
        47.22656051148069,
        50.88908187669428,
        45.76111532596996,
    ]

    percent_improved = caclulate_percent_folds_improved(baseline_scores, new_scores)

    assert percent_improved == 90.0

def test_get_ttest_summary_df(test_data_dir):
    
    models_dir = test_data_dir / "dummy_models" / "t_test"

    summary_df = get_ttest_summary_df(
        models_dir=models_dir,
        target_name="max_pfc",
        num_folds=10,
        model_type="rf",
        feature_set_types=["matminer", "matminer_lob"],
        alternative="greater",
    )
    
    assert "t_stat" in summary_df.columns
    assert "df" in summary_df.columns
    assert "critical_value" in summary_df.columns
    assert "p_value" in summary_df.columns
    assert "significance_stars" in summary_df.columns
    assert "r_bar" in summary_df.columns
    assert "d_av" in summary_df.columns
    assert "rel_improvement" in summary_df.columns
    assert "percent_folds_improved" in summary_df.columns

    assert summary_df.loc["max_pfc", "p_value"] < 0.06
    assert summary_df.loc["max_pfc", "significance_stars"] == ""
    assert abs(summary_df.loc["max_pfc", "d_av"] - 0.809559) < 1e-3
    assert abs(summary_df.loc["max_pfc", "rel_improvement"] - 13.346632) < 1e-3
    assert abs(summary_df.loc["max_pfc", "percent_folds_improved"] - 80.0) < 1e-3
    assert abs(summary_df.loc["max_pfc", "t_stat"] - 1.826292) < 1e-3