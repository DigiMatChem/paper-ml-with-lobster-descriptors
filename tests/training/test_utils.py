from mlproject.training.utils import (
    calculate_cohens_d_av,
    calculate_relative_percentage_improvement,
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
