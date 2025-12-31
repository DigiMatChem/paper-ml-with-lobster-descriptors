import os
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from mlproject.training.feature_selection import GAFeatureSelector


def test_ga_feature_selector(tmp_path, num_jobs):

    # Change to temporary directory to save plot
    os.chdir(tmp_path)

    X, y, coef = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=5,
        noise=0.1,
        coef=True,
        random_state=42,
    )

    feature_names = [f"x{i}" for i in range(X.shape[1])]

    informative_features = [name for name, c in zip(feature_names, coef) if c != 0]

    # X as DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)

    # y as single-column DataFrame
    y_df = pd.DataFrame(y, columns=["target"])

    selector = GAFeatureSelector(
        X=X_df,
        y=y_df,
        model=RandomForestRegressor(n_estimators=20, random_state=42),
        scoring="neg_mean_absolute_error",
        X_test=None,
        y_test=None,
        n_jobs=num_jobs,
        return_train_score=False,
        num_features=5,
        population_size=20,
        generations=50,
    )

    # Run differential evolution strategy
    selected_features = selector.run(strategy="de")

    # Verify that selected features are among the informative features using threshold
    threshold = 0.8 * len(informative_features)
    common_features = set(selected_features).intersection(set(informative_features))
    assert (
        len(common_features) >= threshold
    ), f"Expected at least {threshold} informative features, but got {len(common_features)}"

    # check if convergence plot is created
    assert os.path.isfile("GA_result_De.png")
    # Clean up plot file
    os.remove("GA_result_De.png")

    # Run standard GA strategy
    selected_features = selector.run(strategy="standard")
    # check if convergence plot is created
    assert os.path.isfile("GA_result_Standard.png")
    # Clean up plot file
    os.remove("GA_result_Standard.png")
