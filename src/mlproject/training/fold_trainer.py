"""
Training/evaluation functions for RF/MODNet/GA-SISSO models.
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sissopp import Inputs
from sissopp import SISSORegressor
from sissopp.py_interface import get_fs_solver
from sissopp.postprocess.load_models import load_model
from modnet.hyper_opt.fit_genetic import FitGenetic
from modnet.preprocessing import MODData
from modnet.models import EnsembleMODNetModel
from mlproject.training.feature_selection import GAFeatureSelector
from mlproject.postprocess.utils import mean_absolute_percentage_error

warnings.filterwarnings("ignore")


def train_modnet(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    target_name: str,
    n_jobs: int,
    save_model: bool,
    fold_ind,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, EnsembleMODNetModel, np.ndarray, np.ndarray]:
    """
    Training/evaluation for MODNet model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.DataFrame
        Training target.
    y_test : pd.DataFrame
        Testing target.
    target_name : str
        Name of the target variable.
    n_jobs : int
        Number of parallel jobs.
    save_model : bool
        Whether to save the trained model.
    fold_ind : int
        Fold index for cross-validation.
    **kwargs
        Additional keyword arguments for MODNet's genetic algorithm hyperparameter optimization.
    """

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    moddata_train = MODData(
        df_featurized=X_train,
        targets=y_train.reshape(-1, 1),
        target_names=[target_name],
        structure_ids=list(X_train.index),
    )
    moddata_test = MODData(
        df_featurized=X_test,
        targets=y_test.reshape(-1, 1),
        target_names=[target_name],
        structure_ids=list(X_test.index),
    )

    moddata_train.feature_selection(n=-1, n_jobs=n_jobs, random_state=42)

    ga_settings = {
        "size_pop": 50,
        "num_generations": 20,
        "early_stopping": 4,
        "refit": False,
        "nested": True,
        **kwargs,
    }

    ga = FitGenetic(moddata_train, targets=[[[target_name]]])
    model = ga.run(n_jobs=n_jobs, **ga_settings)

    if save_model:
        model.save(f"model_{fold_ind+1}")

    y_hat_train = model.predict(moddata_train).values.flatten()
    y_hat_test = model.predict(moddata_test).values.flatten()

    return y_hat_train, y_hat_test, model, y_train, y_test


def train_sisso_ga(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    n_jobs: int,
    save_model: bool,
    fold_ind: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, SISSORegressor, np.ndarray, np.ndarray]:
    """
    Training/evaluation for SISSO model with GA-based feature selection.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.DataFrame
        Training target.
    y_test : pd.DataFrame
        Testing target.
    n_jobs : int
        Number of parallel jobs.
    save_model : bool
        Whether to save the trained model.
    fold_ind : int
        Fold index for cross-validation.
    **kwargs
        Additional keyword arguments for GA feature selection and SISSO inputs."""

    default_ga_kwargs = {
        "num_features": 15,
        "generations": 100,
        "population_size": 50,
        "early_stop_patience": 5,
        "cv": 5,
        "mpi_tasks": kwargs.get("mpi_tasks", 1),
        "sissopp_binary_path": kwargs.get("sissopp_binary_path"),
        "sissopp_inputs": kwargs.get("sissopp_inputs"),
        **kwargs,
    }

    current_work_dir = os.getcwd()

    os.makedirs(f"fold_{fold_ind+1}", exist_ok=True)
    os.chdir(f"fold_{fold_ind+1}")

    selector = GAFeatureSelector(
        X=X_train,
        y=y_train,
        model=None,
        scoring="neg_mean_absolute_error",
        X_test=None,
        y_test=None,
        n_jobs=n_jobs,
        return_train_score=False,
        **default_ga_kwargs,
    )

    selected_features = selector.run(strategy="de")

    with open("feature_usage_counts.json", "w") as f:
        json.dump(selector.feature_usage_counts, f)

    X_train_fil = X_train.loc[:, selected_features]
    X_test_fil = X_test.loc[:, selected_features]

    pd.concat([X_train_fil, y_train], axis=1).to_csv("data.csv")

    with open("sisso.json", "w") as f:
        json.dump(kwargs.get("sissopp_inputs"), f, indent=4)

    inputs = Inputs("sisso.json")
    _, model = get_fs_solver(inputs, allow_overwrite=False)

    model.fit()

    model_file_name = Path("models") / f"train_dim_{model.n_dim}_model_0.dat"

    trained_model = load_model(model_file_name.as_posix())

    y_hat_train = trained_model.eval_many(X_train_fil)
    y_hat_test = trained_model.eval_many(X_test_fil)

    os.chdir(current_work_dir)

    return y_hat_train, y_hat_test, model, y_train, y_test


def train_rf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    n_jobs: int,
    save_model: bool,
    fold_ind: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, RandomForestRegressor, np.ndarray, np.ndarray]:
    """
    Training/evaluation for RandomForest model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.DataFrame
        Training target.
    y_test : pd.DataFrame
        Testing target.
    n_jobs : int
        Number of parallel jobs.
    save_model : bool
        Whether to save the trained model.
    fold_ind : int
        Fold index for cross-validation.
    **kwargs
        Additional keyword arguments for RandomForestRegressor.

    Returns
    -------
    tuple
        y_hat_train, y_hat_test, model, y_train, y_test
    """

    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    estimator_kwargs = {
        "n_jobs": n_jobs,
        "n_estimators": 500,
        **kwargs,
    }
    model = RandomForestRegressor(**estimator_kwargs)
    model.fit(X_train, y_train)

    if save_model:
        with open(f"model_{fold_ind+1}.pkl", "wb") as f:
            pickle.dump(model, f)

    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    return y_hat_train, y_hat_test, model, y_train, y_test


def train_eval_fold(
    model_type: str,
    fold_ind: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    target_name: str = None,
    n_jobs: int = 1,
    save_model: bool = True,
    **kwargs,
):
    """
    Training/evaluation for MODNet, GA-SISSO, and RandomForest.

    Parameters
    ----------
    model_type : str
        Type of model to train ("modnet", "ga_sisso", "rf").
    fold_ind : int
        Fold index for cross-validation.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.DataFrame
        Training target.
    y_test : pd.DataFrame
        Testing target.
    target_name : str, optional
        Name of the target variable (required for MODNet).
    n_jobs : int, optional
        Number of parallel jobs (default is 1).
    save_model : bool, optional
        Whether to save the trained model (default is True).
    **kwargs
        Additional keyword arguments for the specific model training function.

    Returns
    -------
    dict
        Dictionary containing training and testing regression metrics.
    """

    if model_type.lower() == "modnet":
        y_hat_train, y_hat_test, _, y_train_true, y_test_true = train_modnet(
            X_train,
            X_test,
            y_train,
            y_test,
            target_name,
            n_jobs,
            save_model,
            fold_ind,
            **kwargs,
        )
    elif model_type.lower() == "ga_sisso":
        y_hat_train, y_hat_test, _, y_train_true, y_test_true = train_sisso_ga(
            X_train, X_test, y_train, y_test, n_jobs, save_model, fold_ind, **kwargs
        )
    elif model_type.lower() == "rf":
        y_hat_train, y_hat_test, _, y_train_true, y_test_true = train_rf(
            X_train, X_test, y_train, y_test, n_jobs, save_model, fold_ind, **kwargs
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # --- Evaluation ---
    train_rmse = root_mean_squared_error(y_train_true, y_hat_train)
    test_rmse = root_mean_squared_error(y_test_true, y_hat_test)
    train_r2 = r2_score(y_train_true, y_hat_train)
    test_r2 = r2_score(y_test_true, y_hat_test)
    train_errors = (
        abs(y_train_true - y_hat_train).values
        if isinstance(abs(y_train_true - y_hat_train), pd.DataFrame)
        else abs(y_train_true - y_hat_train)
    )
    test_errors = (
        abs(y_test_true - y_hat_test).values
        if isinstance(abs(y_test_true - y_hat_test), pd.DataFrame)
        else abs(y_test_true - y_hat_test)
    )
    train_mape = mean_absolute_percentage_error(y_true=y_train_true, y_pred=y_hat_train)
    test_mape = mean_absolute_percentage_error(y_true=y_test_true, y_pred=y_hat_test)

    fold_metrics = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mape": train_mape,
        "test_mape": test_mape,
    }

    return fold_metrics
