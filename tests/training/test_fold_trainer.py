import os
import pickle
import numpy as np
from mlproject.data.preprocessing import get_dataset
from mlproject.training.fold_trainer import train_eval_fold
from mlproject.training.feature_selection import get_relevant_features
from sklearn.model_selection import KFold
import mlflow


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_train_eval_fold_rf(tmp_path, data_dir, num_jobs):
    # Load dataset
    data_parent_dir = data_dir

    abs_path = str(tmp_path.resolve())

    target_name = "max_pfc"

    target, feat = get_dataset(
        target_name=target_name,
        feat_type="matminer_lob",
        data_parent_dir=data_parent_dir,
    )

    feat.dropna(axis=1, inplace=True)

    # Get smaller dataset for faster testing
    feat = feat.head(100)
    target = target.head(100)

    cv_outer = KFold(n_splits=2, shuffle=True, random_state=18012019)

    mlflow.set_tracking_uri(f"file://{abs_path}")
    experiment_name = f"rf_experiment_{target_name}"
    mlflow.set_experiment(experiment_name)

    os.chdir(abs_path)
    os.makedirs(f"rf_{target_name}_matminer_lob", exist_ok=True)
    os.chdir(f"rf_{target_name}_matminer_lob")

    with mlflow.start_run(
        run_name=f"{target_name}_matminer_lob",
        experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id,
    ):

        all_results = {
            "train_mae": [],
            "test_mae": [],
            "train_rmse": [],
            "test_rmse": [],
            "train_r2": [],
            "test_r2": [],
            "train_mape": [],
            "test_mape": [],
        }
        for fold_ind, (train_ix, test_ix) in enumerate(cv_outer.split(feat)):
            X_train, X_test = feat.iloc[train_ix], feat.iloc[test_ix]
            y_train, y_test = target.iloc[train_ix, 0], target.iloc[test_ix, 0]

            pipe, X_train_fil = get_relevant_features(
                X_train=X_train,
                y_train=y_train.values.flatten(),
                grootcv_n_iter=1,
                **{"all_rel_feats__n_jobs": num_jobs},
            )

            with open(f"{fold_ind+1}_pipeline.pkl", "wb") as f:
                pickle.dump(pipe, f)

            X_test_fil = X_test.loc[:, X_train_fil.columns]

            result = train_eval_fold(
                fold_ind=fold_ind,
                X_train=X_train_fil,
                y_train=y_train,
                X_test=X_test_fil,
                y_test=y_test,
                model_type="rf",
                **{
                    "n_jobs": num_jobs,
                    "n_estimators": 20,
                },
            )

            for metric, value in result.items():
                if isinstance(value, float):
                    all_results[metric].append(value)
                elif "train" in metric:
                    all_results["train_mae"].append(value.mean())
                else:
                    all_results["test_mae"].append(value.mean())

            with open(f"{fold_ind+1}_results.pkl", "wb") as f:
                pickle.dump(result, f)

            assert os.path.exists(f"{fold_ind+1}_results.pkl")
            assert os.path.exists(f"{fold_ind+1}_pipeline.pkl")
            assert os.path.exists(f"model_{fold_ind+1}.pkl")

        for metric, value in all_results.items():
            mlflow.log_metric(f"{metric}_mean", np.array(value).mean())
            mlflow.log_metric(f"{metric}_min", np.array(value).min())
            mlflow.log_metric(f"{metric}_max", np.array(value).max())
            mlflow.log_metric(f"{metric}_std", np.array(value).std())

    os.chdir(MODULE_DIR)


def test_train_eval_fold_modnet(tmp_path, data_dir, num_jobs):

    data_parent_dir = data_dir

    abs_path = str(tmp_path.resolve())

    target_name = "last_phdos_peak"

    target, feat = get_dataset(
        target_name=target_name,
        feat_type="matminer_lob",
        data_parent_dir=data_parent_dir,
    )

    feat.dropna(axis=1, inplace=True)

    # Get smaller dataset for faster testing
    feat = feat.head(100)
    target = target.head(100)

    cv_outer = KFold(n_splits=2, shuffle=True, random_state=18012019)

    mlflow.set_tracking_uri(f"file://{abs_path}")
    experiment_name = f"modnet_experiment_{target_name}"
    mlflow.set_experiment(experiment_name)

    os.chdir(abs_path)
    os.makedirs(f"modnet_{target_name}_matminer_lob", exist_ok=True)
    os.chdir(f"modnet_{target_name}_matminer_lob")

    with mlflow.start_run(
        run_name=f"{target_name}_matminer_lob",
        experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id,
    ):

        all_results = {
            "train_mae": [],
            "test_mae": [],
            "train_rmse": [],
            "test_rmse": [],
            "train_r2": [],
            "test_r2": [],
            "train_mape": [],
            "test_mape": [],
        }
        for fold_ind, (train_ix, test_ix) in enumerate(cv_outer.split(feat)):
            X_train, X_test = feat.iloc[train_ix], feat.iloc[test_ix]
            y_train, y_test = target.iloc[train_ix, 0], target.iloc[test_ix, 0]

            pipe, X_train_fil = get_relevant_features(
                X_train=X_train,
                y_train=y_train.values.flatten(),
                grootcv_n_iter=1,
                **{"all_rel_feats__n_jobs": num_jobs},
            )

            with open(f"{fold_ind+1}_pipeline.pkl", "wb") as f:
                pickle.dump(pipe, f)

            X_test_fil = X_test.loc[:, X_train_fil.columns]

            result = train_eval_fold(
                fold_ind=fold_ind,
                X_train=X_train_fil,
                y_train=y_train,
                X_test=X_test_fil,
                y_test=y_test,
                model_type="modnet",
                target_name=target_name,
                **{
                    "n_jobs": num_jobs,
                    "size_pop": 2,
                    "num_generations": 2,
                    "refit": False,
                },
            )

            for metric, value in result.items():
                if isinstance(value, float):
                    all_results[metric].append(value)
                elif "train" in metric:
                    all_results["train_mae"].append(value.mean())
                else:
                    all_results["test_mae"].append(value.mean())

            with open(f"{fold_ind+1}_results.pkl", "wb") as f:
                pickle.dump(result, f)

            assert os.path.exists(f"{fold_ind+1}_results.pkl")
            assert os.path.exists(f"{fold_ind+1}_pipeline.pkl")
            assert os.path.exists(f"model_{fold_ind+1}")

        for metric, value in all_results.items():
            mlflow.log_metric(f"{metric}_mean", np.array(value).mean())
            mlflow.log_metric(f"{metric}_min", np.array(value).min())
            mlflow.log_metric(f"{metric}_max", np.array(value).max())
            mlflow.log_metric(f"{metric}_std", np.array(value).std())

    os.chdir(MODULE_DIR)


def test_train_eval_fold_sisso(tmp_path, data_dir, num_jobs):

    data_parent_dir = data_dir

    abs_path = str(tmp_path.resolve())

    target_name = "last_phdos_peak"

    target, feat = get_dataset(
        target_name=target_name,
        feat_type="matminer_lob",
        data_parent_dir=data_parent_dir,
    )

    feat.dropna(axis=1, inplace=True)


    cv_outer = KFold(n_splits=2, shuffle=True, random_state=18012019)

    sissopp_binary_path = "/home/mluser/sissopp/build/bin/sisso++"

    sissopp_inputs = {
        "data_file": "data.csv",
        "property_key": target_name,
        "desc_dim": 2,
        "n_sis_select": 10,
        "max_leaves": 4,
        "max_rung": 1,
        "calc_type": "regression",
        "min_abs_feat_val": 1e-5,
        "max_abs_feat_val": 1e8,
        "n_residual": 1,
        "n_models_store": 1,
        "n_rung_generate": 1,
        "n_rung_store": 0,
        "leave_out_frac": 0.0,
        "leave_out_inds": [],
        "verbose": False,
        "opset": [
            "add",
            "sub",
            "abs_diff",
            "mult",
            "div",
            "inv",
            "abs",
            "exp",
            "log",
            "sq",
            "cb",
            "sqrt",
            "cbrt",
            "neg_exp",
        ],
    }

    mlflow.set_tracking_uri(f"file://{abs_path}")
    experiment_name = f"sisso_experiment_{target_name}"
    mlflow.set_experiment(experiment_name)

    os.chdir(abs_path)
    os.makedirs(f"sisso_{target_name}_matminer_lob", exist_ok=True)
    os.chdir(f"sisso_{target_name}_matminer_lob")

    with mlflow.start_run(
        run_name=f"{target_name}_matminer_lob",
        experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id,
    ):

        all_results = {
            "train_mae": [],
            "test_mae": [],
            "train_rmse": [],
            "test_rmse": [],
            "train_r2": [],
            "test_r2": [],
            "train_mape": [],
            "test_mape": [],
        }
        for fold_ind, (train_ix, test_ix) in enumerate(cv_outer.split(feat)):
            X_train, X_test = feat.iloc[train_ix], feat.iloc[test_ix]
            y_train, y_test = target.iloc[train_ix, 0], target.iloc[test_ix, 0]

            pipe, X_train_fil = get_relevant_features(
                X_train=X_train,
                y_train=y_train.values.flatten(),
                grootcv_n_iter=1,
                **{"all_rel_feats__n_jobs": num_jobs},
            )

            with open(f"{fold_ind+1}_pipeline.pkl", "wb") as f:
                pickle.dump(pipe, f)

            X_test_fil = X_test.loc[:, X_train_fil.columns]

            result = train_eval_fold(
                fold_ind=fold_ind,
                X_train=X_train_fil,
                y_train=y_train,
                X_test=X_test_fil,
                y_test=y_test,
                model_type="ga_sisso",
                target_name=target_name,
                **{
                    "num_features": 10,
                    "sissopp_binary_path": sissopp_binary_path,
                    "sissopp_inputs": sissopp_inputs,
                    "mpi_tasks": num_jobs,
                    "population_size": 3,
                    "cv": 2,
                    "generations": 2,
                },
            )

            for metric, value in result.items():
                if isinstance(value, float):
                    all_results[metric].append(value)
                elif "train" in metric:
                    all_results["train_mae"].append(value.mean())
                else:
                    all_results["test_mae"].append(value.mean())

            with open(f"{fold_ind+1}_results.pkl", "wb") as f:
                pickle.dump(result, f)

            assert os.path.exists(f"{fold_ind+1}_results.pkl")
            assert os.path.exists(f"{fold_ind+1}_pipeline.pkl")
            assert os.path.exists(f"fold_{fold_ind+1}/models/train_dim_1_model_0.dat")
            assert os.path.exists(f"fold_{fold_ind+1}/models/train_dim_2_model_0.dat")
            assert os.path.exists(f"fold_{fold_ind+1}/GA_result_De.png")

        for metric, value in all_results.items():
            mlflow.log_metric(f"{metric}_mean", np.array(value).mean())
            mlflow.log_metric(f"{metric}_min", np.array(value).min())
            mlflow.log_metric(f"{metric}_max", np.array(value).max())
            mlflow.log_metric(f"{metric}_std", np.array(value).std())

    os.chdir(MODULE_DIR)
