"""
Module implementing adapted paired F-test to compare two feature sets using 5x2cv.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from mlproject.training.fold_trainer import train_eval_fold
from mlproject.training.feature_selection import get_relevant_features
from mlproject.utils.misc import split_features


def paired_ftest_5x2cv_feat_sets(
    model_type: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    num_jobs: int,
    target_name: str,
    random_seed: int | None = None,
    grootcv_n_iter: int = 50,
    **est_kwargs,
) -> dict:
    """
    Runs adapted 5x2cv paired F-test to compare the performance of two feature sets on a model.

    Parameters
    ----------
    model_type : str
       Type of model to train ("modnet", "rf")
    X: pd.DataFrame
        Combined feature set
    y: pd.DataFrame
        Target variable
    n_jobs : int
        Number of parallel jobs.
    target_name : str
        Name of the target variable.
    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.
    **est_kwargs
        Additional keyword arguments for the specific model training function.

    Returns
    ----------
    dict
       A dict that consists of the F-statistic, p-value, paired MAE differences and MAEs list for each feature set with following keys

        * f_stat: The F-statistic
        * p_value: Two-tailed p-value
        * diffs: paired MAE differences from each iteration
        * results_mae: dict with MAEs with keys baseline and extended
            * baseline == Matminer feature set
            * extended == Matiner+Lobster feature set

    References
    -----------
    Code has been adapted using mlextend `combined_ftest_5x2cv` implementation as reference

    * https://github.com/rasbt/mlxtend/blob/4a4a84a1891a9809924bd9e039b6b4cb5885c883/mlxtend/evaluate/f_test.py#L125

    """
    rng = np.random.RandomState(random_seed)
    diffs = np.zeros((5, 2))

    results_mae = {"baseline": [], "extended": []}

    # split lobster and matminer feature names
    lob_feats, matminer_feats = split_features(feats=X.columns)

    # get X_all, X_matminer and y
    X_all = X.dropna(axis=1)
    X_matminer_all = X.loc[:, matminer_feats].dropna(axis=1)

    def score_diff(est_1_result, est_2_result):
        score_diff = np.mean(est_1_result["test_errors"]) - np.mean(
            est_2_result["test_errors"]
        )
        return score_diff

    variances = []
    differences = []

    for i in range(5):
        randint = rng.randint(low=0, high=18012019)
        # all
        X_train_all, X_test_all, y_train, y_test = train_test_split(
            X_all, y, test_size=0.5, random_state=randint
        )
        # matminer only
        X_train_matminer_all, X_test_matminer_all, _, _ = train_test_split(
            X_matminer_all, y, test_size=0.5, random_state=randint
        )

        y_train = y_train.iloc[:, 0]
        y_test = y_test.iloc[:, 0]

        pipe, X_train_fil = get_relevant_features(
            X_train=X_train_all,
            y_train=y_train.values.flatten(),
            grootcv_n_iter=grootcv_n_iter,
            grootcv_nfolds=2,
            **{"all_rel_feats__n_jobs": num_jobs},
        )

        pipe, X_train_matminer_fil = get_relevant_features(
            X_train=X_train_matminer_all,
            y_train=y_train.values.flatten(),
            grootcv_n_iter=grootcv_n_iter,
            grootcv_nfolds=2,
            **{"all_rel_feats__n_jobs": num_jobs},
        )

        X_test_fil = X_test_all.loc[:, X_train_fil.columns]
        X_test_matminer_fil = X_test_matminer_all.loc[:, X_train_matminer_fil.columns]

        result_all = train_eval_fold(
            fold_ind=i,
            X_train=X_train_fil,
            y_train=y_train,
            X_test=X_test_fil,
            y_test=y_test,
            model_type=model_type,
            target_name=target_name,
            save_model=False,
            **est_kwargs,
        )

        result_matminer = train_eval_fold(
            fold_ind=i,
            X_train=X_train_matminer_fil,
            y_train=y_train,
            X_test=X_test_matminer_fil,
            y_test=y_test,
            model_type=model_type,
            target_name=target_name,
            save_model=False,
            **est_kwargs,
        )

        # Now swap train test sets

        _, X_train_fil_swap = get_relevant_features(
            X_train=X_test_all,
            y_train=y_test.values.flatten(),
            grootcv_n_iter=grootcv_n_iter,
            grootcv_nfolds=2,
            **{"all_rel_feats__n_jobs": num_jobs},
        )

        _, X_train_matminer_fil_swap = get_relevant_features(
            X_train=X_test_matminer_all,
            y_train=y_test.values.flatten(),
            grootcv_n_iter=grootcv_n_iter,
            grootcv_nfolds=2,
            **{"all_rel_feats__n_jobs": num_jobs},
        )

        X_test_fil_swap = X_train_all.loc[:, X_train_fil_swap.columns]
        X_test_matminer_fil_swap = X_train_matminer_all.loc[
            :, X_train_matminer_fil_swap.columns
        ]

        result_all_swap = train_eval_fold(
            fold_ind=i,
            X_train=X_train_fil_swap,
            y_train=y_test,
            X_test=X_test_fil_swap,
            y_test=y_train,
            model_type=model_type,
            target_name=target_name,
            save_model=False,
            **est_kwargs,
        )

        result_matminer_swap = train_eval_fold(
            fold_ind=i,
            X_train=X_train_matminer_fil_swap,
            y_train=y_test,
            X_test=X_test_matminer_fil_swap,
            y_test=y_train,
            model_type=model_type,
            target_name=target_name,
            save_model=False,
            **est_kwargs,
        )

        diffs[i, 0] = np.mean(result_matminer["test_errors"]) - np.mean(
            result_all["test_errors"]
        )
        diffs[i, 1] = np.mean(result_matminer_swap["test_errors"]) - np.mean(
            result_all_swap["test_errors"]
        )

        results_mae["baseline"].append(np.mean(result_matminer["test_errors"]))
        results_mae["baseline"].append(np.mean(result_matminer_swap["test_errors"]))

        results_mae["extended"].append(np.mean(result_all["test_errors"]))
        results_mae["extended"].append(np.mean(result_all_swap["test_errors"]))

        score_diff_1 = score_diff(result_matminer, result_all)
        score_diff_2 = score_diff(result_matminer_swap, result_all_swap)
        score_mean = (score_diff_1 + score_diff_2) / 2.0
        score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2

        differences.extend([score_diff_1**2, score_diff_2**2])
        variances.append(score_var)

    numerator = sum(differences)
    denominator = 2 * (sum(variances))
    f_stat = numerator / denominator

    p_value = stats.f.sf(f_stat, 10, 5)

    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "diffs": diffs,
        "results_mae": results_mae,
    }
