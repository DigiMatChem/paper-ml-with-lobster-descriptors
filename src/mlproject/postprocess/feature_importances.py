import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from modnet.preprocessing import MODData
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from mlproject.data.preprocessing import get_dataset
from tqdm import tqdm

matplotlib.rcParams['pdf.fonttype'] = 42

def process_dataset(data, model):
    x = data.get_featurized_df()[model.optimal_descriptors[:model.n_feat]]
    x = model._scaler.transform(x)
    x = np.nan_to_num(x)
    return x


def aggregate_importances(dfs, value_col, all_features):
    df_all = pd.concat(dfs)
    agg = df_all.groupby("feature")[value_col].agg(["mean", "std"]).reindex(all_features).fillna(0)
    return agg



def get_arfs_mean_feature_importances(
    models_parent_dir: str,
    target_name: str,
):
    """
    Get mean feature importances across GrootCV runs.
    
    Parameters
    ----------
    models_parent_dir : str
        Directory containing rf_*_pipeline.pkl files.
    target_name : str
        Target variable name used in model filenames.
    
    Returns
    -------
    arfs_df : pd.DataFrame
        Mean sorted arfs feature importance.
    """
    imp_dfs = []
    
    # Loop through cross-validation folds
    for i in range(1, 6):
        file_path = f"{models_parent_dir}/rf_{target_name}_matminer_lob/{i}_pipeline.pkl"
        if not os.path.exists(file_path):
            raise ValueError(f" File not found: {file_path}")
            
        with open(file_path, "rb") as f:
            grootcv_obj = pickle.load(f)[-1]

        b_df = grootcv_obj.cv_df.T.copy()
        b_df.columns = b_df.iloc[0]

        # Split into shadow and real features
        shadow_columns = [col for col in b_df.columns if col.startswith("ShadowVar")]
        other_columns = [col for col in b_df.columns if not col.startswith("ShadowVar")]
        b_df = b_df[other_columns + shadow_columns]
        b_df = b_df.drop([b_df.index[0], b_df.index[-1]]).convert_dtypes()

        # Separate real vs shadow subsets
        real_df = b_df.iloc[:, : b_df.shape[1] // 2].copy()

        # Sort by mean importance
        real_df = real_df.reindex(
            real_df.select_dtypes(include=[np.number])
            .mean()
            .sort_values(ascending=True)
            .index,
            axis=1,
        )

        # Keep only selected features
        col_idx = np.argwhere(real_df.columns.isin(grootcv_obj.selected_features_)).ravel()
        imp_dfs.append(real_df.iloc[:, col_idx])
        
    
    # Combine all folds
    combined_df = pd.concat(imp_dfs).fillna(0)
    arfs_summary = pd.DataFrame({"mean": combined_df.mean().values, "std": combined_df.std().values}, index=combined_df.columns)
    
    return arfs_summary


def get_rf_pfi_shap_summary(
    models_parent_dir: str,
    data_parent_dir: str,
    target_name: str,
    n_repeats: int = 5,
    random_state: int = 42,
):
    """
    Load trained RandomForest models (5 folds), compute PFI + SHAP values

    Parameters
    ----------
    models_parent_dir : str
        Directory containing rf_*_pipeline.pkl files for each fold.
    target_name : str
        Target variable name (used in file path pattern).
    n_repeats : int, optional
        Number of PFI repeats (default=5).
    random_state : int, optional
        Random seed for reproducibility.
    n_feats : int, optional
        Number of top features to display.
    save_figs : bool
        Flag to enable saving of pfi/shap plots
    figsize : tuple, optional
        Size of the figure (default=(14, 10)).
    title_fontsize : int, optional
        Font size for plot titles.
    tick_label_fontsize : int, optional
        Font size for axis tick labels.

    Returns
    -------
    pfi_df : pd.DataFrame
        Combined permutation feature importance results.
    shap_df : pd.DataFrame
        Combined mean(|SHAP|) values across folds.
    """

    all_pfi = []
    all_shap = []
    all_features = set()

    cv_outer = KFold(n_splits=5, shuffle=True, random_state=18012019)

    target, feat = get_dataset(feat_type="matminer_lob", target_name=target_name,
        data_parent_dir=data_parent_dir)

    # --- Step 1: Load models and compute importances ---
    for fold_ind, (train_ix, test_ix) in enumerate(cv_outer.split(feat)):

        # Model path
        model_path = f"{models_parent_dir}/rf_{target_name}_matminer_lob/model_{fold_ind+1}.pkl"
        pipeline_path = f"{models_parent_dir}/rf_{target_name}_matminer_lob/{fold_ind+1}_pipeline.pkl"
        if not os.path.exists(model_path):
            print(f"Missing model: {model_path}")
            break

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load pipeline
        with open(pipeline_path, "rb") as f:
            grootcv_obj = pickle.load(f)[-1]

        rel_feats = list(grootcv_obj.selected_features_)

        X_train, X_test = feat.iloc[train_ix], feat.iloc[test_ix]
        y_train, y_test = target.iloc[train_ix, 0], target.iloc[test_ix, 0]

        # Get input data for model fold (X_train, y_train)
        X_train = X_train.loc[:, rel_feats]
        X_test = X_test.loc[:, rel_feats]

        model_features = model.feature_names_in_

        
        # --- Compute PFI ---
        pfi_res = permutation_importance(
            model, X_train, y_train, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        pfi_df = pd.DataFrame({
        "feature": model_features,
        "pfi_mean": pfi_res.importances_mean,
        "pfi_std": pfi_res.importances_std,
        })
        all_pfi.append(pfi_df)
        all_features.update(model_features)

        # --- Compute SHAP ---
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        if isinstance(shap_values, list):  # handle multiclass
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.abs(shap_values)
        mean_abs_shap = shap_values.mean(axis=0)
        shap_df = pd.DataFrame({"feature": model_features, "shap_mean": mean_abs_shap})
        all_shap.append(shap_df)

    # --- Step 2: Combine across folds ---
    all_features = sorted(list(all_features))

    pfi_summary = aggregate_importances(all_pfi, "pfi_mean", all_features)
    shap_summary = aggregate_importances(all_shap, "shap_mean", all_features)

    return pfi_summary, shap_summary


def get_modnet_pfi_shap_summary(
    models_parent_dir: str,
    data_parent_dir: str,
    target_name: str,
    n_repeats: int = 5,
    random_state: int = 42,
):
    """
    Load trained MODNet models (5 folds), compute PFI + SHAP values
    Parameters
    ----------
    models_parent_dir : str
        Directory containing rf_*_pipeline.pkl files for each fold.
    target_name : str
        Target variable name (used in file path pattern).
    n_repeats : int, optional
        Number of PFI repeats (default=5).
    random_state : int, optional
        Random seed for reproducibility.
    n_feats : int, optional
        Number of top features to display.
    save_figs : bool
        Flag to enable saving of pfi/shap plots
    figsize : tuple, optional
        Size of the figure (default=(14, 10)).
    title_fontsize : int, optional
        Font size for plot titles.
    tick_label_fontsize : int, optional
        Font size for axis tick labels.

    Returns
    -------
    pfi_df : pd.DataFrame
        Combined permutation feature importance results.
    shap_df : pd.DataFrame
        Combined mean(|SHAP|) values across folds.
    """

    all_pfi = []
    all_shap = []
    all_features = set()

    cv_outer = KFold(n_splits=5, shuffle=True, random_state=18012019)

    target, feat = get_dataset(feat_type="matminer_lob", target_name=target_name,
        data_parent_dir=data_parent_dir)

    # --- Step 1: Load models and compute importances ---
    for fold_ind, (train_ix, test_ix) in enumerate(cv_outer.split(feat)):


        # Model path
        model_path = f"{models_parent_dir}/modnet_{target_name}_matminer_lob/model_{fold_ind+1}"
        pipeline_path = f"{models_parent_dir}/modnet_{target_name}_matminer_lob/{fold_ind+1}_pipeline.pkl"
        if not os.path.exists(model_path):
            print(f"Missing model: {model_path}")
            break

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            model._restore_model() # get model instance from dict for inner models of ensemble

        # Load pipeline
        with open(pipeline_path, "rb") as f:
            grootcv_obj = pickle.load(f)[-1]

        rel_feats = list(grootcv_obj.selected_features_)

        X_train, X_test = feat.iloc[train_ix], feat.iloc[test_ix]
        y_train, y_test = target.iloc[train_ix].values.flatten(), target.iloc[test_ix].values.flatten()

        # Get input data for model fold (X_train, y_train)
        X_train = X_train.loc[:, rel_feats]
        X_test = X_test.loc[:, rel_feats]

        moddata_train = MODData(
            df_featurized=X_train,
            targets=y_train.reshape(-1, 1),
            target_names=[target_name],
            structure_ids=list(X_train.index),
        )

        for inner_model in tqdm(model.models):

            X_train_scaled = process_dataset(moddata_train, inner_model)

            model_features = inner_model.optimal_descriptors[:inner_model.n_feat]

            # --- Compute PFI --- for each inner model of ensemble
            pfi_res = permutation_importance(inner_model.model, X_train_scaled, moddata_train.targets,
                           n_repeats=n_repeats,
                           scoring="neg_mean_absolute_error",
                           n_jobs=16,
                           random_state=42)

            pfi_df = pd.DataFrame({
            "feature": model_features,
            "pfi_mean": pfi_res.importances_mean,
            "pfi_std": pfi_res.importances_std,
            })

            all_pfi.append(pfi_df)
            all_features.update(model_features)

            # --- Compute SHAP --- for each inner model of ensemble
            explainer = shap.DeepExplainer(model=inner_model.model, data=X_train_scaled)
            shap_values_obj = explainer(X_train_scaled)
            shap_values_obj.values = shap_values_obj.values[:,:, 0]
            if isinstance(shap_values_obj.values, list):  # handle multiclass
                shap_values = np.mean(np.abs(shap_values_obj.values), axis=0)
            else:
                shap_values = np.abs(shap_values_obj.values)
            mean_abs_shap = shap_values.mean(axis=0)
            shap_df = pd.DataFrame({"feature": model_features, "shap_mean": mean_abs_shap.flatten()})
            all_shap.append(shap_df)


    # --- Step 2: Combine across folds ---
    all_features = sorted(list(all_features))


    pfi_summary = aggregate_importances(all_pfi, "pfi_mean", all_features)
    shap_summary = aggregate_importances(all_shap, "shap_mean", all_features)

    return pfi_summary, shap_summary