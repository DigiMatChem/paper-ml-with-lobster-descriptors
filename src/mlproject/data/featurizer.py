"""
Function to get matminer features from structures via preset from MODNet.
"""

import pandas as pd
import warnings
from monty.serialization import MontyDecoder
from modnet.preprocessing import MODData
from lobsterpy.featurize.batch import BatchSummaryFeaturizer, BatchIcoxxlistFeaturizer

warnings.filterwarnings("ignore")


def get_matminer_feats(structures_df: pd.DataFrame, n_jobs: int = 8) -> pd.DataFrame:
    """
    Get featurized dataframe using pymatgen structure object.

    Uses MODNet featurizer which uses matminer featurizer implementations

    Args:
        structures_df: pandas Dataframe object column named "structure"(pymatgen structures in dict format)
        n_jobs: number of parallel jobs to run for featurization

    Returns:
        A pandas dataframe with structure and composition based features
    """

    structures_df["structure"] = structures_df["structure"].apply(
        MontyDecoder().process_decoded
    )

    mod_data = MODData(
        materials=structures_df["structure"].values,
        structure_ids=list(structures_df.index),
    )

    mod_data.featurize(n_jobs=n_jobs)

    return mod_data.df_featurized


def get_lobster_feats(path_to_lobster_calcs: str, n_jobs: int = 8) -> pd.DataFrame:
    """
    Get featurized dataframe using parent directory path with LOBSTER calcs.

    Uses LobsterPy featurizer implementations.

    Args:
        path_to_lobster_calcs: Path to directory containing LOBSTER calcs output files
        n_jobs: Number of parallel jobs to run for featurization

    Returns:
        A pandas DataFrame with features extracted from LOBSTER output files
    """
    # Summary features (LobsterPy automatic bonding analysis + COHP based + Charge stats)
    summary_featurizer = BatchSummaryFeaturizer(
        path_to_lobster_calcs=path_to_lobster_calcs,
        n_jobs=n_jobs,
        bonds="all",
        charge_type="both",
        feature_type="bonding",
        e_range=[-15.0, 0.0],
        noise_cutoff=1e-4,
    )

    df_summary = summary_featurizer.get_df()

    # common IcoxxlistFeaturizer settings
    icoxx_kwargs = {
        "path_to_lobster_calcs": path_to_lobster_calcs,
        "normalization": "counts",
        "bin_width": 0.1,
        "max_length": 5,
        "n_jobs": n_jobs,
    }

    # BWDF-features types
    bwdf_types = {
        "stats": "get_bwdf_df",
        "binned": "get_bwdf_df",
        "sorted_bwdf": "get_bwdf_df",
        "sorted_dists": "get_bwdf_df",
    }

    dfs = [df_summary]

    for bwdf_type, method_name in bwdf_types.items():
        featurizer = BatchIcoxxlistFeaturizer(
            bwdf_df_type=bwdf_type,
            **icoxx_kwargs,
        )
        dfs.append(getattr(featurizer, method_name)())

    # Asymmetry index
    asi_featurizer = BatchIcoxxlistFeaturizer(
        bwdf_df_type="sorted_dists",
        **icoxx_kwargs,
    )
    dfs.append(asi_featurizer.get_asymmetry_index_df())

    df_combined = pd.concat(dfs, axis=1)

    return df_combined
