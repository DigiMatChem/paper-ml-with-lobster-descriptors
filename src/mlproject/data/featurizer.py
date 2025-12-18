"""
Function to get matminer features from structures via preset from MODNet.
"""

import pandas as pd
import warnings
from monty.serialization import MontyDecoder
from modnet.preprocessing import MODData

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
