"""
Function to get matminer features from structures via preset from MODNet.
"""

import pandas as pd
import numpy as np
import warnings
from monty.serialization import MontyDecoder
from modnet.preprocessing import MODData
from matminer.featurizers.site import (
    AGNIFingerprints,
    AverageBondAngle,
    AverageBondLength,
    BondOrientationalParameter,
    ChemEnvSiteFingerprint,
    CoordinationNumber,
    CrystalNNFingerprint,
    GaussianSymmFunc,
    GeneralizedRadialDistributionFunction,
    LocalPropertyDifference,
    OPSiteFingerprint,
    VoronoiFingerprint,
    SiteElementalProperty,
)
from matminer.featurizers.base import MultipleFeaturizer
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.lobster import Charge
from lobsterpy.featurize.batch import BatchSummaryFeaturizer, BatchIcoxxlistFeaturizer
from lobsterpy.featurize.core import FeaturizeIcoxxlist
from lobsterpy.featurize.utils import get_file_paths
from scipy.stats import kurtosis, skew

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


def get_matminer_site_feats(
    structures_df: pd.DataFrame, site_featurizers: list | None = None, n_jobs: int = 8
) -> pd.DataFrame:
    """
    Get site featurized dataframe using pymatgen structure object .

    Uses Matminers MultipleFeaturizer

    Args:
        structures_df: pandas Dataframe object column named "structure"(pymatgen structures in dict format) and "site_index"
        site_featurizers: list of matminer site based featurizes to apply on input structures_df
        n_jobs: number of parallel jobs to run for featurization

    Returns:
        A pandas dataframe with structure and composition based features for sites
    """

    structures_df["structure"] = structures_df["structure"].apply(
        MontyDecoder().process_decoded
    )

    if site_featurizers is None:
        site_featurizers = [
            AGNIFingerprints(),
            AverageBondAngle(VoronoiNN()),
            AverageBondLength(VoronoiNN()),
            BondOrientationalParameter(),
            ChemEnvSiteFingerprint.from_preset("simple"),
            CoordinationNumber(),
            CrystalNNFingerprint.from_preset("ops"),
            GaussianSymmFunc(),
            GeneralizedRadialDistributionFunction.from_preset("gaussian"),
            LocalPropertyDifference(),
            OPSiteFingerprint(),
            VoronoiFingerprint(),
            SiteElementalProperty.from_preset("seko-prb-2017"),
        ]

    multi_feat = MultipleFeaturizer(featurizers=site_featurizers)

    matminer_feat_df = multi_feat.featurize_dataframe(
        structures_df, col_id=["structure", "site_index"], ignore_errors=True
    )

    matminer_feat_df.drop(columns=["structure", "site_index"], inplace=True)

    matminer_feat_cleaned_df = matminer_feat_df.loc[
        :, ~matminer_feat_df.columns.duplicated()
    ]

    return matminer_feat_cleaned_df.dropna(axis=1)


def get_lobster_feats(path_to_lobster_calcs: str, n_jobs: int = 8) -> pd.DataFrame:
    """
    Get featurized dataframe using parent directory path with LOBSTER calcs.

    Uses LobsterPy featurizer implementations.

    Args:
        path_to_lobster_calcs: Path to parent directory containing all LOBSTER calcs
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


def get_lobster_site_feat(path_to_lobster_calc: str, site_index: int) -> pd.DataFrame:
    """
    Get featurized dataframe using path provided containing LOBSTER calc files.

    Uses LobsterPy featurizer implementations.

    Args:
        path_to_lobster_calc: Path to directory containing LOBSTER calcs output files
        site_index: site index of structure

    Returns:
        A pandas DataFrame with site features extracted from LOBSTER calculation files
    """

    file_paths = get_file_paths(
        path_to_lobster_calc=path_to_lobster_calc,
        requested_files=["structure", "icohplist", "charge"],
    )

    icoxx_featurizer = FeaturizeIcoxxlist(
        path_to_icoxxlist=file_paths.get("icohplist"),
        path_to_structure=file_paths.get("structure"),
        bin_width=0.1,
        normalization="counts",
        max_length=5,
    )

    chargeobj = Charge(filename=file_paths.get("charge"))

    site_bwdf = icoxx_featurizer.calc_site_bwdf(site_index=site_index)

    bwdf_values = site_bwdf[f"{site_index}"]["icoxx_binned"]

    stats_fns = {
        "max": np.max,
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "sum": np.sum,
        "skew": skew,
        "kurtosis": kurtosis,
    }

    site_feats = {
        f"site_bwdf_{name}": fn(bwdf_values) for name, fn in stats_fns.items()
    }
    site_feats.update(
        {
            "site_asi": icoxx_featurizer.calc_site_asymmetry_index(site_index),
            "charge_loew": chargeobj.loewdin[site_index],
            "charge_mull": chargeobj.mulliken[site_index],
        }
    )

    structure_name = file_paths["structure"].parent.name
    index_name = f"{structure_name}_{site_index}"

    return pd.DataFrame(data=site_feats, index=[index_name])
