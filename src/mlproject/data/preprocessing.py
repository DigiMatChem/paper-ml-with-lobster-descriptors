import pandas as pd
from pathlib import Path
from typing import Literal

UNIT_RULES = [
    # --- Matminer ---
    (True, ("MagpieData", "MeltingT"), " (K)"),
    (True, ("MagpieData", "GSbandgap"), " (eV)"),
    (True, ("MagpieData", "GSvolume_pa"), " (A^3/atom)"),
    (True, ("MagpieData", "CovalentRadius"), " (pm)"),
    (True, ("AtomicOrbitals", "LUMO_energy"), " (eV)"),
    (True, ("AtomicOrbitals", "gap_AO"), " (eV)"),
    (True, ("AtomicOrbitals", "HOMO_energy"), " (eV)"),
    (True, ("AtomicOrbitals", "band center"), " (eV)"),
    (True, ("Miedema", "deltaH_amor"), " (eV/atom)"),
    (True, ("Miedema", "deltaH_ss_min"), " (eV/atom)"),
    (True, ("Miedema", "deltaH_inter"), " (eV/atom)"),
    (False, ("AverageBondLength",), " (A)"),
    (False, ("AverageBondAngle",), " (rad)"),
    (True, ("DensityFeatures", "vpa"), " (A^3/atom)"),
    (True, ("DensityFeatures", "density"), " (g/cm^3)"),
    # --- LOBSTER ---
    (False, ("w_ICOHP",), " (eV)"),
    (False, ("ICOHP_", "asi_"), " (eV)"),
    (
        False,
        ("bwdf_", "Icohp_", "center_COHP", "width_COHP", "edge_COHP", "Madelung_"),
        " (eV)",
    ),
    (False, ("dist_at",), " (A)"),
]


def apply_unit_rule(col: str, new_name: str) -> str:
    """Apply first matching unit rule."""
    if "_kurtosis" in col or "_skew" in col:
        return new_name

    for match_all, substrings, unit in UNIT_RULES:
        if match_all:
            if all(s in col or s in new_name for s in substrings):
                return new_name + unit if not new_name.endswith(unit) else new_name
        else:
            if any(s in col or s in new_name for s in substrings):
                return new_name + unit if not new_name.endswith(unit) else new_name

    return new_name


def build_rename_dict(columns: list[str], clean_names: bool = False) -> dict:
    """Return rename mapping for given columns."""
    renamed = {}
    for col in columns:
        new_name = str(col)
        if clean_names:
            new_name = (
                new_name.replace("[", "_")
                .replace("]", "_")
                .replace(" ", "_")
                .replace("|", "_")
                .replace("<", "less")
                .replace("=", "_")
            )
        renamed[col] = apply_unit_rule(str(col), new_name)
    return renamed


def get_dataset(
    target_name: str = "last_phdos_peak",
    feat_type: Literal["matminer", "matminer_lob"] = "matminer",
    data_parent_dir: (
        str | Path
    ) = "/home/anaik/Work/Dev_Codes/paper-ml-with-bondng-descriptors/data",
    rename_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load target and feature datasets for a given target and feature type.

    Parameters
    ----------
    target_name : str, default="last_phdos_peak"
        Name of the target dataset to load.
    feat_type : {"matminer", "matminer_lob"}, default="matminer"
        Feature type used to select feature data.
    data_parent_dir : str | Path
        Parent directory containing `targets` and `features` subdirectories.
    rename_features : bool, default=True
        If True, clean feature column names and append physical units.

    Returns
    -------
    target_df : pd.DataFrame
        DataFrame containing target values.
    feature_df : pd.DataFrame
        DataFrame containing feature values.

    Raises
    ------
    ValueError
        If `feat_type` is not one of the allowed values.
    """
    data_parent_dir = Path(data_parent_dir)

    # Validate feat_type
    allowed_types = {"matminer", "matminer_lob"}
    if feat_type not in allowed_types:
        raise ValueError(
            f"Invalid feat_type: {feat_type}. Must be one of {allowed_types}."
        )

    # Paths
    target_path = data_parent_dir / "targets" / f"{target_name}.json"
    feature_path_matminer = (
        data_parent_dir / "features" / "matminer" / f"{target_name}.json.gz"
    )

    # Load targets + features
    target_df = pd.read_json(target_path)
    feature_df = pd.read_json(feature_path_matminer)

    if feat_type == "matminer_lob":
        # Merge lobster features
        feature_path_lobster = (
            data_parent_dir / "features" / "lobster" / f"{target_name}.json.gz"
        )
        feature_df_lob = pd.read_json(feature_path_lobster)
        feature_df = feature_df.merge(
            feature_df_lob, left_index=True, right_index=True, how="inner"
        )

    # Optionally rename + add units
    if rename_features:
        feature_df = feature_df.rename(
            columns=build_rename_dict(feature_df.columns, clean_names=True)
        )

    return target_df.loc[feature_df.index, :], feature_df
