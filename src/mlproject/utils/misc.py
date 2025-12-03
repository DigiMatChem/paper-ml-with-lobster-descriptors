"""Misc utility functions"""

def split_features(feats, lob_keywords=None):
    """
    Splits the columns of a DataFrame into two lists: lob_feats and matminer_feats.

    Parameters:
        feats (list): list of feature names.
        lob_keywords (list of str, optional): List of keywords to identify lob_feats.
            Defaults to a predefined list.

    Returns:
        tuple: (lob_feats, matminer_feats)
    """
    if lob_keywords is None:
        lob_keywords = [
            "bwdf", "ICOHP", "Ionicity_", "antibonding_perc", "Icohp", "asi_",
            "COHP", "bonding_perc", "Madelung_", "_asi", "charge_mull", "charge_loew"
        ]

    lob_feats = [col for col in feats if any(keyword in col for keyword in lob_keywords)]
    matminer_feats = [col for col in feats if col not in lob_feats]

    return lob_feats, matminer_feats