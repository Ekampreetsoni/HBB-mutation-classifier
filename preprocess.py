"""
preprocess.py
-------------
Load and preprocess ClinVar HBB mutation data for annotation-based classification.
"""

import pandas as pd
import numpy as np


def load_clinvar(filepath: str) -> pd.DataFrame:
    """Load the ClinVar TSV export."""
    df = pd.read_csv(filepath, sep="\t", low_memory=False)
    # Drop unnamed/empty trailing columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def filter_hbb(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that include HBB in the Gene(s) column."""
    mask = df["Gene(s)"].fillna("").str.contains(r"\bHBB\b", regex=True)
    return df[mask].copy()


def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Germline classification to binary:
        Pathogenic / Likely pathogenic  → 1
        Benign / Likely benign          → 0
    Rows with uncertain / conflicting values are dropped.
    """
    col = "Germline classification"
    pathogenic_terms = ["Pathogenic", "Likely pathogenic"]
    benign_terms = ["Benign", "Likely benign"]

    def _label(val):
        if not isinstance(val, str):
            return np.nan
        v = val.strip()
        if any(t in v for t in pathogenic_terms):
            return 1
        if any(t in v for t in benign_terms):
            return 0
        return np.nan

    df = df.copy()
    df["label"] = df[col].apply(_label)
    return df.dropna(subset=["label"]).reset_index(drop=True)


def encode_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """One-hot encode categorical feature columns, filling NAs with 'Unknown'."""
    df = df.copy()
    for col in feature_cols:
        df[col] = df[col].fillna("Unknown")
    return pd.get_dummies(df[feature_cols + ["label"]], columns=feature_cols)


def get_feature_matrix(df: pd.DataFrame):
    """Return X (features) and y (labels) from preprocessed dataframe."""
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])
    return X, y
