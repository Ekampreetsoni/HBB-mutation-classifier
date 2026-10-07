"""Day 1: patient matching, cleaning, and QC for the three TCGA-BRCA layers.

Steps
1. Load RNA, RPPA, CNV matrices (features x samples) and the clinical matrix.
2. Keep primary tumour samples only (TCGA sample type code "01"), convert the
   15-char sample barcode to the 12-char patient ID, one sample per patient.
3. Intersect patients across the three layers and drop patients without a PAM50 call.
4. Drop features with more than 20 percent missing and near-constant features.
5. Save patient x feature matrices as parquet plus labels.csv, and QC figures.

What this script deliberately does NOT do: imputation, scaling, or any
label-aware feature selection. Those are fit inside each training fold later
(Day 2) so that nothing learned from the test set leaks into training.
The filters here are label-free and per-feature, so they cannot leak label information.

Usage:  python src/preprocess.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: figures are written to disk only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LAYERS = ["rna", "rppa", "cnv"]
# One fixed label order and colour per subtype, reused by every figure and the dashboard.
SUBTYPE_ORDER = ["LumA", "LumB", "Her2", "Basal", "Normal"]
SUBTYPE_COLORS = {"LumA": "#1f77b4", "LumB": "#7fb3e0", "Her2": "#d62728",
                  "Basal": "#2b2b2b", "Normal": "#2ca02c"}


def sample_type_code(barcodes):
    # TCGA barcode TCGA-XX-XXXX-STV: characters 13:15 are the sample type (01 = primary solid tumour).
    return barcodes.str[13:15]


def load_layer(path):
    """Read a Xena genomic matrix and return a patient x feature frame of primary tumours."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.loc[:, sample_type_code(df.columns) == "01"]
    df = df.T
    df.index = df.index.str[:12]
    # Xena barcodes here carry no vial letter, so a patient should appear once; keep the first if not.
    n_dup = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "patient_id"
    return df, n_dup


def clean(df, max_missing, min_var):
    """Label-free feature filters. Returns the cleaned frame and counts for the QC log."""
    n0 = df.shape[1]
    miss = df.isna().mean()
    df = df.loc[:, miss <= max_missing]
    n_missing = n0 - df.shape[1]
    var = df.var(skipna=True)
    df = df.loc[:, var > min_var]
    n_const = n0 - n_missing - df.shape[1]
    return df, {"features_raw": n0, "dropped_missing": n_missing,
                "dropped_near_constant": n_const, "features_kept": df.shape[1]}, miss


def pca_plot(ax, X, labels, title):
    # Median-impute only for this picture; the modelling pipeline imputes inside folds instead.
    Xf = X.fillna(X.median())
    Z = StandardScaler().fit_transform(Xf)
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(Z)
    for s in SUBTYPE_ORDER:
        m = (labels == s).values
        ax.scatter(P[m, 0], P[m, 1], s=8, alpha=0.7, c=SUBTYPE_COLORS[s], label=s)
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title(title)


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    raw = ROOT / cfg["data"]["raw_dir"]
    out = ROOT / cfg["data"]["processed_dir"]
    fig_dir = ROOT / "results" / "figures"
    tab_dir = ROOT / "results" / "tables"
    for d in (out, fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Labels: primary tumour rows of the clinical matrix with a PAM50 call.
    clin = pd.read_csv(raw / "BRCA_clinicalMatrix", sep="\t", index_col=0, low_memory=False)
    clin = clin[sample_type_code(clin.index) == "01"]
    lab = clin[cfg["data"]["label_column"]].dropna()
    lab = pd.Series(lab.values, index=lab.index.str[:12], name="pam50")
    lab = lab[~lab.index.duplicated(keep="first")]

    layers, dups = {}, {}
    for name in LAYERS:
        fname = cfg["data"]["datasets"][name].split("/")[-1]
        layers[name], dups[name] = load_layer(raw / fname)

    log = {"primary_tumour_patients": {k: int(v.shape[0]) for k, v in layers.items()},
           "duplicate_samples_dropped": dups,
           "patients_with_pam50": int(lab.shape[0])}

    common = set(lab.index)
    for df in layers.values():
        common &= set(df.index)
    patients = sorted(common)
    log["patients_in_all_layers_with_pam50"] = len(patients)
    labels = lab.loc[patients]

    qc_cfg = cfg["qc"]
    missing_before = {}
    log["features"] = {}
    for name in LAYERS:
        df = layers[name].loc[patients]
        df, counts, miss = clean(df, qc_cfg["max_missing_frac"], qc_cfg["min_variance"])
        missing_before[name] = miss
        log["features"][name] = counts
        # Prefix every feature with its layer so fused matrices and SHAP output stay unambiguous.
        df.columns = [f"{name}__{c}" for c in df.columns]
        df.astype("float32").to_parquet(out / f"{name}.parquet")
        layers[name] = df

    # sample_id records the primary tumour barcode actually used (patient ID + "-01").
    label_df = pd.DataFrame({"patient_id": patients,
                             "sample_id": [p + "-01" for p in patients],
                             "pam50": labels.values})
    label_df.to_csv(out / "labels.csv", index=False)

    counts = labels.value_counts().reindex(SUBTYPE_ORDER).fillna(0).astype(int)
    log["class_counts"] = counts.to_dict()
    (tab_dir / "qc_summary.json").write_text(json.dumps(log, indent=2))
    counts.rename_axis("pam50").rename("n").to_csv(tab_dir / "class_counts.csv")

    # QC figure 1: class counts.
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, counts.values, color=[SUBTYPE_COLORS[s] for s in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Patients")
    ax.set_title(f"PAM50 class counts (N = {len(patients)})")
    fig.tight_layout()
    fig.savefig(fig_dir / "qc_class_counts.png", dpi=150)
    plt.close(fig)

    # QC figure 2: missingness and what each filter removed, one panel per layer.
    # A bar chart rather than a histogram because these Xena matrices can have no missing
    # values at all, and a histogram piled up at 0 percent is unreadable.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    steps = ["features_raw", "dropped_missing", "dropped_near_constant", "features_kept"]
    step_labels = ["Raw", f">{int(qc_cfg['max_missing_frac'] * 100)}%\nmissing",
                   "Near-\nconstant", "Kept"]
    for ax, name in zip(axes, LAYERS):
        vals = [log["features"][name][s] for s in steps]
        ax.bar(step_labels, vals, color=["grey", "#d62728", "#ff7f0e", "#2ca02c"])
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        miss_cells = float(missing_before[name].mean() * 100)
        ax.set_title(f"{name.upper()}: {miss_cells:.2f}% of values missing", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Features")
    fig.tight_layout()
    fig.savefig(fig_dir / "qc_missingness.png", dpi=150)
    plt.close(fig)

    # QC figure 3: PCA per layer coloured by subtype.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, name in zip(axes, LAYERS):
        pca_plot(ax, layers[name], labels, f"{name.upper()} ({layers[name].shape[1]} features)")
    axes[-1].legend(title="PAM50", markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "qc_pca_per_layer.png", dpi=150)
    plt.close(fig)

    print(json.dumps(log, indent=2))
    print(f"\nFinal N = {len(patients)} patients")
    print(counts.to_string())
    small = counts[counts < qc_cfg["min_class_size"]]
    if len(small):
        print(f"\nWARNING: classes below {qc_cfg['min_class_size']} samples: {small.to_dict()}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
