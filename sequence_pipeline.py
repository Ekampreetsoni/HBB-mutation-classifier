"""
sequence_pipeline.py
---------------------
Sequence-based mutation classification pipeline.

Steps:
    1. Parse SPDI notation to extract chromosomal coordinates + alleles
    2. Fetch ±window bp of reference sequence from Ensembl REST API
    3. Apply the real ALT allele to generate a mutated sequence
    4. Encode both sequences as k-mer frequency vectors
    5. Use the difference vector as input features for classification

Usage:
    python src/sequence_pipeline.py --data data/clinvar_result.txt --max_variants 200
"""

import argparse
import json
import os
import time
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from preprocess import filter_hbb, load_clinvar, make_binary_label

# ─── Configuration ───────────────────────────────────────────────────────────
ENSEMBL_BASE = "https://rest.ensembl.org"
WINDOW = 20          # bases either side of mutation
K = 3                # k-mer size
REQUEST_DELAY = 0.35 # seconds between Ensembl calls (rate limit ~3/s)
RESULTS_DIR = "results"


# ─── SPDI Parsing ────────────────────────────────────────────────────────────
def parse_spdi(spdi: str):
    """
    Parse an SPDI string like 'NC_000011.10:5226966:A:T' into
    (accession, position, ref, alt).
    Returns None if the field is missing or non-standard.
    """
    if not isinstance(spdi, str) or spdi.strip() == "":
        return None
    parts = spdi.strip().split(":")
    if len(parts) != 4:
        return None
    accession, pos, ref, alt = parts
    try:
        pos = int(pos)
    except ValueError:
        return None
    # Only SNVs for now (single-char ref and alt)
    if len(ref) != 1 or len(alt) != 1:
        return None
    return accession, pos, ref, alt


def accession_to_chrom(accession: str) -> str:
    """Map NC_ accession to chromosome number (GRCh38)."""
    mapping = {
        "NC_000001.11": "1",  "NC_000002.12": "2",  "NC_000003.12": "3",
        "NC_000004.12": "4",  "NC_000005.10": "5",  "NC_000006.12": "6",
        "NC_000007.14": "7",  "NC_000008.11": "8",  "NC_000009.12": "9",
        "NC_000010.11": "10", "NC_000011.10": "11", "NC_000012.12": "12",
        "NC_000013.11": "13", "NC_000014.9": "14",  "NC_000015.10": "15",
        "NC_000016.10": "16", "NC_000017.11": "17", "NC_000018.12": "18",
        "NC_000019.10": "19", "NC_000020.11": "20", "NC_000021.9": "21",
        "NC_000022.11": "22", "NC_000023.11": "X",  "NC_000024.10": "Y",
    }
    return mapping.get(accession, accession)


# ─── Ensembl Sequence Fetch ───────────────────────────────────────────────────
def fetch_sequence(chrom: str, start: int, end: int, retries: int = 3) -> str | None:
    """
    Fetch human reference sequence from Ensembl REST API.
    Coordinates are 1-based, inclusive.
    """
    url = f"{ENSEMBL_BASE}/sequence/region/human/{chrom}:{start}..{end}:1"
    headers = {"Content-Type": "text/plain"}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.text.strip().upper()
            elif r.status_code == 429:          # rate limited
                time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(1)
    return None


# ─── Mutation Application ─────────────────────────────────────────────────────
def apply_mutation(seq: str, ref: str, alt: str, center: int) -> str | None:
    """
    Replace the nucleotide at `center` index in `seq` with `alt`.
    Validates that the expected `ref` is at that position.
    """
    if seq[center] != ref:
        return None          # REF mismatch – skip this variant
    return seq[:center] + alt + seq[center + 1:]


# ─── K-mer Encoding ──────────────────────────────────────────────────────────
def all_kmers(k: int):
    """Generate all possible k-mers as a sorted list."""
    return ["".join(p) for p in product("ACGT", repeat=k)]


ALL_3MER = all_kmers(K)
KMER_INDEX = {km: i for i, km in enumerate(ALL_3MER)}


def kmer_vector(seq: str, k: int = K) -> np.ndarray:
    """Return normalised k-mer frequency vector for a DNA sequence."""
    counts = np.zeros(len(ALL_3MER), dtype=float)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if kmer in KMER_INDEX:
            counts[KMER_INDEX[kmer]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def delta_kmer(ref_seq: str, mut_seq: str, k: int = K) -> np.ndarray:
    """Feature vector = difference in k-mer frequencies (mutated − reference)."""
    return kmer_vector(mut_seq, k) - kmer_vector(ref_seq, k)


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def build_sequence_dataset(df: pd.DataFrame, max_variants: int = 300):
    """
    Iterate over HBB variants, fetch sequences, encode k-mers.
    Returns (X, y) arrays.
    """
    features, labels = [], []
    skipped = 0

    for _, row in df.iterrows():
        if len(features) >= max_variants:
            break

        parsed = parse_spdi(row.get("Canonical SPDI", ""))
        if parsed is None:
            skipped += 1
            continue

        accession, pos, ref, alt = parsed
        chrom = accession_to_chrom(accession)

        start = pos - WINDOW + 1   # convert 0-based SPDI to 1-based Ensembl
        end = pos + WINDOW + 1

        ref_seq = fetch_sequence(chrom, start, end)
        time.sleep(REQUEST_DELAY)

        if ref_seq is None or len(ref_seq) < 2 * WINDOW + 1:
            skipped += 1
            continue

        center = WINDOW
        mut_seq = apply_mutation(ref_seq, ref, alt, center)
        if mut_seq is None:
            skipped += 1
            continue

        feat = delta_kmer(ref_seq, mut_seq)
        features.append(feat)
        labels.append(int(row["label"]))

        if len(features) % 20 == 0:
            print(f"  Processed {len(features)} variants (skipped {skipped})")

    print(f"\nFinal: {len(features)} variants encoded, {skipped} skipped.")
    return np.array(features), np.array(labels)


def train_sequence_model(data_path: str, max_variants: int = 300, output_dir: str = RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    raw = load_clinvar(data_path)
    hbb = filter_hbb(raw)
    labelled = make_binary_label(hbb)

    print(f"Building sequence dataset from {len(labelled)} labelled HBB variants…")
    X, y = build_sequence_dataset(labelled, max_variants=max_variants)

    if len(X) < 10:
        print("Not enough variants fetched. Check network access or increase max_variants.")
        return None, {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nSequence Model – Accuracy: {report['accuracy']:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Pathogenic"]))

    metrics = {
        "accuracy": report["accuracy"],
        "roc_auc": auc,
        "n_variants": len(X),
        "k": K,
        "window": WINDOW,
    }
    with open(os.path.join(output_dir, "sequence_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save encoded dataset for reproducibility
    np.save(os.path.join(output_dir, "sequence_X.npy"), X)
    np.save(os.path.join(output_dir, "sequence_y.npy"), y)

    return clf, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/clinvar_result.txt")
    parser.add_argument("--max_variants", type=int, default=200)
    parser.add_argument("--output", default=RESULTS_DIR)
    args = parser.parse_args()
    train_sequence_model(args.data, args.max_variants, args.output)
