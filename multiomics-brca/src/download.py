"""Download the TCGA-BRCA layers from the legacy UCSC Xena TCGA hub.

Files are fetched as-is (gzipped where Xena serves them gzipped) into data/raw/,
and every download is logged to data/raw/SOURCES.md with its URL, date, size,
and the sample count read back from the file header. Existing files are skipped
so the script is safe to re-run.

Usage:  python src/download.py
"""
import datetime as dt
import gzip
import json
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]


def xena_url(base_url, dataset):
    # Xena stores "TCGA.BRCA.sampleMap/HiSeqV2" as one S3 key, so the slash must be encoded.
    return f"{base_url}/{quote(dataset, safe='')}"


def fetch(url, dest):
    if dest.exists():
        print(f"  exists, skipping: {dest.name}")
        return
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.rename(dest)


def count_columns(path):
    """Number of sample columns in a genomic matrix (header minus the feature-ID column)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return len(fh.readline().rstrip("\n").split("\t")) - 1


def count_rows(path):
    """Number of data rows (clinical matrix has one row per sample)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return sum(1 for _ in fh) - 1


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    base = cfg["data"]["base_url"]
    raw = ROOT / cfg["data"]["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()

    rows = []
    for layer, dataset in cfg["data"]["datasets"].items():
        url = xena_url(base, dataset)
        dest = raw / dataset.split("/")[-1]
        print(f"[{layer}] {dataset}")
        fetch(url, dest)

        # Xena publishes a JSON metadata file next to each dataset; keep it for provenance.
        meta_name = dataset.removesuffix(".gz") + ".json"
        meta_dest = raw / (dest.name.removesuffix(".gz") + ".json")
        fetch(xena_url(base, meta_name), meta_dest)
        meta = json.load(open(meta_dest))

        if layer == "clinical":
            n = count_rows(dest)
            what = "samples (rows)"
        else:
            n = count_columns(dest)
            what = "samples (columns)"
        rows.append((layer, dataset, meta.get("longTitle", meta.get("label", "")),
                     meta.get("version", ""), url, today,
                     f"{dest.stat().st_size / 1e6:.1f} MB", f"{n} {what}"))

    lines = [
        "# Raw data sources",
        "",
        f"Cohort: {cfg['data']['cohort']} on the legacy UCSC Xena TCGA hub "
        "(https://tcga.xenahubs.net). Files are fetched from the hub's S3 download bucket.",
        "Written by `src/download.py`. Do not edit by hand.",
        "",
        "| Layer | Xena dataset ID | Title | Xena version | URL | Downloaded | Size | Count |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    (raw / "SOURCES.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {raw / 'SOURCES.md'}")


if __name__ == "__main__":
    main()
