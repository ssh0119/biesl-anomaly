"""Builds cache/index.parquet from the raw metadata.csv.

Filters to patients actually present on disk (the local download may be
partial), and adds derived columns: has_ecg, stage1_label, stage2_group.
Run with: uv run python -m src.data.build_index
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.data.labels import NORMAL_RHYTHM, stage2_group_for

USECOLS = [
    "signal_file_name",
    "folder_path",
    "patient",
    "subject_id",
    "event_rhythm",
    "strat_fold",
    "vector_10s_ecg_sqi",
]
CHUNKSIZE = 500_000


def existing_patients() -> set[str]:
    patients = set()
    for prefix_dir in sorted(config.DATA_ROOT.glob("p??")):
        if prefix_dir.is_dir():
            for p in prefix_dir.iterdir():
                if p.is_dir():
                    patients.add(p.name)
    return patients


def main() -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    on_disk = existing_patients()
    print(f"Found {len(on_disk)} patient folders on disk under {config.DATA_ROOT}")

    total_rows = sum(1 for _ in open(config.METADATA_CSV)) - 1
    chunks = []
    with tqdm(total=total_rows, unit="rows", desc="indexing") as pbar:
        for chunk in pd.read_csv(config.METADATA_CSV, usecols=USECOLS, chunksize=CHUNKSIZE):
            chunk = chunk[chunk["patient"].isin(on_disk)].copy()
            chunk["has_ecg"] = ~chunk["vector_10s_ecg_sqi"].str.contains("nan, nan, nan", na=True)
            chunk["stage1_label"] = (chunk["event_rhythm"] != NORMAL_RHYTHM).astype("int8")
            chunk["stage2_group"] = chunk["event_rhythm"].map(stage2_group_for)
            chunk = chunk.drop(columns=["vector_10s_ecg_sqi"])
            chunks.append(chunk)
            pbar.update(len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(config.INDEX_PARQUET, index=False)

    print(f"\nWrote {len(df):,} rows to {config.INDEX_PARQUET}")
    print(f"has_ecg=True: {df['has_ecg'].mean():.1%}")
    print("\nstage1_label counts:")
    print(df["stage1_label"].value_counts())
    print("\nstage2_group counts (anomalies only):")
    print(df["stage2_group"].value_counts())
    print("\nstrat_fold counts:")
    print(df["strat_fold"].value_counts().sort_index())


if __name__ == "__main__":
    main()
