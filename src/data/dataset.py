import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config
from src.data.labels import STAGE2_CLASS_TO_IDX
from src.data.wfdb_signals import load_channels, normalize

_INDEX_CACHE: pd.DataFrame | None = None


def _load_full_index() -> pd.DataFrame:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        if not config.INDEX_PARQUET.exists():
            raise FileNotFoundError(
                f"{config.INDEX_PARQUET} not found. Run `uv run python -m src.data.build_index` first."
            )
        _INDEX_CACHE = pd.read_parquet(config.INDEX_PARQUET)
    return _INDEX_CACHE


def _folds_for_split(split: str) -> tuple[int, ...]:
    return {"train": config.TRAIN_FOLDS, "val": config.VAL_FOLDS, "test": config.TEST_FOLDS}[split]


class _WFDBDataset(Dataset):
    """Shared loading logic for both cascade stages. Subclasses set self.df and self.label_col."""

    modality: str
    label_col: str
    max_load_retries = 5

    def __init__(self, df: pd.DataFrame, modality: str, label_col: str):
        assert modality in config.MODALITIES
        self.df = df.reset_index(drop=True)
        self.modality = modality
        self.label_col = label_col
        self.channels = {"ecg": ["II"], "ppg": ["PLETH"], "both": ["II", "PLETH"]}[modality]

    def __len__(self) -> int:
        return len(self.df)

    def _build_inputs(self, folder_path: str):
        raw = load_channels(folder_path, self.channels)
        if self.modality == "both":
            return {
                "ecg": torch.from_numpy(normalize(raw["II"])).unsqueeze(0),
                "ppg": torch.from_numpy(normalize(raw["PLETH"])).unsqueeze(0),
            }
        ch = "II" if self.modality == "ecg" else "PLETH"
        return torch.from_numpy(normalize(raw[ch])).unsqueeze(0)

    def __getitem__(self, idx: int):
        for attempt in range(self.max_load_retries):
            row = self.df.iloc[idx]
            try:
                inputs = self._build_inputs(row["folder_path"])
                label = torch.tensor(row[self.label_col], dtype=torch.long)
                return inputs, label
            except Exception:
                if attempt == self.max_load_retries - 1:
                    raise
                idx = random.randrange(len(self.df))


def _maybe_limit(df: pd.DataFrame, limit: int | None, seed: int = 0) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=seed)


class Stage1Dataset(_WFDBDataset):
    """Binary anomaly detection: SR (0) vs. any other rhythm (1)."""

    def __init__(self, modality: str, split: str, limit: int | None = None):
        df = _load_full_index()
        df = df[df["strat_fold"].isin(_folds_for_split(split))]
        if modality in ("ecg", "both"):
            df = df[df["has_ecg"]]
        df = _maybe_limit(df, limit)
        super().__init__(df, modality, label_col="stage1_label")


class Stage2Dataset(_WFDBDataset):
    """Rhythm-type classification, restricted to Stage-1 anomalies."""

    def __init__(self, modality: str, split: str, limit: int | None = None):
        df = _load_full_index()
        df = df[df["strat_fold"].isin(_folds_for_split(split))]
        df = df[df["stage1_label"] == 1]
        df = df[df["stage2_group"].notna()]
        if modality in ("ecg", "both"):
            df = df[df["has_ecg"]]
        df = _maybe_limit(df, limit)
        df = df.copy()
        df["stage2_idx"] = df["stage2_group"].map(STAGE2_CLASS_TO_IDX)
        super().__init__(df, modality, label_col="stage2_idx")
