from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "dataset" / "mimic-data"
METADATA_CSV = DATA_ROOT / "metadata.csv"
CACHE_DIR = ROOT / "cache"
INDEX_PARQUET = CACHE_DIR / "index.parquet"
CHECKPOINT_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs"

FS = 125  # Hz, sampling rate for every channel in this dataset
SEGMENT_LEN = 3750  # samples = 30s at 125 Hz
ECG_CHANNEL = "II"
PPG_CHANNEL = "PLETH"

# strat_fold (0-9) is provided by the dataset for patient-disjoint ML benchmarking splits.
TRAIN_FOLDS = tuple(range(0, 8))
VAL_FOLDS = (8,)
TEST_FOLDS = (9,)

MODALITIES = ("ecg", "ppg", "both")
