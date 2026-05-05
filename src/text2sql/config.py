"""Central project configuration for local paths and default parameters."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# BIRD dev paths. We support both a raw extracted package layout and the
# Hugging Face `save_to_disk` layout already downloaded into `data/bird_dev`.
BIRD_DEV_DIR = DATA_DIR / "bird_dev"
BIRD_DEV_HF_DIR = BIRD_DEV_DIR / "dev_20251106"

DEV_JSON_PATH = BIRD_DEV_DIR / "dev.json"
DEV_TABLES_PATH = BIRD_DEV_DIR / "dev_tables.json"
DEV_DATABASE_DIR = BIRD_DEV_DIR / "dev_databases"

# Optional complete package download locations some users may use later.
SQLITE_DATABASE_DIR = DEV_DATABASE_DIR

DEFAULT_SQL_TIMEOUT = 5
MAX_EVAL_SAMPLES = 100


def ensure_project_dirs() -> None:
    """Create project directories that should always exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_dev_dataset_source() -> str:
    """Return the available dev dataset source type."""
    if DEV_JSON_PATH.exists():
        return "json"
    if BIRD_DEV_HF_DIR.exists():
        return "hf_disk"
    return "missing"


def describe_paths() -> dict:
    """Collect the key configured paths and whether they currently exist."""
    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATA_DIR": DATA_DIR,
        "BIRD_DEV_DIR": BIRD_DEV_DIR,
        "BIRD_DEV_HF_DIR": BIRD_DEV_HF_DIR,
        "DEV_JSON_PATH": DEV_JSON_PATH,
        "DEV_DATABASE_DIR": DEV_DATABASE_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "DEFAULT_SQL_TIMEOUT": DEFAULT_SQL_TIMEOUT,
        "DATASET_SOURCE": get_dev_dataset_source(),
    }


if __name__ == "__main__":
    ensure_project_dirs()
    for key, value in describe_paths().items():
        if isinstance(value, Path):
            print(f"{key}: {value} (exists={value.exists()})")
        else:
            print(f"{key}: {value}")
