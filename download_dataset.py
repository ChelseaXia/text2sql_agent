from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "bird_mini_dev": "birdsql/bird_mini_dev",
    "bird_dev": "birdsql/bird_sql_dev_20251106",
}


def main() -> None:
    data_root = Path("data")
    data_root.mkdir(exist_ok=True)

    for local_name, dataset_name in DATASETS.items():
        target_dir = data_root / local_name
        print(f"Downloading {dataset_name} -> {target_dir}")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(str(target_dir))
        print(f"Saved {dataset_name} to {target_dir}")
        print(f"Splits: {list(dataset.keys())}")


if __name__ == "__main__":
    main()
