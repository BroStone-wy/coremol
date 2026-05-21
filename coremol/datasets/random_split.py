import json
from pathlib import Path

import numpy as np


SPLIT_VERSION = "curvflow_random_70_20_10_v1"


def curvflow_random_split_indices(
    num_items: int,
    seed: int,
    train_fraction: float = 0.7,
    valid_fraction: float = 0.2,
) -> dict[str, list[int]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(num_items)
    test_size = num_items - int(num_items * train_fraction) - int(num_items * valid_fraction)
    valid_size = int(num_items * valid_fraction)
    train_valid, test = _train_test_split_indices(indices, test_size=test_size, rng=rng)
    train, valid = _train_test_split_indices(train_valid, test_size=valid_size, rng=rng)
    return {
        "train": sorted(int(i) for i in train),
        "valid": sorted(int(i) for i in valid),
        "test": sorted(int(i) for i in test),
    }


def _train_test_split_indices(indices, test_size: int, rng) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    test = shuffled[:test_size]
    train = shuffled[test_size:]
    return train, test


def random_split_version(train_fraction: float = 0.7, valid_fraction: float = 0.2) -> str:
    train = int(round(train_fraction * 100))
    valid = int(round(valid_fraction * 100))
    test = 100 - train - valid
    return f"curvflow_random_{train}_{valid}_{test}_v1"


def load_or_create_random_split(
    dataset,
    dataset_name: str,
    seed: int,
    split_root: str | Path,
    train_fraction: float = 0.7,
    valid_fraction: float = 0.2,
):
    split_dir = Path(split_root) / dataset_name.upper() / random_split_version(train_fraction, valid_fraction) / f"seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split.json"
    if split_path.exists():
        return json.loads(split_path.read_text())

    valid_indices = [idx for idx, data in enumerate(dataset) if int(data.num_nodes) > 0]
    raw_split = curvflow_random_split_indices(
        len(valid_indices),
        seed,
        train_fraction=train_fraction,
        valid_fraction=valid_fraction,
    )
    split = {name: [valid_indices[i] for i in values] for name, values in raw_split.items()}
    split_path.write_text(json.dumps(split, indent=2))
    return split
