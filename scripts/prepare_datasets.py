import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coremol.datasets.moleculenet import load_moleculenet
from coremol.datasets.scaffold_split import load_or_create_split


def main():
    data_root = ROOT / "data" / "moleculenet"
    split_root = ROOT / "data" / "splits"
    for name in ["BBBP", "ESOL"]:
        dataset = load_moleculenet(name, data_root)
        for seed in [0, 1, 2]:
            split = load_or_create_split(dataset, name, seed, split_root)
            print(name, seed, {k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()

