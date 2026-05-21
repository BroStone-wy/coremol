import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


SPLIT_VERSION = "deepchem_scaffold_v1"


def data_smiles(data) -> str:
    smiles = getattr(data, "smiles", None)
    if smiles is None:
        raise ValueError("MoleculeNet data object does not expose a smiles attribute.")
    if isinstance(smiles, (list, tuple)):
        return smiles[0]
    return str(smiles)


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def valid_graph_indices(dataset) -> list[int]:
    return [idx for idx, data in enumerate(dataset) if int(data.num_nodes) > 0]


def scaffold_split_indices(
    dataset,
    seed: int,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
) -> dict[str, list[int]]:
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx in valid_graph_indices(dataset):
        data = dataset[idx]
        scaffold_to_indices[scaffold_from_smiles(data_smiles(data))].append(idx)

    groups = sorted(scaffold_to_indices.values(), key=lambda group: (len(group), group[0]), reverse=True)

    n_total = sum(len(group) for group in groups)
    train_cutoff = int(train_fraction * n_total)
    valid_cutoff = int((train_fraction + valid_fraction) * n_total)
    train, valid, test = [], [], []

    for group in groups:
        if len(train) + len(group) <= train_cutoff:
            train.extend(group)
        elif len(train) + len(valid) + len(group) <= valid_cutoff:
            valid.extend(group)
        else:
            test.extend(group)

    return {
        "train": sorted(train),
        "valid": sorted(valid),
        "test": sorted(test),
    }


def load_or_create_split(dataset, dataset_name: str, seed: int, split_root: str | Path) -> dict[str, list[int]]:
    split_dir = Path(split_root) / dataset_name.upper() / SPLIT_VERSION
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split.json"
    if split_path.exists():
        return json.loads(split_path.read_text())

    split = scaffold_split_indices(dataset, seed=seed)
    split_path.write_text(json.dumps(split, indent=2))
    return split
