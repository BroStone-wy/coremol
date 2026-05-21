import json
from pathlib import Path

from scripts.compute_tcm_variants import limit_dataset


def _repo_path(path: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / path


def test_curvflow_classification_configs_define_six_remaining_datasets():
    value = json.loads(_repo_path("configs/curvflow_classification_value.json").read_text())
    delta = json.loads(_repo_path("configs/curvflow_classification_delta.json").read_text())

    assert value["datasets"] == ["BACE", "CLINTOX", "TOX21", "HIV", "SIDER", "TOXCAST"]
    assert delta["datasets"] == value["datasets"]
    assert value["residual_message"] == "value"
    assert delta["residual_message"] == "delta"


def test_limit_dataset_keeps_full_dataset_when_limit_is_zero():
    dataset = [1, 2, 3]

    assert limit_dataset(dataset, 0) == dataset


def test_limit_dataset_truncates_when_limit_is_positive():
    dataset = [1, 2, 3]

    assert limit_dataset(dataset, 2) == [1, 2]
