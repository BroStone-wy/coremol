import os

import torch

from coremol.utils.seed import configure_deterministic_environment, set_seed


def test_configure_deterministic_environment_sets_cublas_workspace(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    configure_deterministic_environment()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_set_seed_enables_deterministic_cuda_backend_flags():
    set_seed(123)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
