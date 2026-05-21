import os
import random

import numpy as np
import torch


def configure_deterministic_environment() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_seed(seed: int) -> None:
    configure_deterministic_environment()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
