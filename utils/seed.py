# utils/seed.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)