# proposed_framework/data/pulse_dataset.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from proposed_framework.data.feature_builder import build_three_channel_representation


class HierPulseDataset(Dataset):
    """
    PyTorch dataset for hierarchical battery passport reconstruction.

    Each sample returns:
    - x3: three-channel pulse-response representation,
          shape (3, 5, 8) if c_rate_combo=None,
          shape (3, len(c_rate_combo), 8) if c_rate_combo is provided.
    - pt: normalized pulse-width feature, shape (1,)
    - y_cls: material-capacity class label
    - soc: SOC target
    - soh: SOH target
    """

    def __init__(
        self,
        X_u: np.ndarray,
        y_cls: np.ndarray,
        meta: pd.DataFrame,
        soc_col: str,
        soh_col: str,
        pt_col: str = "pulse_ms",
        use_pt_as_feature: bool = True,
        pt_norm: Optional[Tuple[float, float]] = None,
        normalize_soc: bool = True,
        zscore_normalize: bool = True,
        soc_norm: Optional[Tuple[float, float]] = None,
        soh_norm: Optional[Tuple[float, float]] = None,
        c_rate_combo: Optional[List[int]] = None,
    ):
        self.X_u = X_u
        self.y_cls = y_cls.astype(np.int64)
        self.meta = meta.reset_index(drop=True)
        self.c_rate_combo = c_rate_combo

        if soc_col not in self.meta.columns or soh_col not in self.meta.columns:
            raise RuntimeError(
                f"Meta must contain soc_col='{soc_col}' and soh_col='{soh_col}'."
            )

        soc = self.meta[soc_col].astype(float).to_numpy(dtype=np.float32)
        soh = self.meta[soh_col].astype(float).to_numpy(dtype=np.float32)

        self.normalize_soc = bool(normalize_soc)
        self.zscore_normalize = bool(zscore_normalize)

        if self.normalize_soc:
            soc = soc / 100.0

        if self.zscore_normalize:
            if soc_norm is None or soh_norm is None:
                raise RuntimeError(
                    "zscore_normalize=True requires soc_norm and soh_norm from training data."
                )

            soc_mean, soc_std = float(soc_norm[0]), float(soc_norm[1])
            soh_mean, soh_std = float(soh_norm[0]), float(soh_norm[1])

            soc = (soc - soc_mean) / (soc_std + 1e-8)
            soh = (soh - soh_mean) / (soh_std + 1e-8)

        self.soc = soc
        self.soh = soh

        self.use_pt = bool(use_pt_as_feature)

        if self.use_pt and pt_col in self.meta.columns:
            self.pt_ms = self.meta[pt_col].astype(float).to_numpy(dtype=np.float32)
            p = np.log1p(self.pt_ms)

            if pt_norm is None:
                self.pt_mean = float(p.mean())
                self.pt_std = float(p.std() + 1e-8)
            else:
                self.pt_mean = float(pt_norm[0])
                self.pt_std = float(pt_norm[1])
        else:
            self.pt_ms = None
            self.pt_mean = 0.0
            self.pt_std = 1.0

    def __len__(self) -> int:
        return int(self.X_u.shape[0])

    def __getitem__(self, idx: int):
        x3 = torch.from_numpy(
            build_three_channel_representation(
                self.X_u[idx],
                c_rate_combo=self.c_rate_combo,
            )
        )

        y_cls = torch.tensor(self.y_cls[idx], dtype=torch.long)

        if self.use_pt and self.pt_ms is not None:
            p = (np.log1p(float(self.pt_ms[idx])) - self.pt_mean) / self.pt_std
            pt = torch.tensor([p], dtype=torch.float32)
        else:
            pt = torch.tensor([0.0], dtype=torch.float32)

        soc = torch.tensor(float(self.soc[idx]), dtype=torch.float32)
        soh = torch.tensor(float(self.soh[idx]), dtype=torch.float32)

        return x3, pt, y_cls, soc, soh