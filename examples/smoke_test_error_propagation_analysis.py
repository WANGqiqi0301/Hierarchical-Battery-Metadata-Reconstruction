# -*- coding: utf-8 -*-
"""
examples/smoke_test_error_propagation_analysis.py

Smoke test for:
    analysis/error_propagation_analysis.py

This test does NOT require real battery data.
It uses a tiny fake model and fake dataloader to verify:
    1) module import works
    2) E0-E3 counterfactual evaluation works
    3) repeated summary CSV can be generated
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from analysis.error_propagation_analysis import run_error_propagation_analysis


class FakeFlow(nn.Module):
    """
    Minimal fake conditional 1D flow.

    It only implements:
        sample(context, num_samples)
    which is enough for error_propagation_analysis.
    """
    def __init__(self, context_dim: int):
        super().__init__()
        self.net = nn.Linear(context_dim, 1)

    def sample(self, context: torch.Tensor, num_samples: int = 16) -> torch.Tensor:
        """
        Return shape (num_samples, B, 1), like nflows often does.
        """
        mean = self.net(context)  # (B, 1)
        B = context.size(0)
        noise = 0.01 * torch.randn(
            int(num_samples),
            B,
            1,
            device=context.device,
            dtype=context.dtype,
        )
        return mean.unsqueeze(0) + noise


class FakeEncoder(nn.Module):
    def __init__(self, z_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 5 * 8, z_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FakeHierModel(nn.Module):
    """
    Fake version of the proposed hierarchical model.

    This version intentionally uses material head with [z, pt],
    so the smoke test checks the newer z + pt material-head logic.
    """
    def __init__(self, num_classes: int = 4, z_dim: int = 8, use_pt: bool = True):
        super().__init__()
        self.use_pt = bool(use_pt)
        self.encoder = FakeEncoder(z_dim=z_dim)

        pt_dim = 1 if self.use_pt else 0
        self.head_mat = nn.Sequential(
            nn.Linear(z_dim + pt_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

        self.soc_flow = FakeFlow(context_dim=z_dim + num_classes + pt_dim)
        self.soh_flow = FakeFlow(context_dim=z_dim + num_classes + 1 + pt_dim)


def build_fake_loader(
    n: int = 24,
    num_classes: int = 4,
    batch_size: int = 8,
) -> DataLoader:
    x3 = torch.randn(n, 3, 5, 8)
    pt = torch.randn(n, 1)

    y_cls = torch.randint(low=0, high=num_classes, size=(n,))

    # Fake target-space SOC/SOH.
    # Here we use already normalized-ish values.
    soc = torch.rand(n)           # 0~1
    soh = 0.8 + 0.2 * torch.rand(n)

    ds = TensorDataset(x3, pt, y_cls, soc, soh)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def main():
    num_classes = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FakeHierModel(num_classes=num_classes, z_dim=8, use_pt=True).to(device)
    loader = build_fake_loader(n=24, num_classes=num_classes, batch_size=8)

    save_dir = os.path.join(
        "results",
        "analysis",
        "error_propagation_smoke_test",
    )

    df_runs, df_summary = run_error_propagation_analysis(
        model=model,
        test_loader=loader,
        num_classes=num_classes,
        soc_norm=None,
        soh_norm=None,
        normalize_soc=True,
        zscore_normalize=False,
        device=device,
        n_mc_soc=8,
        n_mc_soh=8,
        repeats=2,
        base_seed=2026,
        save_dir=save_dir,
    )

    print("\n[SMOKE TEST PASSED]")
    print("df_runs shape:", df_runs.shape)
    print("df_summary shape:", df_summary.shape)
    print("Saved to:", save_dir)


if __name__ == "__main__":
    main()