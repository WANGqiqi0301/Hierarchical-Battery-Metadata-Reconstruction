# proposed_framework/training/losses.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple

import torch


def bin_index_from_edges(
    x: torch.Tensor,
    edges: List[Tuple[float, float]],
) -> torch.Tensor:
    """
    Map values to bin indices based on given bin edges.

    For all bins except the last:
        lo <= x < hi

    For the last bin:
        lo <= x <= hi
    """
    idx = torch.zeros_like(x, dtype=torch.long)

    for i, (lo, hi) in enumerate(edges):
        if i < len(edges) - 1:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi)

        idx[mask] = i

    return torch.clamp(idx, 0, len(edges) - 1)


def soc_bin_index_from_edges(
    soc_raw: torch.Tensor,
    edges: List[Tuple[float, float]],
) -> torch.Tensor:
    """
    Backward-compatible alias for SOC bin indexing.
    """
    return bin_index_from_edges(soc_raw, edges)


def heteroscedastic_nll_per_sample(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample heteroscedastic Gaussian negative log-likelihood.
    """
    logvar = torch.clamp(logvar, min=-10.0, max=5.0)
    inv_var = torch.exp(-logvar)

    return 0.5 * (inv_var * (y - mu) ** 2 + logvar)


def heteroscedastic_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Mean heteroscedastic Gaussian negative log-likelihood.
    """
    return heteroscedastic_nll_per_sample(mu, logvar, y).mean()