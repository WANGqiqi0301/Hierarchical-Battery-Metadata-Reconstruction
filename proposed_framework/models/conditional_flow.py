# proposed_framework/models/conditional_flow.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)


class Conditional1DFlow(nn.Module):
    """
    One-dimensional conditional normalizing flow for scalar regression targets.

    It is used for probabilistic SOC and SOH estimation.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_features: int = 64,
        num_transforms: int = 6,
        num_bins: int = 8,
        tail_bound: float = 3.0,
    ):
        super().__init__()

        transforms = []

        for _ in range(int(num_transforms)):
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=1,
                    hidden_features=int(hidden_features),
                    context_features=int(context_dim),
                    num_bins=int(num_bins),
                    tails="linear",
                    tail_bound=float(tail_bound),
                )
            )

        transform = CompositeTransform(transforms)
        base_dist = StandardNormal([1])

        self.flow = Flow(transform, base_dist)

    def log_prob(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y.unsqueeze(1)

        return self.flow.log_prob(inputs=y, context=context)

    def sample(self, context: torch.Tensor, num_samples: int = 16) -> torch.Tensor:
        return self.flow.sample(num_samples=int(num_samples), context=context)