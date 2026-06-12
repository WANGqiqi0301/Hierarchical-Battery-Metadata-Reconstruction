# proposed_framework/models/hierarchical_model.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from proposed_framework.models.encoder import MicroResNetEncoder2D3Ch
from proposed_framework.models.conditional_flow import Conditional1DFlow


class Hier3HeadModel(nn.Module):
    """
    Hierarchical probabilistic model.

    Hierarchy:
    1. Material-capacity classification.
    2. SOC estimation conditioned on encoder features, material probabilities and pulse width.
    3. SOH estimation conditioned on encoder features, material probabilities, SOC and pulse width.
    """

    def __init__(
        self,
        num_classes: int,
        width: int = 32,
        blocks: int = 4,
        drop2d: float = 0.0,
        use_pt_as_feature: bool = True,
        soc_hidden: int = 64,
        soh_hidden: int = 64,
        head_dropout: float = 0.2,
        flow_layers: int = 6,
        flow_bins: int = 8,
        flow_tail_bound: float = 3.0,
    ):
        super().__init__()

        self.encoder = MicroResNetEncoder2D3Ch(
            width=width,
            blocks=blocks,
            drop2d=drop2d,
        )

        self.use_pt = bool(use_pt_as_feature)

        pt_dim = 1 if self.use_pt else 0
        p_dim = int(num_classes)

        self.head_mat = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )

        soc_ctx_dim = width + p_dim + pt_dim
        self.soc_flow = Conditional1DFlow(
            context_dim=soc_ctx_dim,
            hidden_features=int(soc_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

        soh_ctx_dim = width + p_dim + 1 + pt_dim
        self.soh_flow = Conditional1DFlow(
            context_dim=soh_ctx_dim,
            hidden_features=int(soh_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

    def _sample_mean_1d(
        self,
        samples: torch.Tensor,
        batch_size: int,
        num_samples: int,
        name: str,
    ) -> torch.Tensor:
        if samples.ndim == 3:
            if samples.shape[0] == int(num_samples) and samples.shape[1] == batch_size:
                return samples.mean(dim=0).squeeze(-1)

            if samples.shape[0] == batch_size and samples.shape[1] == int(num_samples):
                return samples.mean(dim=1).squeeze(-1)

            samples = samples.reshape(int(num_samples), batch_size, 1)
            return samples.mean(dim=0).squeeze(-1)

        if samples.ndim == 2:
            samples = samples.view(int(num_samples), batch_size, 1)
            return samples.mean(dim=0).squeeze(-1)

        raise RuntimeError(f"Unexpected {name} sample shape: {tuple(samples.shape)}")

    def forward(
        self,
        x_img: torch.Tensor,
        x_pt: torch.Tensor,
        soc_tf: Optional[torch.Tensor] = None,
        n_mc: int = 16,
    ):
        z = self.encoder(x_img)
        batch_size = z.size(0)

        logits_mat = self.head_mat(z)
        p_mat = torch.softmax(logits_mat, dim=1)

        if self.use_pt:
            cond_soc = torch.cat([z, p_mat, x_pt], dim=1)
        else:
            cond_soc = torch.cat([z, p_mat], dim=1)

        soc_logp = None

        if soc_tf is not None:
            soc_tf = soc_tf.view(-1)
            soc_logp = self.soc_flow.log_prob(soc_tf, cond_soc)

        with torch.no_grad():
            soc_samples = self.soc_flow.sample(cond_soc, num_samples=int(n_mc))
            soc_pred = self._sample_mean_1d(
                samples=soc_samples,
                batch_size=batch_size,
                num_samples=int(n_mc),
                name="SOC",
            )

        soc_pred = soc_pred.view(-1)

        if soc_tf is not None:
            soc_value = soc_tf.detach().view(-1, 1)
        else:
            soc_value = soc_pred.detach().view(-1, 1)

        if self.use_pt:
            cond_soh = torch.cat([z, p_mat, soc_value, x_pt], dim=1)
        else:
            cond_soh = torch.cat([z, p_mat, soc_value], dim=1)

        with torch.no_grad():
            soh_samples = self.soh_flow.sample(cond_soh, num_samples=int(n_mc))
            soh_pred = self._sample_mean_1d(
                samples=soh_samples,
                batch_size=batch_size,
                num_samples=int(n_mc),
                name="SOH",
            )

        soh_pred = soh_pred.view(-1)

        return logits_mat, soc_pred, soc_logp, cond_soc, soh_pred, cond_soh