# examples/test_model_forward.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from proposed_framework.models.hierarchical_model import Hier3HeadModel


def main() -> None:
    torch.manual_seed(42)

    batch_size = 4
    num_classes = 8

    model = Hier3HeadModel(
        num_classes=num_classes,
        width=16,
        blocks=1,
        drop2d=0.0,
        use_pt_as_feature=True,
        soc_hidden=32,
        soh_hidden=32,
        head_dropout=0.1,
        flow_layers=2,
        flow_bins=8,
        flow_tail_bound=3.0,
    )

    model.eval()

    x_img = torch.randn(batch_size, 3, 5, 8)
    x_pt = torch.randn(batch_size, 1)
    soc_tf = torch.randn(batch_size)

    with torch.no_grad():
        logits_mat, soc_pred, soc_logp, cond_soc, soh_pred, cond_soh = model(
            x_img=x_img,
            x_pt=x_pt,
            soc_tf=soc_tf,
            n_mc=4,
        )

    print("[TEST] logits_mat:", logits_mat.shape)
    print("[TEST] soc_pred:", soc_pred.shape)
    print("[TEST] soc_logp:", soc_logp.shape)
    print("[TEST] cond_soc:", cond_soc.shape)
    print("[TEST] soh_pred:", soh_pred.shape)
    print("[TEST] cond_soh:", cond_soh.shape)

    assert logits_mat.shape == (batch_size, num_classes)
    assert soc_pred.shape == (batch_size,)
    assert soc_logp.shape == (batch_size,)
    assert soh_pred.shape == (batch_size,)

    expected_cond_soc_dim = 16 + num_classes + 1
    expected_cond_soh_dim = 16 + num_classes + 1 + 1

    assert cond_soc.shape == (batch_size, expected_cond_soc_dim)
    assert cond_soh.shape == (batch_size, expected_cond_soh_dim)

    with torch.no_grad():
        logits_mat2, soc_pred2, soc_logp2, cond_soc2, soh_pred2, cond_soh2 = model(
            x_img=x_img,
            x_pt=x_pt,
            soc_tf=None,
            n_mc=4,
        )

    assert soc_logp2 is None
    assert logits_mat2.shape == (batch_size, num_classes)
    assert soc_pred2.shape == (batch_size,)
    assert soh_pred2.shape == (batch_size,)

    print("[PASS] Model forward test passed.")


if __name__ == "__main__":
    main()