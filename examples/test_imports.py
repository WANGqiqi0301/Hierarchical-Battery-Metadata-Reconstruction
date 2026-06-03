# examples/test_imports.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    print("[TEST] Project root:", PROJECT_ROOT)

    from utils.cache import ensure_dir, load_or_build_cache, drop_nan_inf_rows
    from utils.data_loader import LoadConfig, load_pulsebat_classification, preview
    from utils.metrics import rmse, mae, ape, mape, medape
    from utils.seed import set_random_seed

    from proposed_framework.data.build_dataset import (
        load_one_soc_one_pulse,
        build_train_mix_soc_mix_pt,
        build_test_random_mix_pt,
        pick_test_ids,
        apply_id_split,
    )
    from proposed_framework.data.feature_builder import (
        build_three_channel_representation,
        build_3ch_5x8_from_u41,
    )
    from proposed_framework.data.pulse_dataset import HierPulseDataset

    from proposed_framework.models.conditional_flow import Conditional1DFlow
    from proposed_framework.models.encoder import ResBlock, MicroResNetEncoder2D3Ch
    from proposed_framework.models.hierarchical_model import Hier3HeadModel

    from proposed_framework.training.losses import (
        bin_index_from_edges,
        soc_bin_index_from_edges,
        heteroscedastic_nll,
        heteroscedastic_nll_per_sample,
    )
    from proposed_framework.training.trainer import train_one_epoch
    from proposed_framework.training.evaluator import (
        eval_one_epoch,
        inverse_targets,
        soc_z_to_raw_tensor,
        soh_z_to_raw_tensor,
    )

    from proposed_framework.run_proposed_framework import run_experiment

    print("[PASS] All imports passed.")


if __name__ == "__main__":
    main()