# examples/preview_dataset.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

# Add project root directory to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import LoadConfig, load_pulsebat_classification, preview


def main() -> None:
    cfg = LoadConfig(
        data_root=PROJECT_ROOT / "data" ,
        soc="TEST_RANDOM",
        pulse_width_ms=5000,
        u_start=1,
        u_end=41,
        drop_first_21_only_class=True,
        include_soc_in_X=False,
        verbose=True,
    )

    X, y, meta = load_pulsebat_classification(cfg)
    preview(X, y, meta, n=8)


if __name__ == "__main__":
    main()