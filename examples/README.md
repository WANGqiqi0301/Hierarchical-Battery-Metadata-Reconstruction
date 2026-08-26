# Examples and Smoke Tests

This folder contains lightweight example scripts and smoke tests for checking whether the main components of the repository can run correctly.

These scripts are intended for quick verification only. They use reduced settings such as fewer epochs, smaller models, or fewer pulse-width configurations. They are not used to reproduce the full experimental results reported in the manuscript.

## Scripts

| Script | Purpose |
|---|---|
| `preview_dataset.py` | Preview whether the dataset can be loaded correctly. |
| `smoke_test_proposed_framework.py` | Test the proposed framework pipeline, including data loading, training, checkpoint saving, metrics saving, and resume training. |
| `smoke_test_benchmark_all.py` | Test all benchmark models under fair and enhanced settings. |
| `smoke_test_analysis.py` | Test analysis scripts, including further analysis, error propagation, and calibration baseline. |
| `smoke_test_ablation.py` | Test ablation-study scripts with lightweight settings, including channel, hierarchy, hierarchy order, input representation, transfer, and material-conditioning ablations. |
| `smoke_test_measurement_sensitivity.py` | Test measurement-sensitivity scripts with lightweight settings. |

## Usage

Run scripts from the project root:

```bash
python examples/preview_dataset.py
python examples/smoke_test_proposed_framework.py
python examples/smoke_test_benchmark_all.py
python examples/smoke_test_analysis.py
python examples/smoke_test_ablation.py
python examples/smoke_test_measurement_sensitivity.py