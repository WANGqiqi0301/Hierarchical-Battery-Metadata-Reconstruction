# Hierarchical Battery Metadata Reconstruction

Python implementation for the manuscript:

**Hierarchical battery metadata reconstruction from short-pulse responses for retired lithium-ion batteries**

This repository provides the code for reconstructing key battery metadata from short-pulse voltage responses, including material-capacity group, state of charge (SOC), and state of health (SOH). The proposed framework follows a hierarchical diagnostic order from material-capacity classification to SOC estimation and finally SOH estimation.

## 1. Project Overview

Retired lithium-ion batteries require rapid diagnostic metadata reconstruction before reuse, remanufacturing, and recycling. Conventional diagnostics are often time-consuming, invasive, or difficult to apply at scale.

This project develops a hierarchical battery passport reconstruction framework using short-pulse voltage responses. The framework uses structured voltage-response features and probabilistic learning to infer:

* Material-capacity group
* State of charge (SOC)
* State of health (SOH)

The repository includes code for:

* Proposed hierarchical reconstruction framework
* Benchmark comparisons with machine-learning and deep-learning baselines
* Measurement sensitivity analysis
* Ablation studies
* Error-propagation analysis
* Smoke tests and example scripts

## 2. Repository Structure

```text
Hierarchical-Battery-Metadata-Reconstruction/
├── ablation/
│   ├── channel_ablation.py
│   ├── hierarchy_ablation.py
│   ├── hierarchy_order_ablation.py
│   ├── input_representation_ablation.py
│   └── transfer_ablation.py
│
├── analysis/
│   └── error_propagation_analysis.py
│
├── benchmark/
│   ├── common.py
│   ├── enhanced_inputs.py
│   ├── xgboost_benchmark.py
│   ├── tabnet_benchmark.py
│   ├── node_benchmark.py
│   ├── ft_transformer_benchmark.py
│   └── run_all_benchmarks.py
│
├── examples/
│   ├── preview_dataset.py
│   ├── smoke_test_benchmark_all.py
│   ├── smoke_test_benchmark_common.py
│   ├── smoke_test_proposed_framework.py
│   └── test_imports.py
│
├── measurement_sensitivity/
│   ├── c_rate_sensitivity.py
│   ├── input_quality_sensitivity.py
│   ├── pulse_polarity_sensitivity.py
│   └── pulse_width_sensitivity.py
│
├── proposed_framework/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── run_proposed_framework.py
│
├── utils/
│   ├── cache.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── seed.py
│
├── data/
├── results/
├── i10_normalization_flow.py
├── README.md
├── LICENSE
└── .gitignore
```

## 3. Folder Description

### `proposed_framework/`

This folder contains the implementation of the proposed hierarchical battery metadata reconstruction framework.

Main entry point:

```bash
python proposed_framework/run_proposed_framework.py
```

This module is used to reproduce the main proposed method, including material-capacity classification, SOC estimation, and SOH estimation.

### `benchmark/`

This folder contains benchmark models used for comparison with the proposed framework.

Implemented benchmark models include:

* XGBoost
* TabNet
* NODE
* FT-Transformer

The benchmark module supports two settings:

1. **Fair setting**

   All benchmark models use the same base input features.

2. **Enhanced setting**

   Downstream SOC and SOH regressors are provided with controlled upstream material and SOC inputs at error levels comparable to those of the proposed framework. This setting is used to evaluate whether benchmark performance is limited by upstream error propagation.

Example commands:

```bash
python benchmark/xgboost_benchmark.py --setting both
python benchmark/tabnet_benchmark.py --setting both
python benchmark/node_benchmark.py --setting both
python benchmark/ft_transformer_benchmark.py --setting both
python benchmark/run_all_benchmarks.py
```

For a quick test:

```bash
python benchmark/xgboost_benchmark.py --setting both --quick
```

### `ablation/`

This folder contains ablation experiments used to evaluate the contribution of different components of the framework.

Included experiments:

* Channel ablation
* Hierarchy ablation
* Hierarchy-order ablation
* Input-representation ablation
* Transfer ablation

Example commands:

```bash
python ablation/channel_ablation.py
python ablation/hierarchy_ablation.py
python ablation/hierarchy_order_ablation.py
python ablation/input_representation_ablation.py
python ablation/transfer_ablation.py
```

### `measurement_sensitivity/`

This folder contains experiments for evaluating the influence of measurement configuration and input quality.

Included experiments:

* Pulse-width sensitivity
* C-rate sensitivity
* Pulse-polarity sensitivity
* Input-quality sensitivity, including missing features and voltage-noise perturbation

Example commands:

```bash
python measurement_sensitivity/pulse_width_sensitivity.py
python measurement_sensitivity/c_rate_sensitivity.py
python measurement_sensitivity/input_quality_sensitivity.py
python measurement_sensitivity/pulse_polarity_sensitivity.py
```

### `analysis/`

This folder contains additional analysis scripts.

Currently included:

* Error-propagation analysis

Example command:

```bash
python analysis/error_propagation_analysis.py
```

### `utils/`

This folder contains shared utility modules, including:

* Data loading
* Cache handling
* Evaluation metrics
* Random-seed control

### `examples/`

This folder contains smoke tests and lightweight examples for checking whether the code environment, data loader, model modules, and benchmark pipeline work correctly.

Example commands:

```bash
python examples/test_imports.py
python examples/test_data_loading.py
python examples/test_feature_builder.py
python examples/test_model_forward.py
python examples/smoke_test_proposed_framework.py
python examples/smoke_test_benchmark_common.py
```

## 4. Data Preparation

The full experimental dataset is not included directly in this repository.

To run the code, place the battery pulse-response data under:

```text
data/
```

Expected data folders include:

```text
data/
├── 2.1Ah NMC/
├── 10Ah LMO/
├── 15Ah NMC/
├── 21Ah NMC/
├── 24Ah LMO/
├── 25Ah LMO/
├── 26Ah LMO/
├── 35Ah LFP/
└── 68Ah LFP/
```

If the dataset is released through an external repository or data archive, please download it from:

```text
[DATASET LINK TO BE ADDED]
```

After downloading, make sure the data folder structure matches the expected format above.

## 5. Installation

Create a new Python environment:

```bash
conda create -n battery-metadata python=3.11
conda activate battery-metadata
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not yet been prepared, the main dependencies include:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch xgboost openpyxl tqdm
```

Additional packages may be required for TabNet, NODE, or FT-Transformer benchmarks.

## 6. Quick Start

### 6.1 Check imports

```bash
python examples/test_imports.py
```

### 6.2 Preview dataset loading

```bash
python examples/preview_dataset.py
```

### 6.3 Run proposed framework smoke test

```bash
python examples/smoke_test_proposed_framework.py
```

### 6.4 Run proposed framework

```bash
python proposed_framework/run_proposed_framework.py
```

### 6.5 Run all benchmark models

```bash
python benchmark/run_all_benchmarks.py
```

## 7. Main Experiment Commands

### Proposed framework

```bash
python proposed_framework/run_proposed_framework.py
```

### Benchmark comparison

```bash
python benchmark/run_all_benchmarks.py
```

### Measurement sensitivity

```bash
python measurement_sensitivity/pulse_width_sensitivity.py
python measurement_sensitivity/c_rate_sensitivity.py
python measurement_sensitivity/input_quality_sensitivity.py
python measurement_sensitivity/pulse_polarity_sensitivity.py
```

### Ablation studies

```bash
python ablation/input_representation_ablation.py
python ablation/hierarchy_ablation.py
python ablation/hierarchy_order_ablation.py
python ablation/channel_ablation.py
python ablation/transfer_ablation.py
```

### Error-propagation analysis

```bash
python analysis/error_propagation_analysis.py
```

## 8. Output Files

Most scripts save results under:

```text
results/
```

Typical output files include:

```text
predictions.csv
summary.json
report.txt
```

Benchmark results are usually saved under:

```text
results/benchmark/
```

For example:

```text
results/benchmark/xgboost/fair/
results/benchmark/xgboost/enhanced/
```

## 9. Reproducibility Notes

Random seeds are fixed where applicable.

The benchmark pipeline uses ID-based train/test splitting to reduce data leakage between cells. Feature preprocessing uses training-set statistics only, including train-only imputation and scaling.

When reproducing results, please make sure that:

* The same dataset version is used
* The same pulse-width list is used
* Cached files from earlier experiments are removed if the data-loading logic has changed
* Fair and enhanced benchmark settings are compared separately

To rebuild benchmark cache, use:

```bash
python benchmark/xgboost_benchmark.py --setting fair --no-cache
```

## 10. Figure and Table Reproduction

The figure-generation scripts can be placed under:

```text
figures/
```

Recommended mapping:

```text
Main Fig. 4e  -> figures/plot_pulse_width_sensitivity.py
Main Fig. 4f  -> figures/plot_c_rate_sensitivity.py
Main Fig. 4g  -> figures/plot_input_quality_sensitivity.py
Main Fig. 4h  -> figures/plot_input_quality_sensitivity.py
Main Fig. 4i  -> figures/plot_pulse_polarity_sensitivity.py
Main Fig. 5a  -> figures/plot_input_representation_ablation.py
Main Fig. 5d  -> figures/plot_benchmark_comparison.py
Main Fig. 5e  -> figures/plot_channel_ablation.py
Main Fig. 5f  -> figures/plot_hierarchy_order_ablation.py
```

A detailed figure-to-code mapping can be added later in:

```text
docs/figure_code_mapping.md
```

## 11. Citation

If you use this code or dataset, please cite:

```text
[CITATION TO BE ADDED AFTER PUBLICATION]
```

## 12. License

This repository is released under the MIT License. See `LICENSE` for details.
