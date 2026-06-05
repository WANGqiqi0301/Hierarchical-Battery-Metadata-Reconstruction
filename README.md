# Hierarchical Battery Metadata Reconstruction

Python implementation for the manuscript:

**Hierarchical battery metadata reconstruction from short-pulse responses for retired lithium-ion batteries**

This repository provides code for reconstructing key battery metadata from short-pulse voltage responses, including material-capacity group, state of charge (SOC), and state of health (SOH). The proposed framework follows a hierarchical reconstruction order from **material-capacity classification** to **SOC estimation** and finally **SOH estimation**.

---

## 1. Project Overview

Retired lithium-ion batteries require rapid diagnostic metadata reconstruction before reuse, remanufacturing, and recycling. Conventional diagnostics are often time-consuming, invasive, or difficult to apply at scale.

This project implements a hierarchical battery metadata reconstruction framework using short-pulse voltage responses. The framework converts pulse-response measurements into structured diagnostic features and uses probabilistic learning to infer:

* Material-capacity group
* State of charge (SOC)
* State of health (SOH)

The repository includes code for:

* Proposed hierarchical reconstruction framework
* Benchmark comparisons with machine-learning and deep-learning baselines
* Measurement sensitivity analysis
* Ablation studies
* Error-propagation analysis
* Manuscript figure generation
* Smoke tests and example scripts

---

## 2. Repository Structure

```text
Hierarchical-Battery-Metadata-Reconstruction/
├── ablation/                    # Ablation studies
├── analysis/                    # Additional analysis scripts
├── benchmark/                   # Benchmark models and fair/enhanced comparisons
├── data/                        # Battery pulse-response data
├── examples/                    # Smoke tests and lightweight examples
├── figures/                     # Manuscript figure-generation scripts
├── measurement_sensitivity/     # Pulse width, C-rate, polarity, missing/noise sensitivity
├── proposed_framework/          # Proposed hierarchical reconstruction framework
├── results/                     # Generated experiment outputs
├── utils/                       # Shared utility functions
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 3. Installation

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
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch xgboost openpyxl tqdm joblib
```

Additional packages may be required for TabNet, NODE, FT-Transformer, and flow-based uncertainty models.

---

## 4. Data Preparation

Place the battery pulse-response data under:

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

After downloading, make sure the folder structure matches the expected format above.

---

## 5. Quick Start

### 5.1 Check imports

```bash
python examples/test_imports.py
```

### 5.2 Preview dataset loading

```bash
python examples/preview_dataset.py
```

### 5.3 Run proposed framework smoke test

```bash
python examples/smoke_test_proposed_framework.py
```

### 5.4 Run proposed framework

```bash
python proposed_framework/run_proposed_framework.py
```

### 5.5 Run all benchmark models

```bash
python benchmark/run_all_benchmarks.py
```

---

## 6. Main Workflow

The recommended workflow is:

```bash
# 1. Check whether the environment and imports are correct
python examples/test_imports.py

# 2. Preview whether the dataset can be loaded correctly
python examples/preview_dataset.py

# 3. Run the proposed hierarchical reconstruction framework
python proposed_framework/run_proposed_framework.py

# 4. Run benchmark comparisons
python benchmark/run_all_benchmarks.py

# 5. Run ablation studies
python ablation/input_representation_ablation.py
python ablation/hierarchy_ablation.py
python ablation/hierarchy_order_ablation.py
python ablation/channel_ablation.py
python ablation/transfer_ablation.py

# 6. Run measurement-sensitivity analyses
python measurement_sensitivity/pulse_width_sensitivity.py
python measurement_sensitivity/c_rate_sensitivity.py
python measurement_sensitivity/input_quality_sensitivity.py
python measurement_sensitivity/pulse_polarity_sensitivity.py

# 7. Run additional analysis
python analysis/error_propagation_analysis.py
```

---

## 7. Folder Description

### 7.1 `proposed_framework/`

This folder contains the implementation of the proposed hierarchical battery metadata reconstruction framework.

Main entry point:

```bash
python proposed_framework/run_proposed_framework.py
```

This module is used to reproduce the main proposed method, including material-capacity classification, SOC estimation, and SOH estimation.

---

### 7.2 `benchmark/`

This folder contains benchmark models used for comparison with the proposed framework.

Implemented benchmark models include:

* XGBoost
* TabNet
* NODE
* FT-Transformer

The benchmark module supports two settings:

1. **Fair setting**

   All benchmark models use the same base input features and their own upstream predictions.

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

---

### 7.3 `ablation/`

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

---

### 7.4 `measurement_sensitivity/`

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

---

### 7.5 `analysis/`

This folder contains additional analysis scripts.

Currently included:

* Error-propagation analysis

Example command:

```bash
python analysis/error_propagation_analysis.py
```

---

### 7.6 `figures/`

This folder contains manuscript figure-generation scripts.

The plotting scripts are organized by figure number and panel. Most scripts read intermediate result files from `results/` and save publication-style figures to figure-specific output folders.

---

### 7.7 `utils/`

This folder contains shared utility modules, including:

* Data loading
* Cache handling
* Evaluation metrics
* Random-seed control

---

### 7.8 `examples/`

This folder contains smoke tests and lightweight examples for checking whether the code environment, data loader, model modules, and benchmark pipeline work correctly.

Example commands:

```bash
python examples/test_imports.py
python examples/preview_dataset.py
python examples/smoke_test_proposed_framework.py
```

---

## 8. Figure Reproduction

All figure-generation scripts are organized under:

```text
figures/
```

The scripts are named according to the corresponding manuscript figure panels. Most plotting scripts read intermediate result files from `results/` and save publication-style figures to figure-specific output folders. Therefore, the corresponding experiment scripts should be run first before generating the final plots.

---

### 8.1 Dataset and feature-description figures

```text
Fig. 2a  -> figures/plot_fig2a_dataset_sankey.py
Fig. 2b  -> figures/plot_fig2b_soc_soh_distribution.py
Fig. 2c  -> figures/plot_fig2c_soh_distribution_by_material.py
Fig. 2d  -> figures/plot_fig2d_soc_soh_joint_distribution.py
Fig. 2e  -> figures/plot_fig2e_ocv_u1_curves.py
Fig. 2e  -> figures/plot_fig2e_u1_u41_distribution_soc20.py
Fig. 2e  -> figures/plot_fig2e_delta_u_distribution_soc20.py
Fig. 2f  -> figures/plot_fig2f_u_evolution.py
Fig. 2f  -> figures/plot_fig2f_delta_u_evolution.py
Fig. 2g  -> figures/plot_fig2g_pulse_response_feature_sequence.py
Fig. 2h  -> figures/plot_fig2h_delta_u_correlation_contour.py
```

---

### 8.2 Main reconstruction-performance figures

```text
Fig. 3a  -> figures/plot_fig3a_prediction_scatter.py
Fig. 3b  -> figures/plot_fig3b_material_confusion.py
Fig. 3b  -> figures/plot_fig3b_material_soc_error.py
Fig. 3b  -> figures/plot_fig3b_material_soh_error.py
Fig. 3c  -> figures/plot_fig3c_pulse_soc_rmse.py
Fig. 3c  -> figures/plot_fig3c_pulse_soh_rmse.py
Fig. 3d  -> figures/plot_fig3d_soc_bin_median_ape.py
Fig. 3d  -> figures/plot_fig3d_soh_bin_median_ape.py
Fig. 3e  -> figures/plot_fig3e_tsne_material_raw_vs_embedded.py
Fig. 3f  -> figures/plot_fig3f_error_cascading_grid.py
```

---

### 8.3 Sensitivity and robustness figures

```text
Fig. 4c  -> figures/plot_fig4c_soc_error_distribution.py
Fig. 4d  -> figures/plot_fig4d_error_propagation_waterfall.py
Fig. 4e  -> figures/plot_fig4e_pulse_matrix.py
Fig. 4e  -> figures/plot_fig4e_pulse_width_plot.py
Fig. 4f  -> figures/plot_fig4f_crate_combination_map.py
Fig. 4f  -> figures/plot_fig4f_crate_combination.py
Fig. 4g  -> figures/plot_fig4g_drop_robustness.py
Fig. 4h  -> figures/plot_fig4h_noise_ocv_effect.py
Fig. 4i  -> figures/plot_fig4i_pulse_polarity_effect.py
```

---

### 8.4 Ablation, calibration, and benchmark figures

```text
Fig. 5c  -> figures/plot_fig5c_calibration.py
Fig. 5d  -> figures/plot_fig5d_fair_unfair_comparison.py
Fig. 5e  -> figures/plot_fig5e_fewer_channel.py
Fig. 5f  -> figures/plot_fig5f_hierarchy_order_bubble.py
Fig. 5g  -> figures/plot_fig5g_material_conditioning.py
```

---

### 8.5 Recommended workflow for reproducing figures

```bash
# 1. Run the main proposed framework
python proposed_framework/run_proposed_framework.py

# 2. Run benchmark comparisons
python benchmark/run_all_benchmarks.py

# 3. Run ablation studies
python ablation/input_representation_ablation.py
python ablation/hierarchy_ablation.py
python ablation/hierarchy_order_ablation.py
python ablation/channel_ablation.py
python ablation/transfer_ablation.py

# 4. Run measurement-sensitivity analyses
python measurement_sensitivity/pulse_width_sensitivity.py
python measurement_sensitivity/c_rate_sensitivity.py
python measurement_sensitivity/input_quality_sensitivity.py
python measurement_sensitivity/pulse_polarity_sensitivity.py

# 5. Generate representative manuscript figures
python figures/plot_fig2a_dataset_sankey.py
python figures/plot_fig3a_prediction_scatter.py
python figures/plot_fig4e_pulse_width_plot.py
python figures/plot_fig5d_fair_unfair_comparison.py
```

Not every figure script is independent. Some scripts require result files produced by the corresponding experiment modules under `results/`. If a plotting script reports that an input file is missing, first run the associated experiment script and then rerun the plotting script.

---

## 9. Output Files

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

Figure-generation scripts may save figures to figure-specific folders or output paths defined in each plotting script.

---

## 10. Reproducibility Notes

Random seeds are fixed where applicable.

The benchmark pipeline uses ID-based train/test splitting to reduce data leakage between cells. Feature preprocessing uses training-set statistics only, including train-only imputation and scaling.

When reproducing results, please make sure that:

* The same dataset version is used
* The same pulse-width list is used
* Cached files from earlier experiments are removed if the data-loading logic has changed
* Fair and enhanced benchmark settings are compared separately
* Experiment scripts are run before figure-generation scripts that depend on `results/`

To rebuild benchmark cache, use:

```bash
python benchmark/xgboost_benchmark.py --setting fair --no-cache
```

---

## 11. Requirements File

This repository includes a `requirements.txt` file for dependency installation.

If a clean requirements file needs to be regenerated from the current environment, a helper script such as `generate_clean_requirements.py` can be used to avoid local Anaconda paths in the exported dependency list.

The expected package format is:

```text
numpy==1.26.4
pandas==2.1.4
torch==2.2.2
xgboost==2.0.3
```

rather than local environment paths such as:

```text
numpy @ file:///...
```

---

## 12. Citation

If you use this code or dataset, please cite:

```text
[CITATION TO BE ADDED AFTER PUBLICATION]
```

---

## 13. License

This repository is released under the MIT License. See `LICENSE` for details.
