# Hierarchical Battery Metadata Reconstruction

Python implementation for the manuscript:

**Hierarchical battery metadata reconstruction from short-pulse responses for retired lithium-ion batteries**

This repository provides the code used to reconstruct battery metadata from short-pulse voltage responses of retired lithium-ion batteries. The reconstructed metadata include:

* material-capacity group;
* state of charge (SOC);
* state of health (SOH).

The proposed method follows a hierarchical reconstruction strategy in which material-capacity information is first inferred, then used to support SOC estimation, and finally propagated to SOH estimation.

This repository is organised to support the main proposed framework, benchmark comparisons, ablation studies, measurement-sensitivity analyses, further analyses, and manuscript figure generation.

---

## Repository status

The code is provided to support transparency, reproducibility, and reuse of the computational workflow described in the manuscript. Minor updates may be made before final publication, including citation information, licence information, and the public data archive link.

---

## Repository structure

```text
Hierarchical-Battery-Metadata-Reconstruction/
├── ablation/                    # Ablation studies for model design choices
├── analysis/                    # Further analysis, calibration, and error-propagation analysis
├── benchmark/                   # Benchmark models and fair/enhanced comparison settings
├── data/                        # Battery pulse-response data
├── examples/                    # Smoke tests and lightweight usage examples
├── figures/                     # Scripts for generating manuscript and supplementary figures
├── measurement_sensitivity/     # Pulse width, C-rate, polarity, missing-feature, and noise analyses
├── proposed_framework/          # Proposed hierarchical metadata reconstruction framework
├── results/                     # Generated outputs, prediction files, metrics, and figures
├── utils/                       # Shared utility functions
├── .gitignore
├── README.md
└── requirements.txt
```

Each major folder contains, or is intended to contain, a local `README.md` file with folder-specific instructions.

---

## Main components

### Proposed framework

The proposed hierarchical framework is implemented in:

```text
proposed_framework/
```

The main entry point is:

```bash
python proposed_framework/run_proposed_framework.py
```

This script trains and evaluates the hierarchical metadata reconstruction framework, including material-capacity classification, SOC estimation, and SOH estimation.

---

### Benchmark models

Benchmark models are implemented in:

```text
benchmark/
```

The benchmark folder includes implementations for:

* XGBoost;
* TabNet;
* NODE;
* FT-Transformer;
* fair and enhanced-input comparison settings.

The main benchmark runner is:

```bash
python benchmark/run_all_benchmarks.py
```

---

### Ablation studies

Ablation studies are implemented in:

```text
ablation/
```

The ablation scripts evaluate the influence of different design choices, including:

* input representation;
* hierarchical reconstruction structure;
* hierarchy order;
* information transfer strategy;
* input channel composition.

Representative scripts include:

```bash
python ablation/input_representation_ablation.py
python ablation/hierarchy_ablation.py
python ablation/hierarchy_order_ablation.py
python ablation/transfer_ablation.py
python ablation/channel_ablation.py
```

---

### Measurement-sensitivity analyses

Measurement-sensitivity analyses are implemented in:

```text
measurement_sensitivity/
```

These scripts evaluate the robustness of the proposed framework under different diagnostic measurement settings, including:

* pulse-width selection;
* C-rate selection;
* pulse polarity;
* missing input features;
* voltage noise.

Representative scripts include:

```bash
python measurement_sensitivity/pulse_width_sensitivity.py
python measurement_sensitivity/c_rate_sensitivity.py
python measurement_sensitivity/pulse_polarity_sensitivity.py
python measurement_sensitivity/input_quality_sensitivity.py
```

---

### Further analysis and calibration

Additional analyses are implemented in:

```text
analysis/
```

This folder includes scripts for:

* post-processing proposed-framework predictions;
* calibration baseline training;
* error-propagation analysis;
* generation of analysis tables used by figure scripts.

Representative scripts include:

```bash
python analysis/run_further_analysis_proposed.py
python analysis/train_calibration_baseline.py
python analysis/error_propagation_analysis.py
```

---

### Manuscript figures

Figure-generation scripts are implemented in:

```text
figures/
```

The folder contains scripts for generating main-text and supplementary figures. Example scripts include:

```bash
python figures/plot_fig3a_prediction_scatter.py
python figures/plot_fig3b_material_confusion.py
python figures/plot_fig3e_tsne_material_raw_vs_embedded.py
python figures/plot_fig4d_error_propagation_waterfall.py
python figures/plot_fig5c_calibration.py
python figures/plot_fig5d_fair_unfair_comparison.py
python figures/plot_fig5f_hierarchy_order_bubble.py
```

Some figure scripts require result files generated by the proposed framework, benchmark scripts, ablation scripts, or analysis scripts. If a plotting script reports a missing input file, first run the corresponding upstream experiment or analysis script.

Generated figures are saved under:

```text
results/figures/
```

---

## Data availability

The battery pulse-response dataset should be placed under:

```text
data/
```

The expected dataset organisation is described in:

```text
data/README.md
```

The data folder is expected to contain separate subfolders for different chemistry-capacity groups.

The public data archive link will be added after data deposition:

```text
[DATASET LINK TO BE ADDED]
```

Until the public archive is released, users should follow the folder structure described in `data/README.md` and place the required data files under `data/`.

---

## Installation

Create a new Python environment:

```bash
conda create -n battery-metadata python=3.11
conda activate battery-metadata
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The code was developed and tested with Python 3.11. Major dependencies include:

* PyTorch;
* scikit-learn;
* XGBoost;
* pandas;
* NumPy;
* matplotlib.

A complete package list is provided in:

```text
requirements.txt
```

---

## Quick start

After installing the dependencies and placing the dataset under `data/`, the proposed framework can be run with:

```bash
python proposed_framework/run_proposed_framework.py
```

A lightweight check of the repository setup can be performed using the example and smoke-test scripts:

```bash
python examples/test_imports.py
python examples/test_data_loading.py
python examples/test_model_forward.py
python examples/smoke_test_proposed_framework.py
```

---

## Minimal reproducibility workflow

A typical workflow for reproducing the main computational results is:

```bash
# 1. Run the proposed hierarchical reconstruction framework
python proposed_framework/run_proposed_framework.py

# 2. Post-process proposed-framework outputs
python analysis/run_further_analysis_proposed.py

# 3. Train or evaluate calibration baselines
python analysis/train_calibration_baseline.py

# 4. Run benchmark comparisons
python benchmark/run_all_benchmarks.py

# 5. Run ablation studies
python ablation/input_representation_ablation.py
python ablation/hierarchy_ablation.py
python ablation/hierarchy_order_ablation.py
python ablation/transfer_ablation.py
python ablation/channel_ablation.py

# 6. Run measurement-sensitivity analyses
python measurement_sensitivity/pulse_width_sensitivity.py
python measurement_sensitivity/c_rate_sensitivity.py
python measurement_sensitivity/pulse_polarity_sensitivity.py
python measurement_sensitivity/input_quality_sensitivity.py

# 7. Generate representative manuscript figures
python figures/plot_fig3a_prediction_scatter.py
python figures/plot_fig5c_calibration.py
python figures/plot_fig5d_fair_unfair_comparison.py
```

The exact figure-to-code mapping is provided in:

```text
figures/README.md
```

---

## Output files

Most scripts save generated outputs under:

```text
results/
```

The `results/` folder may contain:

```text
results/
├── ablation/                    # Outputs from ablation studies
├── analysis/                    # Further-analysis outputs
├── benchmark/                   # Benchmark results
├── calibration_baseline/        # Calibration baseline outputs
├── figures/                     # Generated manuscript and supplementary figures
├── measurement_sensitivity/     # Sensitivity-analysis outputs
├── proposed_framework/          # Proposed-framework checkpoints, predictions, and summaries
└── proposed_method/             # Additional proposed-method outputs
```

Depending on the script, outputs may include model checkpoints, prediction tables, summary metrics, intermediate analysis files, and generated figures.

---

## Folder-level documentation

Detailed instructions are provided in the corresponding folders:

* [`data/README.md`](data/README.md): expected dataset structure;
* [`proposed_framework/README.md`](proposed_framework/README.md): proposed hierarchical reconstruction framework;
* [`benchmark/README.md`](benchmark/README.md): benchmark models and comparison settings;
* [`ablation/README.md`](ablation/README.md): ablation studies;
* [`measurement_sensitivity/README.md`](measurement_sensitivity/README.md): measurement-sensitivity analyses;
* [`analysis/README.md`](analysis/README.md): further analysis, calibration, and error propagation;
* [`figures/README.md`](figures/README.md): manuscript figure-generation scripts;
* [`examples/README.md`](examples/README.md): smoke tests and lightweight examples.

---

## Reproducibility notes

The scripts are designed to be run from the repository root:

```text
Hierarchical-Battery-Metadata-Reconstruction/
```

For example:

```bash
python proposed_framework/run_proposed_framework.py
```

Relative paths are used where possible so that the repository can be moved or cloned to a different location without editing absolute paths.

Some computational results depend on trained model checkpoints and intermediate result files. If a downstream analysis or figure-generation script reports a missing file, run the corresponding upstream training, benchmark, ablation, or sensitivity-analysis script first.

---

## Code availability

All scripts required for the proposed framework, benchmark comparisons, ablation studies, measurement-sensitivity analyses, further analyses, and manuscript figure generation are provided in this repository.

---


## Citation

If you use this code or dataset, please cite:

```text
WANG Qiqi, TAO Shengyu, MO Huadong*. 
Hierarchical battery metadata reconstruction from short-pulse responses for retired lithium-ion batteries. 
Manuscript prepared for submission to Nature Communications.
```

`*` Corresponding author.

A formal citation will be added after publication.


---

## Licence

Licence information will be added before public release.
