# Hierarchical Battery Metadata Reconstruction

Python implementation for the manuscript:

**Hierarchical battery metadata reconstruction from short-pulse responses for retired lithium-ion batteries**

This repository provides the code used to reconstruct battery metadata from short-pulse voltage responses of retired lithium-ion batteries.

The reconstructed metadata include:

- material-capacity group;
- state of charge (SOC);
- state of health (SOH).

The proposed framework follows the hierarchical reconstruction order:

```text
Material-capacity classification → SOC estimation → SOH estimation
```

The repository includes the proposed framework, post-training analysis, benchmark comparisons, ablation studies, measurement-sensitivity analyses, smoke tests, and scripts for generating the manuscript figures.

---

## Repository structure

```text
Hierarchical-Battery-Metadata-Reconstruction/
├── ablation/                    # Ablation studies
├── analysis/                    # Further analysis, per-class results, calibration, and error propagation
├── benchmark/                   # Benchmark models and fair/enhanced comparisons
├── data/                        # Battery pulse-response data
├── examples/                    # Dataset preview and smoke tests
├── figures/                     # Main and supplementary figure scripts
├── lib/                         # Local NODE dependency
├── measurement_sensitivity/     # Pulse width, C-rate, polarity, missing-value, and noise analyses
├── proposed_framework/          # Proposed hierarchical framework
├── results/                     # Generated outputs
├── utils/                       # Shared utilities
├── README.md
└── requirements.txt
```

Run all commands from the repository root.

---

## Installation

Create and activate a Python environment:

```bash
conda create -n battery-metadata python=3.11
conda activate battery-metadata
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The code was developed with Python 3.11.

The local `lib/` folder is required by the NODE benchmark and should remain at the repository root.

---

## Data

Place the battery pulse-response dataset under:

```text
data/
```

The expected data structure is described in:

```text
data/README.md
```

Before running the full experiments, preview the dataset:

```bash
python examples/preview_dataset.py
```

---

# Recommended workflow

The recommended execution order is:

```text
1. Install the required packages
2. Check the dataset
3. Run smoke tests
4. Train and evaluate the proposed framework
5. Run further analysis
6. Generate per-class results
7. Run the remaining analysis and comparison experiments
8. Generate the manuscript figures
```

---

## Step 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2. Check the dataset

Run the dataset preview script:

```bash
python examples/preview_dataset.py
```

This confirms that the raw Excel files can be found and loaded correctly.

---

## Step 3. Run smoke tests

Before running the full experiments, use the lightweight smoke tests to check the main code paths, dependencies, data loading, and output directories.

```bash
python examples/smoke_test_proposed_framework.py
python examples/smoke_test_analysis.py
python examples/smoke_test_benchmark_all.py
python examples/smoke_test_ablation.py
python examples/smoke_test_measurement_sensitivity.py
```

Smoke-test outputs are saved under:

```text
results/smoke_test/
```

Passing the smoke tests confirms that the corresponding workflow can run under reduced settings. It does not reproduce the full experimental results.

---

## Step 4. Run the proposed framework

Train and evaluate the proposed hierarchical model:

```bash
python proposed_framework/run_proposed_framework.py
```

The main outputs are saved under:

```text
results/proposed_framework/
```

Typical outputs include:

```text
checkpoints/
logs/
metrics/
cache/
splits/
run_config.json
u41_norm_train_only.npz
target_norm_train_only.npz
```

The proposed framework must be completed before running the post-training analysis scripts.

For background execution on a server:

```bash
mkdir -p logs

nohup python -u proposed_framework/run_proposed_framework.py     > logs/run_proposed_framework.log 2>&1 &
```

View the log:

```bash
tail -f logs/run_proposed_framework.log
```

---

## Step 5. Run further analysis

Generate stable train, validation, and test predictions using the trained proposed-framework checkpoint:

```bash
python analysis/run_further_analysis_proposed.py
```

Main outputs are saved under:

```text
results/proposed_framework/further_analysis/tables/
```

Including:

```text
train_predictions_for_scatter.csv
val_predictions_per_sample.csv
test_predictions_per_sample.csv
proposed_method_summary.csv
further_analysis_inference_config.json
```

These prediction tables are used by the per-class analysis, error-propagation analysis, and several figure scripts.

---

## Step 6. Generate per-class results

After running further analysis, generate performance results for all records and each material-capacity class:

```bash
python analysis/generate_per_class_results.py
```

Default input:

```text
results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv
```

Default output:

```text
results/analysis/per_class/per_class_results.csv
```

The required order is:

```bash
python proposed_framework/run_proposed_framework.py
python analysis/run_further_analysis_proposed.py
python analysis/generate_per_class_results.py
```

---

## Step 7. Run the remaining experiments

After the main proposed-framework results have been generated, run the additional experiments required for the manuscript.

### Additional analysis

#### Error-propagation analysis

```bash
python analysis/error_propagation_analysis.py
```

Outputs:

```text
results/analysis/error_propagation/
```

#### Gaussian calibration baseline

```bash
python analysis/train_calibration_baseline.py
```

Outputs:

```text
results/calibration_baseline/
```

#### Flow-versus-Gaussian probabilistic evaluation

Run this after training the Gaussian baseline:

```bash
python analysis/run_flow_calibration_analysis.py
```

Outputs:

```text
results/analysis/probabilistic_evaluation/
```

---

### Benchmark comparisons

Run all benchmark models:

```bash
python benchmark/run_all_benchmarks.py     --models all     --setting both
```

For a quick workflow test:

```bash
python benchmark/run_all_benchmarks.py     --models all     --setting both     --quick
```

The benchmark models include:

```text
XGBoost
TabNet
NODE
FT-Transformer
```

Benchmark results are saved under:

```text
results/benchmark/
```

The combined summary is saved as:

```text
results/benchmark/benchmark_comparison_summary.csv
```

---

### Ablation studies

Run the required ablation experiments:

```bash
python ablation/input_representation_ablation.py
python ablation/hierarchy_ablation.py
python ablation/hierarchy_order_ablation.py
python ablation/material_conditioning_ablation.py
python ablation/transfer_ablation.py
python ablation/channel_ablation.py
```

These scripts evaluate:

- raw versus structured input;
- independent, partial, and full hierarchy;
- hierarchy order;
- material conditioning;
- information transfer;
- input-channel composition.

Outputs are saved under:

```text
results/ablation/
```

---

### Measurement-sensitivity analyses

Run the measurement-sensitivity experiments:

```bash
python measurement_sensitivity/pulse_width_sensitivity.py
python measurement_sensitivity/c_rate_sensitivity.py
python measurement_sensitivity/pulse_polarity_sensitivity.py
python measurement_sensitivity/input_quality_sensitivity.py
```

These scripts evaluate:

- reduced pulse-width settings;
- reduced C-rate combinations;
- positive-only and negative-only pulse responses;
- missing input features;
- voltage noise.

Outputs are saved under:

```text
results/measurement_sensitivity/
```

---

## Step 8. Generate figures

After the required proposed-framework, analysis, benchmark, ablation, and measurement-sensitivity results have been generated, run the plotting scripts under:

```text
figures/
```

Examples:

```bash
python figures/plot_fig2a_dataset_sankey.py
python figures/plot_fig3a_prediction_scatter.py
python figures/plot_fig4d_error_propagation_waterfall.py
python figures/plot_fig5c_calibration.py
python figures/plot_fig5d_fair_unfair_comparison.py
```

Generated figures are saved under:

```text
results/figures/main/
results/figures/supp/
```

The complete figure-to-script mapping is provided in:

```text
figures/README.md
```

If a figure script reports a missing input file, run the corresponding upstream experiment first.

---

# Complete execution order

A complete workflow is:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check the dataset
python examples/preview_dataset.py

# 3. Run smoke tests
python examples/smoke_test_proposed_framework.py
python examples/smoke_test_analysis.py
python examples/smoke_test_benchmark_all.py
python examples/smoke_test_ablation.py
python examples/smoke_test_measurement_sensitivity.py

# 4. Train the proposed framework
python proposed_framework/run_proposed_framework.py

# 5. Generate further-analysis predictions
python analysis/run_further_analysis_proposed.py

# 6. Generate per-class results
python analysis/generate_per_class_results.py

# 7. Run additional analysis
python analysis/error_propagation_analysis.py
python analysis/train_calibration_baseline.py
python analysis/run_flow_calibration_analysis.py

# 8. Run benchmark models
python benchmark/run_all_benchmarks.py --models all --setting both

# 9. Run ablation studies
python ablation/input_representation_ablation.py
python ablation/hierarchy_ablation.py
python ablation/hierarchy_order_ablation.py
python ablation/material_conditioning_ablation.py
python ablation/transfer_ablation.py
python ablation/channel_ablation.py

# 10. Run measurement-sensitivity analyses
python measurement_sensitivity/pulse_width_sensitivity.py
python measurement_sensitivity/c_rate_sensitivity.py
python measurement_sensitivity/pulse_polarity_sensitivity.py
python measurement_sensitivity/input_quality_sensitivity.py

# 11. Generate figures
python figures/plot_fig3a_prediction_scatter.py
python figures/plot_fig5c_calibration.py
python figures/plot_fig5d_fair_unfair_comparison.py
```

Only run the experiments and figure scripts required for the specific results being reproduced.

---

## Output structure

Most generated files are saved under:

```text
results/
```

Typical subfolders include:

```text
results/
├── ablation/
├── analysis/
├── benchmark/
├── calibration_baseline/
├── figures/
├── measurement_sensitivity/
├── proposed_framework/
└── smoke_test/
```

Depending on the script, outputs may include:

- model checkpoints;
- normalization statistics;
- prediction tables;
- metric summaries;
- intermediate analysis files;
- generated figures.

---

## Folder-level documentation

Detailed instructions are available in:

- [`data/README.md`](data/README.md)
- [`proposed_framework/README.md`](proposed_framework/README.md)
- [`analysis/README.md`](analysis/README.md)
- [`benchmark/README.md`](benchmark/README.md)
- [`ablation/README.md`](ablation/README.md)
- [`measurement_sensitivity/README.md`](measurement_sensitivity/README.md)
- [`figures/README.md`](figures/README.md)
- [`examples/README.md`](examples/README.md)
- [`utils/README.md`](utils/README.md)

---

## Reproducibility notes

- Run scripts from the repository root.
- Keep the raw data under the top-level `data/` folder.
- Run smoke tests before starting the full experiments.
- Use the same battery-ID split across training and downstream analyses.
- Use training-only normalization statistics.
- Run `run_further_analysis_proposed.py` before scripts that depend on per-sample predictions.
- Run `generate_per_class_results.py` only after the test prediction table has been generated.
- Run the Gaussian baseline before the flow-versus-Gaussian probabilistic comparison.
- Generate figures only after their required upstream result files are available.

---

## Code availability

The repository includes code for:

- the proposed hierarchical framework;
- post-training analysis;
- per-class performance analysis;
- probabilistic calibration;
- error propagation;
- benchmark comparisons;
- ablation studies;
- measurement-sensitivity analyses;
- smoke tests;
- main and supplementary figure generation.

---

## Citation

If you use this code or dataset, please cite:

```text
Qiqi Wang, Shengyu Tao*, Chen Liang, Fusen Guo, Ajith Kumar Parlikad, 
Guangmin Zhou*, Min Xie*, Huadong Mo*.
Hierarchical retired battery metadata reconstruction from short-pulse-enabled electrochemical fingerprints

```

`*` Corresponding author.

A formal citation will be added after publication.

---

## Licence

This repository is released under the MIT License.

See the `LICENSE` file for details.
