# Analysis

This folder contains post-training analysis scripts for the hierarchical battery passport reconstruction framework.

The scripts in this folder are **not used to train the main model**. Instead, they are used after training to reload trained checkpoints, run standardized inference, reduce stochastic variation from conditional-flow sampling, and generate detailed prediction tables and diagnostic results required by the manuscript figures and supplementary analyses.

## Purpose of `run_further_analysis_proposed.py`

The main training script focuses on model optimization and reports final aggregate metrics, such as material-capacity classification accuracy, SOC error and SOH error. However, these outputs are not sufficient for manuscript-level analysis.

A separate further-analysis script is needed for two main reasons.

First, SOC and SOH reconstruction in the proposed framework is based on conditional-flow estimators. Unlike deterministic regression heads, conditional-flow inference involves Monte Carlo sampling. Therefore, SOC and SOH predictions may show stochastic variation if only a small number of samples is used. The further-analysis script reloads the fixed trained checkpoint and performs controlled inference so that the prediction results are more stable and consistent with the final evaluation protocol.

Second, many manuscript analyses require outputs that are not produced by the main training script. The training code mainly reports aggregate metrics, whereas the figures and supplementary analyses require per-sample predictions, predicted material labels, train/test scatter data and other diagnostic tables. These outputs are generated after training from the saved checkpoint, rather than being mixed into the training loop.

In short:

```text
Main training code:
    trains the model and saves checkpoints

analysis/run_further_analysis_proposed.py:
    reloads the trained checkpoint and generates manuscript-level prediction tables
```

This separation keeps model training, post-hoc analysis and figure-oriented result generation independent and reproducible.

## Files

### `run_further_analysis_proposed.py`

This is the main further-analysis script for the proposed hierarchical framework.

It reloads the trained proposed-framework checkpoint, rebuilds the same train/test data split, applies the same train-only normalization, performs inference on the test and training sets, converts normalized SOC/SOH outputs back to raw units, and saves per-sample prediction tables for downstream analysis and plotting.

In the current implementation, the script:

* imports the trained framework from `proposed_framework/run_proposed_framework.py`;
* rebuilds cached training and test data using the same data-loading functions as the proposed framework;
* drops invalid rows containing NaN or infinite values;
* reconstructs the ID-based train/test split using the fixed random seed;
* applies U1–U41 normalization using train-only statistics;
* recomputes SOC/SOH target normalization from the training split;
* reloads the trained `stage2_soh` checkpoint;
* performs conditional-flow inference for SOC and SOH;
* converts normalized predictions back to raw SOC percentage and raw SOH units;
* saves test-set and train-set prediction tables.

Typical outputs are saved under:

```text
results/proposed_framework/further_analysis/tables/
```

Current output files include:

```text
test_predictions_per_sample.csv
train_predictions_for_scatter.csv
```

`test_predictions_per_sample.csv` contains per-sample test predictions, including battery ID, pulse width, true material label, predicted material label, true SOC, predicted SOC, true SOH and predicted SOH.

`train_predictions_for_scatter.csv` contains train-set SOC/SOH true and predicted values, mainly for generating train/test scatter comparisons.

This script is especially important because SOC and SOH predictions are produced through conditional-flow estimators. The Monte Carlo sampling setting used during inference affects the stability of the resulting estimates, so it should be kept consistent with the reported evaluation protocol.

### `error_propagation_analysis.py`

This script supports the error-propagation analysis of the hierarchical reconstruction pipeline.

The proposed framework follows the reconstruction order:

```text
Material classification → SOC estimation → SOH estimation
```

Because SOC estimation is conditioned on material information, and SOH estimation is further conditioned on both material and SOC information, upstream prediction errors may influence downstream reconstruction results. This script is used to quantify and visualize how such errors propagate across the diagnostic stages.

It is mainly used to support the cascading-error or error-propagation analysis in the manuscript, for example by examining how material-classification errors affect SOC estimation and how SOC errors further influence SOH estimation.

Use this script when reproducing the error-propagation results.

### `train_calibration_baseline.py`

This script trains or fits calibration baselines for uncertainty analysis.

The proposed framework uses conditional-flow estimators to output predictive distributions for SOC and SOH, rather than only point estimates. To evaluate whether these uncertainty estimates are reliable, they are compared with simpler uncertainty-calibration baselines, such as Gaussian or temperature-scaled uncertainty estimates.

This script is used to generate calibration-related baseline results for the manuscript uncertainty analysis.

Use this script when reproducing the uncertainty calibration comparison.

### `__init__.py`

This file marks `analysis/` as a Python package. It allows scripts or utilities in this folder to be imported by other modules if needed.

### `__pycache__/`

This is an automatically generated Python cache folder. It does not contain source code and does not need to be edited manually. It should not be committed to the repository.

## Recommended usage

Run the analysis scripts from the repository root:

```bash
python analysis/run_further_analysis_proposed.py
python analysis/error_propagation_analysis.py
python analysis/train_calibration_baseline.py
```

Running from the repository root helps keep relative paths consistent for loading data, checkpoints, cached files and saving analysis outputs.

## Relationship to the full workflow

A typical workflow is:

```text
1. Train the proposed framework
2. Save the trained checkpoints
3. Run analysis scripts in analysis/
4. Generate per-sample prediction tables and diagnostic summaries
5. Use the generated CSV files for manuscript figures and supplementary analyses
```

This design keeps model training, post-hoc analysis and figure generation clearly separated.
