# Measurement Sensitivity

This folder contains scripts for evaluating the proposed hierarchical battery passport reconstruction framework under different short-pulse measurement settings.

The scripts in this folder do not define the main proposed model architecture and are not general benchmark models. Instead, they reuse the proposed framework under controlled measurement configurations to examine how reconstruction performance changes when the input measurement protocol or input quality is modified.

The generated results are mainly used for manuscript figures and supplementary tables related to measurement efficiency, reduced testing protocols and robustness.

## Overview

The proposed framework reconstructs material-capacity class, state of charge (SOC) and state of health (SOH) from short-pulse voltage responses. In practical retired-battery screening, the full measurement protocol may not always be available or necessary. For example, one may want to reduce pulse widths, use fewer C-rate settings, simplify the pulse polarity configuration, or evaluate robustness under noisy or incomplete input features.

This folder provides scripts for these measurement-sensitivity analyses:

* `pulse_width_sensitivity.py`: evaluates different pulse-width subsets;
* `c_rate_sensitivity.py`: evaluates different C-rate subsets;
* `pulse_polarity_sensitivity.py`: compares bidirectional, positive-only and negative-only pulse settings;
* `input_quality_sensitivity.py`: evaluates robustness to degraded input quality.

## Files

### `pulse_width_sensitivity.py`

This script evaluates the effect of pulse-width selection on reconstruction performance.

The full pulse-response protocol contains multiple pulse widths. This script tests selected pulse-width subsets and records the corresponding material classification accuracy, SOC error and SOH error.

The purpose of this script is to generate reduced-protocol results showing how much performance is retained when fewer pulse widths are used. These results are used for manuscript analyses related to acquisition-burden reduction and pulse-width selection.

Typical outputs include pulse-width sensitivity summary tables saved under the `results/measurement_sensitivity/` directory.

### `c_rate_sensitivity.py`

This script evaluates the effect of C-rate selection.

Different current amplitudes provide different voltage-response characteristics, including changes in ohmic voltage jumps, polarization behavior and dynamic response strength. This script tests selected C-rate subsets and evaluates the resulting material classification, SOC estimation and SOH estimation performance.

The generated results are used to compare reconstruction performance under different current-amplitude configurations and to support the manuscript analysis of measurement-protocol design.

Typical outputs include C-rate sensitivity summary tables saved under the `results/measurement_sensitivity/` directory.

### `pulse_polarity_sensitivity.py`

This script evaluates the effect of pulse polarity.

The full proposed setting uses bidirectional pulse responses. This script compares the full bidirectional setting with reduced polarity settings, such as positive-only and negative-only pulse responses.

The generated results are used to quantify whether using both pulse directions provides additional diagnostic information compared with using only one polarity.

Typical outputs include pulse-polarity comparison tables containing material classification accuracy, SOC error and SOH error for each polarity setting.

### `input_quality_sensitivity.py`

This script evaluates the robustness of the proposed framework under degraded input quality.

In practical measurements, voltage-response features may be affected by noise, missing values, sensor limitations, contact instability or incomplete acquisition. This script tests how the model performance changes when the input features are perturbed, masked or otherwise degraded.

The generated results are used for manuscript analyses related to measurement robustness and input-quality sensitivity.

Typical outputs include noise-sensitivity or missing-feature summary tables saved under the `results/measurement_sensitivity/` directory.

### `__init__.py`

This file marks `measurement_sensitivity/` as a Python package, allowing scripts and utilities in this folder to be imported when needed.

### `__pycache__/`

This is an automatically generated Python cache folder. It does not contain source code and does not need to be edited manually.

## Recommended usage

Run the scripts from the repository root:

```bash
python measurement_sensitivity/pulse_width_sensitivity.py
python measurement_sensitivity/c_rate_sensitivity.py
python measurement_sensitivity/pulse_polarity_sensitivity.py
python measurement_sensitivity/input_quality_sensitivity.py
```

Running from the repository root is recommended because the scripts use relative paths for loading data, checkpoints, cached files and saving result tables.

## Typical outputs

The generated files are saved under the repository `results/` directory, usually in measurement-sensitivity-related subfolders such as:

```text
results/measurement_sensitivity/
```

Depending on the script, outputs may include:

* pulse-width sensitivity summaries;
* C-rate sensitivity summaries;
* pulse-polarity comparison results;
* input-quality sensitivity summaries;
* CSV tables used by figure-generation scripts and supplementary tables.

## Relationship to the full workflow

A typical workflow is:

```text
1. Train the proposed framework or prepare the required checkpoints
2. Run the selected measurement-sensitivity script
3. Save the sensitivity summary tables under results/
4. Use the generated CSV files for manuscript figures and supplementary analyses
```

This separation keeps the main model training, measurement-sensitivity evaluation and figure generation clearly organized.
