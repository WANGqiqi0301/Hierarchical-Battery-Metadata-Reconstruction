# Benchmark

This folder contains benchmark models used to compare against the proposed hierarchical battery passport reconstruction framework.

The benchmark scripts are designed to evaluate whether standard tabular and deep tabular learning models can reconstruct the same passport attributes from short-pulse voltage features, including:

* material-capacity class;
* state of charge (SOC);
* state of health (SOH).

These scripts are **not part of the proposed hierarchical model**. They are used as comparison methods for the manuscript.

## Purpose of the benchmark experiments

The proposed framework reconstructs battery passport information through a material-to-charge-to-health hierarchy. To show that the performance gain does not simply come from using a machine-learning model, we compare it with several representative benchmark models under controlled settings.

The benchmark experiments are used to answer two questions:

1. **Fair comparison:**
   How well do existing tabular or deep tabular models perform when they receive the same short-pulse input information as the proposed framework?

2. **Enhanced comparison:**
   If benchmark models are additionally supplied with controlled upstream information, such as material or SOC-related information, can they close the gap with the proposed hierarchical framework?

This distinction is important because the proposed framework explicitly propagates upstream diagnostic information through the hierarchy, whereas many baseline models treat material, SOC and SOH as independent or weakly coupled prediction targets.

## Fair and enhanced benchmark settings

### Fair benchmark

In the fair benchmark setting, baseline models are trained using the same basic short-pulse input features. They do not receive extra privileged information about upstream passport attributes.

This setting evaluates whether the proposed framework improves performance because of its structured representation, hierarchical reconstruction order and uncertainty-aware conditional-flow design.

### Enhanced benchmark

In the enhanced benchmark setting, selected downstream benchmark models are provided with controlled upstream information. For example, SOC or SOH predictors may receive additional material-related or SOC-related inputs.

This setting is used to test a stricter question: whether standard benchmark models can match the proposed framework when part of the hierarchical information is manually supplied to them.

The enhanced setting is therefore not the main fair comparison, but a controlled diagnostic comparison used to examine the value of the proposed end-to-end hierarchy.

## Files

### `common.py`

This file contains shared utilities used by multiple benchmark scripts.

Typical functions in this file include common data-loading, preprocessing, train/test split handling, metric calculation and result-saving utilities. Keeping these functions in `common.py` helps ensure that different benchmark models are evaluated under consistent data and metric settings.

Use this file when modifying shared benchmark behavior.

### `enhanced_inputs.py`

This file defines utilities for constructing enhanced benchmark inputs.

The enhanced inputs are used when downstream benchmark models are supplied with additional controlled upstream information. This allows the manuscript to compare the proposed framework not only with fair baselines, but also with stronger baselines that are given extra diagnostic information.

This file is mainly used by benchmark scripts that evaluate enhanced SOC or SOH prediction settings.

### `xgboost_benchmark.py`

This script runs the XGBoost benchmark.

XGBoost is a strong tree-based baseline for tabular regression and classification tasks. In this repository, it is used to evaluate material-capacity classification, SOC estimation and SOH estimation from short-pulse features.

Use this script to reproduce the XGBoost comparison results.

### `tabnet_benchmark.py`

This script runs the TabNet benchmark.

TabNet is a deep tabular learning model based on sequential attention. It is included as a neural-network benchmark for structured tabular inputs.

Use this script to reproduce the TabNet comparison results.

### `node_benchmark.py`

This script runs the NODE benchmark.

NODE, or Neural Oblivious Decision Ensembles, is a deep learning model inspired by decision-tree ensembles. It is included as another representative deep tabular benchmark.

Use this script to reproduce the NODE comparison results.

### `ft_transformer_benchmark.py`

This script runs the FT-Transformer benchmark.

FT-Transformer is a transformer-based model for tabular data. It is included to compare the proposed framework with a modern attention-based tabular learning baseline.

Use this script to reproduce the FT-Transformer comparison results.

### `run_all_benchmarks.py`

This is a convenience script for running multiple benchmark models in one command.

It is useful when regenerating the complete benchmark comparison table for the manuscript. Depending on the configuration, it may sequentially run XGBoost, TabNet, NODE and FT-Transformer benchmarks.

Use this script when you want to reproduce all benchmark results together.

### `__init__.py`

This file marks `benchmark/` as a Python package. It allows benchmark modules and utilities to be imported by other scripts.

### `__pycache__/`

This is an automatically generated Python cache folder. It does not contain source code and does not need to be edited manually.

## Recommended usage

Run benchmark scripts from the repository root:

```bash
python benchmark/xgboost_benchmark.py
python benchmark/tabnet_benchmark.py
python benchmark/node_benchmark.py
python benchmark/ft_transformer_benchmark.py
```

To run all benchmark models together:

```bash
python benchmark/run_all_benchmarks.py
```

Running scripts from the repository root is recommended because the scripts use relative paths for loading data, accessing cached features and saving result files.

## Typical outputs

Benchmark results are saved under the repository `results/` directory, typically in benchmark-related subfolders such as:

```text
results/benchmark/
```

Depending on the script and configuration, outputs may include:

* benchmark metric summaries;
* fair benchmark results;
* enhanced benchmark results;
* cached benchmark feature tables;
* classification accuracy, SOC error and SOH error metrics.

The generated results are used for manuscript benchmark comparison figures and supplementary tables.

## Relationship to the proposed framework

The benchmark models provide external comparison methods. They do not replace the proposed hierarchical framework.

A typical comparison workflow is:

```text
1. Train and evaluate the proposed framework
2. Run benchmark models under fair settings
3. Run enhanced benchmark settings when needed
4. Compare material classification, SOC estimation and SOH estimation metrics
5. Use the resulting tables for manuscript figures and supplementary analyses
```

This design keeps the proposed framework and benchmark comparison methods clearly separated.
