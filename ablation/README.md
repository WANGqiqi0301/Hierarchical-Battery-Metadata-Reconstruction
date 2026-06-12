# Ablation Studies

This folder contains scripts for ablation experiments used to evaluate the contribution of different components of the proposed hierarchical battery metadata reconstruction framework.

The purpose of ablation studies is to understand how each part of the model or input representation affects the prediction of:

- Material-capacity classification
- State of Charge (SOC)
- State of Health (SOH)

---

## Scripts

| Script | Description |
|--------|-------------|
| `channel_ablation.py` | Evaluates the effect of using different input channels (e.g., Raw voltage, ΔU, OCV) on model performance. |
| `hierarchy_ablation.py` | Compares direct vs hierarchical prediction of SOC and SOH, showing the impact of the hierarchical structure. |
| `hierarchy_order_ablation.py` | Studies the effect of different orderings in the hierarchical prediction (e.g., Material → SOC → SOH vs other permutations). |
| `input_representation_ablation.py` | Compares different input representations (raw features vs structured features) on reconstruction performance. |
| `transfer_ablation.py` | Evaluates how different transfer strategies (hard labels vs soft probability) affect downstream prediction accuracy. |

---

## Usage

To run all ablation studies:

```bash
python channel_ablation.py
python hierarchy_ablation.py
python hierarchy_order_ablation.py
python input_representation_ablation.py
python transfer_ablation.py