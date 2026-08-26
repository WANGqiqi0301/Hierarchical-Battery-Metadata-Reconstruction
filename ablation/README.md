# Ablation Studies

This folder contains scripts for ablation experiments used to evaluate the contribution of different components of the proposed hierarchical battery metadata reconstruction framework.

The purpose of these ablation studies is to understand how different architectural choices, conditioning strategies, hierarchical structures, and input representations affect the reconstruction of:

- Material-capacity classification
- State of Charge (SOC)
- State of Health (SOH)

---

## Scripts

| Script | Description |
|----------|-------------|
| `channel_ablation.py` | Evaluates the effect of different input channels (e.g., raw voltage, ΔU, and OCV-related features) on reconstruction performance. |
| `hierarchy_ablation.py` | Compares direct prediction and hierarchical prediction to quantify the benefit of the hierarchical framework. |
| `hierarchy_order_ablation.py` | Evaluates different hierarchical prediction orders (e.g., Material → SOC → SOH and alternative permutations). |
| `input_representation_ablation.py` | Compares raw input representations and structured feature representations. |
| `material_conditioning_ablation.py` | Compares soft material conditioning and hard material conditioning for downstream SOC and SOH estimation. |

---

## Usage

Run an individual ablation study:

```bash
python ablation/channel_ablation.py
python ablation/hierarchy_ablation.py
python ablation/hierarchy_order_ablation.py
python ablation/input_representation_ablation.py
python ablation/material_conditioning_ablation.py
