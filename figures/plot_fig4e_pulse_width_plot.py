# -*- coding: utf-8 -*-

"""
plot_fig4e_pulse_width_plot_broken_4_9.py

Generate Figure 4e:
SOH MedAE vs pulse-width configurations, using a fixed broken x-axis
that removes the interval 4-9.

Outputs:
1. Standard version with text.
2. Pure version with graphical elements only and transparent background.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 1. Paths
# =========================

BASE_DIR = "results/measurement_sensitivity/pulse_width"
SUMMARY_CSV = os.path.join(BASE_DIR, "pulse_width_sensitivity_summary.csv")
SAVE_DIR = "results/figures/main/fig4e"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 2. Load data
# =========================

if not os.path.exists(SUMMARY_CSV):
    raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")

df = pd.read_csv(SUMMARY_CSV)

custom_order = [
    "P1_70",
    "P2_3000",
    "P3_30_50_70_100",
    "P4_300_500_700",
    "P5_1000_3000_5000",
    "P6_30_50_300_500",
    "P7_30_50_3000_5000",
    "P8_300_500_3000_5000",
    "P9_All",
]

df["config"] = pd.Categorical(
    df["config"],
    categories=custom_order,
    ordered=True,
)

df = df.sort_values(
    "config",
    ascending=False,
).reset_index(drop=True)

df["plot_label"] = (
    df["config"]
    .astype(str)
    .str.extract(r"^(P\d+)")
)


# =========================
# 3. Resolve MedAE column
# =========================

MEDAE_CANDIDATES = [
    "SOH MedAE",
    "soh_medae",
    "test_soh_medae_raw",
]

metric_col = next(
    (
        col
        for col in MEDAE_CANDIDATES
        if col in df.columns
    ),
    None,
)

if metric_col is None:
    raise RuntimeError(
        "SOH MedAE column not found in pulse_width_sensitivity_summary.csv.\n"
        f"Expected one of: {MEDAE_CANDIDATES}\n"
        f"Available columns: {list(df.columns)}"
    )


# =========================
# 4. Prepare plot data
# =========================

x_data = pd.to_numeric(
    df[metric_col],
    errors="coerce",
).values

if pd.isna(x_data).any():
    raise RuntimeError(
        f"NaN values found in SOH MedAE column: {metric_col}"
    )

y_labels = df["plot_label"].values
y_pos = list(range(len(df)))

n_train = pd.to_numeric(
    df["num_pulse_widths"],
    errors="coerce",
).values

if pd.isna(n_train).any():
    raise RuntimeError(
        "NaN values found in num_pulse_widths."
    )

cmap = sns.color_palette(
    "rocket",
    as_cmap=True,
)

norm = plt.Normalize(
    vmin=0,
    vmax=max(n_train),
)

bubble_colors = [
    cmap(norm(size))
    for size in n_train
]

size_base = (
    max(n_train)
    if max(n_train) > 0
    else 1
)

point_sizes = (
    n_train / size_base
) * 1800


# =========================
# 5. Fixed broken-axis configuration
# =========================

BREAK_LEFT_END = 4.0
BREAK_RIGHT_START = 9.0

left_mask = x_data <= BREAK_LEFT_END
right_mask = x_data >= BREAK_RIGHT_START
middle_mask = (
    (x_data > BREAK_LEFT_END)
    &
    (x_data < BREAK_RIGHT_START)
)

if middle_mask.any():
    hidden_points = df.loc[
        middle_mask,
        [
            "config",
            metric_col,
        ],
    ].copy()

    print(
        "\n[WARNING] The following points fall inside "
        "the broken-out interval (4, 9)"
    )
    print(
        "and will be clipped by the fixed broken axis:"
    )
    print(
        hidden_points.to_string(
            index=False
        )
    )


# If one side has no points, keep a sensible axis anyway
left_vals = x_data[left_mask]
right_vals = x_data[right_mask]

if len(left_vals) > 0:
    left_xlim = (
        max(
            0.0,
            float(left_vals.min()) - 0.5,
        ),
        BREAK_LEFT_END,
    )
else:
    left_xlim = (
        0.0,
        BREAK_LEFT_END,
    )

if len(right_vals) > 0:
    right_xlim = (
        BREAK_RIGHT_START,
        float(right_vals.max()) + 1.0,
    )
else:
    right_xlim = (
        BREAK_RIGHT_START,
        BREAK_RIGHT_START + 1.0,
    )


# =========================
# 6. Standard plot
# =========================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    sharey=True,
    figsize=(6, 6),
    dpi=150,
    gridspec_kw={
        "width_ratios": [3, 1],
    },
)

plt.subplots_adjust(
    wspace=0.1,
    right=0.8,
)

fig.patch.set_alpha(
    0.0
)


# ---------------------------------------------------------
# Common style only
# IMPORTANT:
# Do NOT draw all bubbles inside this loop.
# Otherwise ax1 and ax2 each draw the whole dataset.
# ---------------------------------------------------------

for ax in [
    ax1,
    ax2,
]:
    ax.patch.set_alpha(
        0.0
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25,
    )

    ax.tick_params(
        axis="x",
        labelsize=10,
    )

    ax.tick_params(
        axis="y",
        labelsize=11,
        length=0,
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )


# ---------------------------------------------------------
# LEFT AXIS:
# only draw points with SOH MedAE <= 4
# ---------------------------------------------------------

left_indices = [
    i
    for i in range(len(df))
    if left_mask[i]
]

left_colors = [
    bubble_colors[i]
    for i in left_indices
]

if left_mask.any():
    ax1.hlines(
        y=y_labels[left_mask],
        xmin=0,
        xmax=x_data[left_mask],
        color="#CCCCCC",
        alpha=0.3,
        linewidth=1,
        zorder=1,
    )

    ax1.scatter(
        x_data[left_mask],
        y_labels[left_mask],
        s=point_sizes[left_mask],
        c=left_colors,
        alpha=0.8,
        edgecolors="#111111",
        linewidth=0.8,
        zorder=3,
        clip_on=True,
    )


# ---------------------------------------------------------
# RIGHT AXIS:
# only draw points with SOH MedAE >= 9
# ---------------------------------------------------------

right_indices = [
    i
    for i in range(len(df))
    if right_mask[i]
]

right_colors = [
    bubble_colors[i]
    for i in right_indices
]

if right_mask.any():
    ax2.hlines(
        y=y_labels[right_mask],
        xmin=BREAK_RIGHT_START,
        xmax=x_data[right_mask],
        color="#CCCCCC",
        alpha=0.3,
        linewidth=1,
        zorder=1,
    )

    ax2.scatter(
        x_data[right_mask],
        y_labels[right_mask],
        s=point_sizes[right_mask],
        c=right_colors,
        alpha=0.8,
        edgecolors="#111111",
        linewidth=0.8,
        zorder=3,
        clip_on=True,
    )


# Broken x-axis ranges
ax1.set_xlim(
    *left_xlim
)

ax2.set_xlim(
    *right_xlim
)


# Hide duplicated y-axis labels on the right panel
ax2.tick_params(
    labelleft=False,
    left=False,
)

ax2.spines["left"].set_visible(
    False
)


# Axis labels
ax1.set_ylabel(
    "Pulse-width configuration",
    fontsize=12,
    fontweight="bold",
)

fig.text(
    0.42,
    0.03,
    "SOH MedAE (%)",
    ha="center",
    fontsize=12,
    fontweight="bold",
)


# Broken-axis markers
d = 0.015

kwargs = dict(
    transform=ax1.transAxes,
    color="#333333",
    clip_on=False,
    lw=1.2,
)

ax1.plot(
    (
        1 - d / 3,
        1 + d / 3,
    ),
    (
        -d,
        +d,
    ),
    **kwargs,
)

ax1.plot(
    (
        1 - d / 3,
        1 + d / 3,
    ),
    (
        1 - d,
        1 + d,
    ),
    **kwargs,
)

kwargs.update(
    transform=ax2.transAxes
)

ax2.plot(
    (
        -d,
        +d,
    ),
    (
        -d,
        +d,
    ),
    **kwargs,
)

ax2.plot(
    (
        -d,
        +d,
    ),
    (
        1 - d,
        1 + d,
    ),
    **kwargs,
)


# Legend for bubble size
legend_sizes = [
    1,
    3,
    4,
    10,
]

legend_handles = []

for s in legend_sizes:
    size = (
        s / size_base
    ) * 1800

    h = ax2.scatter(
        [],
        [],
        s=size,
        c=[
            cmap(
                norm(s)
            )
        ],
        alpha=0.85,
        edgecolors="#111111",
        linewidth=0.8,
    )

    legend_handles.append(
        h
    )

ax2.legend(
    legend_handles,
    [
        str(s)
        for s in legend_sizes
    ],
    title="Number of pulse widths",
    loc="center left",
    bbox_to_anchor=(
        1.05,
        0.5,
    ),
    frameon=False,
    fontsize=10,
    title_fontsize=11,
)


# Title
fig.suptitle(
    "SOH prediction error by pulse-width configuration",
    fontsize=15,
    fontweight="bold",
    x=0.12,
    y=0.96,
    ha="left",
)


# Value labels
bbox_props = dict(
    boxstyle="round,pad=0.2",
    facecolor="#FFFFFF90",
    edgecolor="none",
    zorder=4,
)

for i, val in enumerate(
    x_data
):
    if val <= BREAK_LEFT_END:
        target_ax = ax1
        x_offset = 0.12

    elif val >= BREAK_RIGHT_START:
        target_ax = ax2
        x_offset = 0.25

    else:
        # Value is inside the broken interval 4-9.
        # Do not draw it.
        continue

    target_ax.text(
        val + x_offset,
        i,
        f"{val:.2f}%",
        va="center",
        fontsize=10,
        fontweight=(
            "bold"
            if y_labels[i] == "P9"
            else "normal"
        ),
        bbox=bbox_props,
    )


# =========================
# 7. Save standard version
# =========================

out_file = os.path.join(
    SAVE_DIR,
    "fig4e_pulse_width_performance.png",
)

fig.savefig(
    out_file,
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)

plt.close(
    fig
)


# =========================
# 8. Pure plot
# =========================

fig_pure, (
    ax1_pure,
    ax2_pure,
) = plt.subplots(
    1,
    2,
    sharey=True,
    figsize=(6, 6),
    dpi=150,
    gridspec_kw={
        "width_ratios": [3, 1],
    },
)

plt.subplots_adjust(
    wspace=0.1
)

fig_pure.patch.set_alpha(
    0.0
)


# ---------------------------------------------------------
# Common pure style only
# Again: do NOT draw all bubbles in both axes.
# ---------------------------------------------------------

for ax in [
    ax1_pure,
    ax2_pure,
]:
    ax.patch.set_alpha(
        0.0
    )

    ax.set_xticks(
        []
    )

    ax.set_yticks(
        []
    )

    ax.set_xlabel(
        ""
    )

    ax.set_ylabel(
        ""
    )

    ax.grid(
        False
    )

    for spine in ax.spines.values():
        spine.set_visible(
            False
        )


# ---------------------------------------------------------
# PURE LEFT AXIS:
# only draw <= 4
# ---------------------------------------------------------

if left_mask.any():
    left_y_pos = [
        y_pos[i]
        for i in left_indices
    ]

    ax1_pure.hlines(
        y=left_y_pos,
        xmin=0,
        xmax=x_data[left_mask],
        color="#CCCCCC",
        alpha=0.3,
        linewidth=1,
        zorder=1,
    )

    ax1_pure.scatter(
        x_data[left_mask],
        left_y_pos,
        s=point_sizes[left_mask],
        c=left_colors,
        alpha=0.8,
        edgecolors="#111111",
        linewidth=0.8,
        zorder=3,
        clip_on=True,
    )


# ---------------------------------------------------------
# PURE RIGHT AXIS:
# only draw >= 9
# ---------------------------------------------------------

if right_mask.any():
    right_y_pos = [
        y_pos[i]
        for i in right_indices
    ]

    ax2_pure.hlines(
        y=right_y_pos,
        xmin=BREAK_RIGHT_START,
        xmax=x_data[right_mask],
        color="#CCCCCC",
        alpha=0.3,
        linewidth=1,
        zorder=1,
    )

    ax2_pure.scatter(
        x_data[right_mask],
        right_y_pos,
        s=point_sizes[right_mask],
        c=right_colors,
        alpha=0.8,
        edgecolors="#111111",
        linewidth=0.8,
        zorder=3,
        clip_on=True,
    )


# Same broken-axis ranges
ax1_pure.set_xlim(
    *left_xlim
)

ax2_pure.set_xlim(
    *right_xlim
)


# Add vertical margins to avoid cropping bubbles
ax1_pure.set_ylim(
    -1.0,
    len(df),
)

ax2_pure.set_ylim(
    -1.0,
    len(df),
)


# =========================
# 9. Save pure version
# =========================

pure_out_file = os.path.join(
    SAVE_DIR,
    "fig4e_pulse_width_performance_pure.png",
)

fig_pure.savefig(
    pure_out_file,
    dpi=600,
    bbox_inches="tight",
    transparent=True,
    pad_inches=0.02,
)

plt.close(
    fig_pure
)


# =========================
# 10. Terminal output
# =========================

print(
    f"[METRIC] Using SOH MedAE column: {metric_col}"
)

print(
    "[BROKEN AXIS] True"
)

print(
    f"[BREAK RANGE] {BREAK_LEFT_END} to {BREAK_RIGHT_START}"
)

print(
    f"[LEFT POINTS] {int(left_mask.sum())}"
)

print(
    f"[RIGHT POINTS] {int(right_mask.sum())}"
)

print(
    f"[HIDDEN POINTS] {int(middle_mask.sum())}"
)

print(
    f"[OK] Figure 4e saved at: {out_file}"
)

print(
    f"[OK] Pure Figure 4e saved at: {pure_out_file}"
)
