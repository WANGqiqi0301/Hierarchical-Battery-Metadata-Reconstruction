# -*- coding: utf-8 -*-
"""
plot_fig2a_dataset_sankey.py

Figure 2a:
Dataset material-capacity distribution Sankey diagram.

This script generates the Sankey diagram showing:
- total samples
- material-level sample counts
- capacity-level sample counts

Default output:
    results/figures/main/fig2a/fig2a_dataset_sankey.png

Required:
    plotly
    kaleido, for saving static PNG
"""

from __future__ import annotations

import argparse
import os

import plotly.graph_objects as go


# =============================================================================
# Default configuration
# =============================================================================
DEFAULT_OUTPUT_DIR = r"results/figures/main/fig2a"
DEFAULT_OUTPUT_NAME = "fig2a_dataset_sankey.png"


# =============================================================================
# Argument parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2a dataset Sankey diagram."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the output figure.",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Output figure filename.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=700,
        help="Figure width in pixels.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Figure height in pixels.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Image export scale. Higher values give higher resolution.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the interactive figure after saving.",
    )

    return parser.parse_args()


# =============================================================================
# Figure construction
# =============================================================================
def build_sankey_figure(width: int = 700, height: int = 600) -> go.Figure:
    """Build the Figure 2a Sankey diagram."""

    # Node labels are intentionally hidden in the plot, as in the original code.
    # Kept here for readability and future editing.
    labels = [
        "Total Samples<br>640",
        "LMO<br>353",
        "LFP<br>152",
        "NMC<br>135",
        "10Ah (95)",
        "25Ah (96)",
        "26Ah (98)",
        "24Ah (64)",
        "68Ah (96)",
        "35Ah (56)",
        "15Ah (83)",
        "21Ah (52)",
    ]

    node_colors = [
        "#5DADE2",  # Total
        "#1ABC9C",
        "#2ECC71",
        "#16A085",  # LMO, LFP, NMC
        "#A9DFBF",
        "#A9DFBF",
        "#A9DFBF",
        "#A9DFBF",  # LMO branches
        "#ABEBC6",
        "#ABEBC6",  # LFP branches
        "#82E0AA",
        "#82E0AA",  # NMC branches
    ]

    sources = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3]
    targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    values = [353, 152, 135, 95, 96, 98, 64, 96, 56, 83, 52]

    link_colors = [
        "rgba(93, 173, 226, 0.4)",
        "rgba(93, 173, 226, 0.4)",
        "rgba(93, 173, 226, 0.4)",
        "rgba(26, 188, 156, 0.4)",
        "rgba(26, 188, 156, 0.4)",
        "rgba(26, 188, 156, 0.4)",
        "rgba(26, 188, 156, 0.4)",
        "rgba(46, 204, 113, 0.4)",
        "rgba(46, 204, 113, 0.4)",
        "rgba(22, 160, 133, 0.4)",
        "rgba(22, 160, 133, 0.4)",
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=10,
                    thickness=100,
                    line=dict(color="white", width=4),
                    label=[],  # keep identical to original: no node text
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)

    fig = build_sankey_figure(
        width=args.width,
        height=args.height,
    )

    try:
        fig.write_image(
            save_path,
            scale=args.scale,
        )
        print(f"[OK] Saved Figure 2a Sankey diagram: {save_path}")
    except ValueError as exc:
        raise RuntimeError(
            "Failed to save PNG. Please install kaleido first:\n"
            "    pip install kaleido"
        ) from exc

    if args.show:
        fig.show()


if __name__ == "__main__":
    main()