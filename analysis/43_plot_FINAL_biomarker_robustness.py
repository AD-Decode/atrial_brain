#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/data/qiallab/Framingham")

INFILE = (
    ROOT
    / "results/tier3_AD_biomarkers_completecase"
    / "APOE_AD_biomarker_completecase_proper_CI.csv"
)

OUTDIR = (
    ROOT
    / "results/tier3_AD_biomarkers_completecase"
)

df = pd.read_csv(INFILE)

# ============================================================
# Labels / ordering
# ============================================================

pair_order = [
    ("lh_middletemporal_vol", "LVEF"),
    ("rh_lingual_area", "LV_MASSi"),
    ("rh_lingual_vol", "LV_MASSi"),
    ("rh_superiorfrontal_area", "LVESVi"),
]

pair_labels = {
    ("lh_middletemporal_vol", "LVEF"):
        "Left middle temporal volume × LVEF",

    ("rh_lingual_area", "LV_MASSi"):
        "Right lingual area × LV mass index",

    ("rh_lingual_vol", "LV_MASSi"):
        "Right lingual volume × LV mass index",

    ("rh_superiorfrontal_area", "LVESVi"):
        "Right superior frontal area × LVESV index",
}

spec_order = [
    "M1_AMYLOID",
    "M2_PTAU",
    "M3_GFAP",
    "M4_NFL",
    "M5_AMYLOID_PTAU",
    "M6_GFAP_NFL",
    "M7_ALL4",
]

spec_labels = {
    "M1_AMYLOID": "Aβ42/40",
    "M2_PTAU": "pTau181",
    "M3_GFAP": "GFAP",
    "M4_NFL": "NfL",
    "M5_AMYLOID_PTAU": "Aβ42/40 + pTau181",
    "M6_GFAP_NFL": "GFAP + NfL",
    "M7_ALL4": "All four biomarkers",
}

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 6.4),
    sharey=True
)

for ax, pair in zip(axes, pair_order):

    region, cardiac = pair

    sub = df[
        (df["region"] == region)
        &
        (df["cardiac"] == cardiac)
    ].copy()

    sub["spec"] = pd.Categorical(
        sub["spec"],
        categories=spec_order,
        ordered=True
    )

    sub = sub.sort_values("spec")

    y = np.arange(len(spec_order))

    # zero reference
    ax.axvline(
        0,
        color="black",
        linewidth=1,
        alpha=0.55
    )

    # matched baseline for ALL4 complete-case sample
    all4 = sub[
        sub["spec"] == "M7_ALL4"
    ]

    if len(all4) == 1:
        ref_beta = float(
            all4[
                "base_beta_interaction"
            ].iloc[0]
        )

        ax.axvline(
            ref_beta,
            color="#9A9A9A",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85
        )

    for j, spec in enumerate(spec_order):

        r = sub[sub["spec"] == spec]

        if len(r) != 1:
            continue

        r = r.iloc[0]

        beta = r[
            "adjusted_beta_interaction"
        ]

        lo = r[
            "adjusted_CI_interaction_low"
        ]

        hi = r[
            "adjusted_CI_interaction_high"
        ]

        # neutral models vs highlighted ALL4
        if spec == "M7_ALL4":
            point_color = "#7B4CC2"
            marker = "D"
            markersize = 8.5
            linewidth = 2.0
            zorder = 4
        else:
            point_color = "#6F6F6F"
            marker = "o"
            markersize = 6.5
            linewidth = 1.5
            zorder = 3

        ax.errorbar(
            beta,
            j,
            xerr=[
                [beta - lo],
                [hi - beta]
            ],
            fmt=marker,
            markersize=markersize,
            color=point_color,
            ecolor=point_color,
            elinewidth=linewidth,
            capsize=3.5,
            zorder=zorder
        )

    ax.set_title(
        pair_labels[pair],
        fontsize=10.5,
        fontweight="bold",
        pad=11
    )

    ax.set_xlabel(
        "Cardiac × APOE4 interaction β",
        fontsize=9.5
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [spec_labels[s] for s in spec_order],
        fontsize=9
    )

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "APOE4-modified cardiac–brain associations are robust to blood biomarker adjustment",
    fontsize=15,
    fontweight="bold",
    y=0.98
)

fig.text(
    0.5,
    0.025,
    (
        "Points show standardized interaction coefficients with HC3 robust 95% confidence intervals. "
        "Gray points show individual or paired biomarker adjustments; purple diamonds show the full "
        "Aβ42/40 + pTau181 + GFAP + NfL model. Dashed lines show the matched complete-case baseline."
    ),
    ha="center",
    fontsize=9
)

plt.tight_layout(
    rect=[0.02, 0.07, 1, 0.94]
)

png = (
    OUTDIR
    / "figure_biomarker_robustness_FINAL.png"
)

pdf = (
    OUTDIR
    / "figure_biomarker_robustness_FINAL.pdf"
)

plt.savefig(
    png,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    pdf,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Save exact plotted table
# ============================================================

cols = [
    "spec",
    "region",
    "cardiac",
    "N",
    "base_beta_interaction",
    "adjusted_beta_interaction",
    "adjusted_CI_interaction_low",
    "adjusted_CI_interaction_high",
    "adjusted_p_interaction",
    "q_adjusted_interaction",
    "attenuation_pct",
]

df[cols].to_csv(
    OUTDIR
    / "figure_biomarker_robustness_FINAL_table.csv",
    index=False
)

print("\nCreated:")
print(png)
print(pdf)

print("\nTable:")
print(
    OUTDIR
    / "figure_biomarker_robustness_FINAL_table.csv"
)
