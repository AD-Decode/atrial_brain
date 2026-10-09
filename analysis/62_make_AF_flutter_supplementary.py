#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/data/qiallab/Framingham")

INFILE = (
    ROOT
    / "results"
    / "AF_flutter_sensitivity"
    / "AF_flutter_primary4_sensitivity.csv"
)

OUTDIR = (
    ROOT
    / "results"
    / "supplementary_AF_flutter"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

TABLE_CSV = (
    OUTDIR
    / "SupplementaryTableS4_AF_flutter_sensitivity.csv"
)

TABLE_XLSX = (
    OUTDIR
    / "SupplementaryTableS4_AF_flutter_sensitivity.xlsx"
)

FIG_PDF = (
    OUTDIR
    / "SupplementaryFigureS4_AF_flutter_sensitivity.pdf"
)

FIG_PNG = (
    OUTDIR
    / "SupplementaryFigureS4_AF_flutter_sensitivity.png"
)

FIG_SVG = (
    OUTDIR
    / "SupplementaryFigureS4_AF_flutter_sensitivity.svg"
)

CAPTION_TXT = (
    OUTDIR
    / "SupplementaryFigureS4_AF_flutter_caption.txt"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INFILE)

print("Loaded:")
print(INFILE)
print("Rows:", len(df))


# ============================================================
# LABELS
# ============================================================

association_labels = {
    ("lh_middletemporal_vol", "LVEF"):
        "LVEF × APOE ε4 →\nL middle temporal volume",

    ("rh_superiorfrontal_area", "LVESVi"):
        "LVESVi × APOE ε4 →\nR superior frontal area",

    ("rh_lingual_vol", "LV_MASSi"):
        "LV mass index × APOE ε4 →\nR lingual volume",

    ("rh_lingual_area", "LV_MASSi"):
        "LV mass index × APOE ε4 →\nR lingual surface area",
}

short_labels = {
    ("lh_middletemporal_vol", "LVEF"):
        "LVEF × APOE ε4 → L middle temporal volume",

    ("rh_superiorfrontal_area", "LVESVi"):
        "LVESVi × APOE ε4 → R superior frontal area",

    ("rh_lingual_vol", "LV_MASSi"):
        "LV mass index × APOE ε4 → R lingual volume",

    ("rh_lingual_area", "LV_MASSi"):
        "LV mass index × APOE ε4 → R lingual surface area",
}

analysis_order = [
    "Original M3",
    "M3 + AF history",
    "Exclude AF positive",
]

analysis_labels = {
    "Original M3":
        "Original M3",

    "M3 + AF history":
        "M3 + AF history",

    "Exclude AF positive":
        "Exclude AF history",
}

association_order = [
    ("lh_middletemporal_vol", "LVEF"),
    ("rh_superiorfrontal_area", "LVESVi"),
    ("rh_lingual_vol", "LV_MASSi"),
    ("rh_lingual_area", "LV_MASSi"),
]


# ============================================================
# VALIDATE
# ============================================================

required = [
    "region",
    "cardiac",
    "analysis",
    "N",
    "N_APOE4_noncarrier",
    "N_APOE4_carrier",
    "beta_interaction",
    "CI_low",
    "CI_high",
    "p_interaction",
    "q_interaction",
    "percent_change_abs",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise KeyError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# SUPPLEMENTARY TABLE S4
# ============================================================

table_rows = []

for region, cardiac in association_order:

    sub = df.loc[
        (df["region"] == region)
        & (df["cardiac"] == cardiac)
    ].copy()

    for analysis in analysis_order:

        r = sub.loc[
            sub["analysis"] == analysis
        ]

        if len(r) != 1:
            raise RuntimeError(
                f"Expected exactly one row for "
                f"{region}, {cardiac}, {analysis}; "
                f"found {len(r)}"
            )

        r = r.iloc[0]

        pct = (
            np.nan
            if analysis == "Original M3"
            else r["percent_change_abs"]
        )

        table_rows.append({
            "Association":
                short_labels[(region, cardiac)],

            "Sensitivity analysis":
                analysis_labels[analysis],

            "N":
                int(r["N"]),

            "APOE ε4 noncarriers, n":
                int(r["N_APOE4_noncarrier"]),

            "APOE ε4 carriers, n":
                int(r["N_APOE4_carrier"]),

            "Interaction β":
                r["beta_interaction"],

            "95% CI lower":
                r["CI_low"],

            "95% CI upper":
                r["CI_high"],

            "P":
                r["p_interaction"],

            "FDR q":
                r["q_interaction"],

            "Change in |β| vs original, %":
                pct,
        })


table = pd.DataFrame(table_rows)


# ------------------------------------------------------------
# Save numeric table
# ------------------------------------------------------------

table.to_csv(
    TABLE_CSV,
    index=False
)

table.to_excel(
    TABLE_XLSX,
    index=False
)


# ------------------------------------------------------------
# Print publication-style formatted table
# ------------------------------------------------------------

formatted = table.copy()

formatted["Interaction β"] = (
    formatted["Interaction β"]
    .map(lambda x: f"{x:.3f}")
)

formatted["95% CI"] = (
    table.apply(
        lambda r:
            f"{r['95% CI lower']:.3f} to "
            f"{r['95% CI upper']:.3f}",
        axis=1
    )
)

formatted["P"] = (
    table["P"]
    .map(
        lambda x:
            f"{x:.2e}"
            if x < 0.001
            else f"{x:.4f}"
    )
)

formatted["FDR q"] = (
    table["FDR q"]
    .map(
        lambda x:
            f"{x:.2e}"
            if x < 0.001
            else f"{x:.4f}"
    )
)

formatted["Change in |β| vs original, %"] = (
    table["Change in |β| vs original, %"]
    .map(
        lambda x:
            "—"
            if pd.isna(x)
            else f"{x:+.1f}"
    )
)

formatted = formatted[
    [
        "Association",
        "Sensitivity analysis",
        "N",
        "APOE ε4 noncarriers, n",
        "APOE ε4 carriers, n",
        "Interaction β",
        "95% CI",
        "P",
        "FDR q",
        "Change in |β| vs original, %",
    ]
]

FORMATTED_CSV = (
    OUTDIR
    / "SupplementaryTableS4_AF_flutter_sensitivity_formatted.csv"
)

formatted.to_csv(
    FORMATTED_CSV,
    index=False
)


# ============================================================
# FOREST PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.8, 6.8)
)

# One center position per association
centers = np.arange(
    len(association_order)
)[::-1] * 1.5

offsets = {
    "Original M3": 0.28,
    "M3 + AF history": 0.00,
    "Exclude AF positive": -0.28,
}

markers = {
    "Original M3": "o",
    "M3 + AF history": "s",
    "Exclude AF positive": "^",
}

# Slightly different marker sizes so original model reads as anchor
sizes = {
    "Original M3": 7.2,
    "M3 + AF history": 6.5,
    "Exclude AF positive": 6.5,
}


for i, key in enumerate(association_order):

    region, cardiac = key
    center = centers[i]

    sub = df.loc[
        (df["region"] == region)
        & (df["cardiac"] == cardiac)
    ]

    for analysis in analysis_order:

        r = sub.loc[
            sub["analysis"] == analysis
        ].iloc[0]

        beta = r["beta_interaction"]
        lo = r["CI_low"]
        hi = r["CI_high"]

        y = center + offsets[analysis]

        ax.errorbar(
            beta,
            y,
            xerr=np.array([
                [beta - lo],
                [hi - beta]
            ]),
            fmt=markers[analysis],
            markersize=sizes[analysis],
            capsize=3,
            linewidth=1.4,
            label=(
                analysis_labels[analysis]
                if i == 0
                else None
            ),
        )


# Reference line
ax.axvline(
    0,
    linewidth=1,
    linestyle="--",
)


# Association separators
for i in range(
    len(association_order) - 1
):
    y = (
        centers[i]
        + centers[i + 1]
    ) / 2

    ax.axhline(
        y,
        linewidth=0.6,
        alpha=0.35,
    )


ax.set_yticks(centers)

ax.set_yticklabels(
    [
        association_labels[x]
        for x in association_order
    ],
    fontsize=10,
)

ax.set_xlabel(
    "Standardized cardiac × APOE ε4 interaction β (95% CI)",
    fontsize=11,
)

ax.set_title(
    "Robustness of primary heart–brain interactions to atrial fibrillation history",
    fontsize=13,
    pad=14,
)

ax.legend(
    frameon=False,
    loc="lower right",
    fontsize=9,
)

ax.grid(
    axis="x",
    alpha=0.18,
    linewidth=0.6,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylim(
    centers[-1] - 0.75,
    centers[0] + 0.75
)

fig.subplots_adjust(
    left=0.34,
    right=0.97,
    top=0.88,
    bottom=0.13,
)

fig.savefig(
    FIG_PDF,
    bbox_inches="tight"
)

fig.savefig(
    FIG_PNG,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    FIG_SVG,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# CAPTION / FOOTNOTE
# ============================================================

caption = """Supplementary Figure S4. Robustness of primary cardiac × APOE ε4 interactions to atrial fibrillation history.

Forest plots show standardized cardiac × APOE ε4 interaction coefficients and 95% confidence intervals for the four primary heart–brain associations under the original M3 specification, after additional adjustment for atrial fibrillation (AF) history, and after exclusion of participants with AF history. Regional brain volumes were normalized to intracranial volume before standardization; cortical surface-area models additionally adjusted for total cortical surface area. All models adjusted for age at brain MRI, sex, MRI–CMR interval, body mass index, systolic blood pressure, current smoking, and diabetes history. Heteroskedasticity-consistent HC3 standard errors were used. AF history was defined as af_history_nearest_exam = 1; code 2 and missing AF history were treated as unknown. ECG evidence of AF/flutter (rhythm code 6) identified seven participants, all of whom were already classified as AF-history positive. Interaction estimates remained directionally consistent and changed only modestly after AF adjustment or exclusion.
"""

CAPTION_TXT.write_text(caption)


# ============================================================
# TABLE FOOTNOTE
# ============================================================

TABLE_NOTE = (
    OUTDIR
    / "SupplementaryTableS4_AF_flutter_notes.txt"
)

table_note = """Supplementary Table S4. Sensitivity of primary cardiac × APOE ε4 interactions to atrial fibrillation history.

β values represent standardized cardiac × APOE ε4 interaction coefficients. Regional brain volumes were normalized to intracranial volume before standardization. Surface-area models additionally adjusted for total cortical surface area. M3 included age at brain MRI, sex, MRI–CMR interval, body mass index, systolic blood pressure, current smoking, diabetes history, and APOE ε4 carrier status. HC3 heteroskedasticity-consistent standard errors were used. FDR q values were calculated across the four prespecified primary interactions within each sensitivity-analysis specification. AF-history code 1 was considered positive; code 0 negative; code 2 and missing values were treated as unknown.
"""

TABLE_NOTE.write_text(table_note)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SUPPLEMENTARY AF/FLUTTER OUTPUTS CREATED")
print("=" * 80)

for f in [
    TABLE_CSV,
    TABLE_XLSX,
    FORMATTED_CSV,
    FIG_PDF,
    FIG_PNG,
    FIG_SVG,
    CAPTION_TXT,
    TABLE_NOTE,
]:
    print(f)

print("\nCompact table preview:\n")
print(
    formatted.to_string(
        index=False
    )
)

