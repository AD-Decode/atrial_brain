#!/usr/bin/env python3
"""
Figure 1 - Study design and analytic framework
Framingham Heart Study Offspring Cohort, APOE e4 heart-brain analyses.

Outputs PDF, SVG and 600-dpi PNG.
Usage:
    python make_Figure1_study_design.py [outdir]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

C = {
    "top": ("#EEF1F5", "#5B6B7F"),
    "reg": ("#EAF2FB", "#2F6DB5"),
    "lon": ("#EAF6EE", "#2E8B57"),
    "sec": ("#F6F3EC", "#8A7444"),
    "sub": ("#FFFFFF", "#9AA5B1"),
}

INK = "#1F2933"

W, H = 7.35, 6.6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, style, lw=0.9, r=1.2):
    face, edge = C[style]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            fc=face,
            ec=edge,
            lw=lw
        )
    )


def text(
    x, y, s, size=7, weight="normal",
    color=INK, ha="left", va="top",
    style="normal"
):
    ax.text(
        x, y, s,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        fontstyle=style,
        linespacing=1.22,
        wrap=True
    )


def arrow(x0, y0, x1, y1, color="#5B6B7F"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.9,
            color=color,
            shrinkA=0,
            shrinkB=0
        )
    )


# ============================================================
# TOP COHORT BOX
# ============================================================

box(9, 90.9, 82, 8.8, "top")

text(
    50, 97.9,
    "Framingham Heart Study Offspring Cohort",
    8.5, "bold", ha="center"
)

text(
    50, 95.0,
    "Controlled-access data via dbGaP (phs000007): echocardiography, CMR, brain MRI, APOE genotype,\n"
    "plasma biomarkers, neuropsychological testing, and TOPMed whole-genome sequencing",
    6.1, ha="center"
)

arrow(36, 90.9, 24, 87.8, C["reg"][1])
arrow(64, 90.9, 76, 87.8, C["lon"][1])


# ============================================================
# A. REGIONAL IMAGING ARM
# ============================================================

xa, wa = 2, 47

box(xa, 30.2, wa, 57.4, "reg", lw=1.1)

text(
    xa + 2, 86.3,
    "A  Regional imaging arm",
    8.5, "bold", C["reg"][1]
)

text(
    xa + 2, 82.8,
    "Cardiac magnetic resonance (CMR) and brain MRI at separate visits\n"
    "(CMR preceded MRI in 99.6%; mean interval 2.96 years)",
    6.45, style="italic"
)

# Samples
box(xa + 2, 64.5, 20.5, 13.7, "sub")

text(xa + 3.2, 77.2, "Samples", 7, "bold")

text(
    xa + 3.2, 74.1,
    "785  CMR + brain MRI\n"
    "769  classified APOE genotype\n"
    "765  ventricular models\n"
    "451  atrial models (LAVI)",
    6.45
)

# Cardiac exposures
box(xa + 24, 64.5, 21, 13.7, "sub")

text(
    xa + 25.2, 77.2,
    "Cardiac exposures",
    7, "bold"
)

text(
    xa + 25.2, 74.1,
    "Principal: LVEF, LVESVi,\n"
    "   LV mass index\n"
    "Additional: LVEDVi,\n"
    "   LV longitudinal strain\n"
    "Atrial: LAVI, LA emptying fraction",
    6.2
)

# Brain outcomes
box(xa + 2, 47.0, 43, 15.0, "sub")

text(
    xa + 3.2, 60.9,
    "Brain outcomes (FreeSurfer morphometry)",
    7, "bold"
)

text(
    xa + 3.2, 58.1,
    "Regional ventricular screen: 28 Desikan–Killiany + 35 Destrieux\n"
    "regions/hemisphere (volume, surface area, thickness)\n"
    "+ 7 bilateral subcortical volumes: 392 outcomes × 5 cardiac phenotypes\n"
    "= 1,960 tests\n"
    "Atrial whole-brain screen: 1,086 FreeSurfer-derived outcomes,\n"
    "including thickness variability and mean curvature",
    6.0
)

# Analysis
box(xa + 2, 31.0, 43, 14.8, "sub")

text(
    xa + 3.2, 44.4,
    "Analysis",
    7, "bold"
)

text(
    xa + 3.2, 41.6,
    "Cardiac phenotype × APOE ε4 interaction; OLS, HC3 robust SEs\n"
    "BH FDR within metric × anatomical family × cardiac phenotype\n"
    "(8 anatomical families; 110 testing groups)\n"
    "Freedman–Lane permutation of the complete ventricular screen\n"
    "Atrial screen: FDR across 1,086 outcomes + permutation",
    6.15
)

for y0, y1 in ((64.5, 62.0), (47.0, 45.8)):
    arrow(
        xa + 23.5, y0,
        xa + 23.5, y1,
        C["reg"][1]
    )


# ============================================================
# B. LONGITUDINAL ARM
# ============================================================

xb, wb = 51, 47

box(xb, 30.2, wb, 57.4, "lon", lw=1.1)

text(
    xb + 2, 86.3,
    "B  Longitudinal arm",
    8.5, "bold", C["lon"][1]
)

text(
    xb + 2, 82.8,
    "Serial echocardiography before repeated brain MRI",
    6.45, style="italic"
)

# Timeline
ty = 76.0

ax.plot(
    [xb + 4, xb + 43],
    [ty, ty],
    color=C["lon"][1],
    lw=1.0
)

for xx, lab in (
    (xb + 6, "Exam 4"),
    (xb + 12.5, "Exam 5"),
    (xb + 19, "Exam 6")
):
    ax.plot(
        xx, ty,
        "o",
        ms=4.2,
        color=C["lon"][1]
    )
    text(
        xx, ty - 1.7,
        lab,
        6.15,
        ha="center"
    )

for xx in (
    xb + 29,
    xb + 34.5,
    xb + 40
):
    ax.plot(
        xx, ty,
        "s",
        ms=3.8,
        mfc="white",
        mec=C["lon"][1],
        mew=1.0
    )

text(
    xb + 12.5, ty + 3.4,
    "Echocardiographic trajectories",
    6.4, "bold", ha="center"
)

text(
    xb + 34.5, ty + 3.4,
    "Repeated brain MRI",
    6.4, "bold", ha="center"
)

text(
    xb + 34.5, ty - 1.7,
    "median follow-up 12.5 years",
    6.15, ha="center"
)

ax.annotate(
    "",
    xy=(xb + 27.2, ty + 0.9),
    xytext=(xb + 21, ty + 0.9),
    arrowprops=dict(
        arrowstyle="-|>",
        color=C["lon"][1],
        lw=0.9
    )
)

# Exposures and outcomes
box(xb + 2, 47.0, 20.5, 22.0, "sub")

text(
    xb + 3.2, 68.2,
    "Exposures",
    7, "bold"
)

text(
    xb + 3.2, 65.3,
    "Slopes across Exams 4–6:\n"
    "LA dimension, LV mass,\n"
    "LV end-diastolic dimension,\n"
    "fractional shortening",
    6.35
)

text(
    xb + 3.2, 56.7,
    "Outcomes",
    7, "bold"
)

text(
    xb + 3.2, 53.9,
    "Hippocampal volume,\n"
    "total brain volume,\n"
    "lateral ventricular\n"
    "volume, WMH",
    6.3
)

# Samples
box(xb + 24, 47.0, 21, 22.0, "sub")

text(
    xb + 25.2, 68.2,
    "Samples",
    7, "bold"
)

text(
    xb + 25.2, 65.3,
    "2,522  with LA trajectory\n"
    "           and APOE genotype\n"
    "1,305  with ≥2 later MRIs\n"
    "           (3,787 observations)\n"
    "1,260  ≥2-year gap\n"
    "           (3,646 observations)",
    6.35
)

# Analysis
box(xb + 2, 31.0, 43, 14.8, "sub")

text(
    xb + 3.2, 44.4,
    "Analysis",
    7, "bold"
)

text(
    xb + 3.2, 41.6,
    "Trajectory × MRI time × APOE ε4 in mixed-effects models\n"
    "(4 trajectories × 4 outcomes = 16 tests, FDR); random-slope refit\n"
    "GEE, selection weighting (IPSW), head-size (ICV),\n"
    "and genotype-specific sensitivity analyses",
    6.15
)

arrow(
    xb + 23.5, 47.0,
    xb + 23.5, 45.8,
    C["lon"][1]
)


# ============================================================
# BOTTOM: SENSITIVITY AND SECONDARY ANALYSES
# ============================================================

arrow(
    25, 30.2,
    36, 24.8,
    C["sec"][1]
)

arrow(
    75, 30.2,
    64, 24.8,
    C["sec"][1]
)

box(
    2, 2,
    96, 22.8,
    "sec",
    lw=1.1
)

text(
    4, 23.2,
    "Sensitivity and secondary analyses",
    8.5, "bold", C["sec"][1]
)

cols = [
    (
        "Robustness",
        "APOE ε4 allele dose and genotype\n"
        "Early vascular/metabolic covariates\n"
        "Antihypertensive treatment\n"
        "Atrial fibrillation/flutter\n"
        "Pedigree-cluster robust inference\n"
        "Imaging interval; influential points"
    ),
    (
        "Biomarkers and cognition",
        "Plasma Aβ42/40, p-tau181,\n"
        "GFAP, NfL\n"
        "Repeated MMSE-based scores\n"
        "Logical Memory, Trails B,\n"
        "animal fluency, Digit Span Backward"
    ),
    (
        "Molecular context",
        "TOPMed whole-genome sequencing:\n"
        "pathway-level effect modification\n"
        "Allen Human Brain Atlas:\n"
        "regional enrichment and spatial\n"
        "correspondence with interaction maps"
    ),
]

cw = 29.3

for i, (h, body) in enumerate(cols):
    x0 = 4 + i * (cw + 1.85)

    box(
        x0, 4.2,
        cw, 16.0,
        "sub"
    )

    text(
        x0 + 1.2, 18.8,
        h,
        7, "bold"
    )

    text(
        x0 + 1.2, 16.0,
        body,
        6.25
    )


# ============================================================
# SAVE
# ============================================================

for ext, kw in (
    ("pdf", {}),
    ("svg", {}),
    ("png", {"dpi": 600})
):
    fig.savefig(
        OUT / f"Figure1.{ext}",
        bbox_inches="tight",
        **kw
    )

print(
    "written:",
    *[
        OUT / f"Figure1.{e}"
        for e in ("pdf", "svg", "png")
    ],
    sep="\n  "
)
