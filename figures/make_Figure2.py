#!/usr/bin/env python3
"""
Figure 2 - Antecedent left atrial remodeling, APOE e4,
and subsequent lateral ventricular change.

Panels
A  Predicted lateral ventricular expansion across antecedent
   LA remodeling by APOE e4 status.
B  APOE e4-stratified slopes from the principal model.
C  Three-way interaction across estimators.
D  Three-way interaction after Exam-4 vascular/metabolic and
   antihypertensive-treatment adjustment.

Usage:
    python make_Figure2.py [--strict] [--outdir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker


ROOT = Path("/data/qiallab/Framingham")
TP = ROOT / "results/longitudinal_heart_brain/temporal_prediction"

IN = {
    "native":
        ROOT
        / "results/longitudinal_heart_brain/native_effect_predictions"
        / "151_APOE4_LA_predicted_ventricular_expansion.csv",

    "effects":
        TP
        / "gee_validation"
        / "FOCUS_LA_lateral_ventricle_effects.tsv",

    "slopes":
        TP
        / "gee_validation"
        / "FOCUS_LA_lateral_ventricle_simple_slopes.tsv",

    "random_slope":
        TP
        / "mixedlm_random_time_slope"
        / "PRIMARY_APOE4_temporal_interactions_random_time_slope.tsv",

    "vascular":
        TP
        / "vascular_adjustment"
        / "LA_ventricle_early_vascular_adjustment.tsv",

    # Optional dedicated location if we later separate script 161 outputs.
    "antihyp":
        TP
        / "vascular_adjustment_antihypertensive"
        / "LA_ventricle_early_vascular_adjustment.tsv",
}

TERM3 = "mri_time_years:cardiac_z:APOE4_carrier"

ANNOT_A = (
    "LA remodeling × MRI time × APOE ε4: "
    "β = −0.00880, P = 0.0022\n"
    "(expanded random-slope model)"
)

VERIFY = {
    ("B", "noncarrier"):
        0.00357,

    ("B", "carrier"):
        -0.00501,

    ("C", "Random intercept, prospective"):
        -0.00858,

    ("C", "GEE, prospective"):
        -0.00852,

    ("C", "Random slope, prospective"):
        -0.00883,

    ("D", "M1_early_vascular"):
        -0.00882,

    ("D", "M2_early_vascular_plus_DBP"):
        -0.00882,

    ("D", "M3_matched_no_HTNtx"):
        -0.00890,

    ("D", "M4_plus_HTNtx"):
        -0.00893,
}

TOL = 5e-5

NC = "#3B6FB6"
CA = "#D9822B"
INK = "#1F2933"
GREY = "#7B8794"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "font.size": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def read_file(path):
    return pd.read_csv(
        path,
        sep="\t" if path.suffix == ".tsv" else ","
    )


def need(key):
    p = IN[key]

    if not p.exists():
        sys.exit(
            f"Missing input for {key}: {p}"
        )

    return read_file(p)


def letter(ax, s):
    ax.text(
        -0.16,
        1.10,
        s,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom"
    )


def style(ax):
    for sp in (
        "top",
        "right",
    ):
        ax.spines[
            sp
        ].set_visible(
            False
        )


def forest(
    ax,
    rows,
    colors
):
    """
    rows:
      list of (label, beta, lo, hi)
      displayed top to bottom.
    """

    y = np.arange(
        len(rows)
    )[::-1]

    for yi, (
        lab,
        b,
        lo,
        hi
    ), c in zip(
        y,
        rows,
        colors
    ):

        ax.errorbar(
            b,
            yi,
            xerr=[
                [b - lo],
                [hi - b]
            ],
            fmt="o",
            ms=4,
            color=c,
            capsize=2.5,
            lw=1.1
        )

    ax.axvline(
        0,
        color=GREY,
        lw=0.7,
        ls="--"
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        [
            r[0]
            for r in rows
        ]
    )

    ax.tick_params(
        axis="y",
        length=0
    )

    style(
        ax
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--strict",
        action="store_true"
    )

    ap.add_argument(
        "--outdir",
        default=str(
            ROOT
            / "results/paper_figures_tables/Figure2"
        )
    )

    a = ap.parse_args()

    out = Path(
        a.outdir
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_rows = []
    checks = []

    def check(
        key,
        value
    ):
        if key in VERIFY:

            ok = (
                abs(
                    value
                    - VERIFY[key]
                )
                <= TOL
            )

            checks.append(
                {
                    "panel":
                        key[0],

                    "item":
                        key[1],

                    "published":
                        VERIFY[key],

                    "plotted":
                        round(
                            float(value),
                            6
                        ),

                    "match":
                        ok,
                }
            )

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(
            7.4,
            6.3
        ),
        gridspec_kw={
            "height_ratios": [1.0, 1.35],
            "hspace": 0.52,
            "wspace": 0.64,
        }
    )

    axA, axB, axC, axD = (
        axs.ravel()
    )

    # ========================================================
    # A. Native-unit predicted ventricular expansion
    # ========================================================

    nat = need(
        "native"
    )

    for apoe, c, lab in (
        (
            0,
            NC,
            "APOE ε4 noncarriers"
        ),
        (
            1,
            CA,
            "APOE ε4 carriers"
        ),
    ):

        z = (
            nat[
                nat[
                    "APOE4_carrier"
                ]
                == apoe
            ]
            .sort_values(
                "la_remodeling_cm_per_year"
            )
        )

        x = (
            z[
                "la_remodeling_cm_per_year"
            ]
            .to_numpy(
                float
            )
        )

        axA.plot(
            x,
            z[
                "predicted_ventricular_expansion_ml_per_year"
            ],
            color=c,
            lw=1.6,
            label=lab
        )

        axA.fill_between(
            x,
            z[
                "ci95_low_ml_per_year"
            ],
            z[
                "ci95_high_ml_per_year"
            ],
            color=c,
            alpha=0.15,
            lw=0
        )

        for _, r in z.iterrows():

            plot_rows.append(
                {
                    "panel":
                        "A",

                    "group":
                        lab,

                    "x_cm_per_year":
                        r[
                            "la_remodeling_cm_per_year"
                        ],

                    "estimate":
                        r[
                            "predicted_ventricular_expansion_ml_per_year"
                        ],

                    "ci_low":
                        r[
                            "ci95_low_ml_per_year"
                        ],

                    "ci_high":
                        r[
                            "ci95_high_ml_per_year"
                        ],
                }
            )

    axA.set_xlabel(
        "Antecedent LA remodeling (cm/year)"
    )

    axA.set_ylabel(
        "Predicted lateral ventricular\n"
        "expansion (mL/year)"
    )

    axA.legend(
        frameon=False,
        loc="lower left",
        fontsize=6.5
    )

    axA.text(
        0.0,
        1.02,
        ANNOT_A,
        transform=axA.transAxes,
        fontsize=5.8,
        va="bottom",
        color=INK,
        linespacing=1.2
    )

    style(
        axA
    )

    letter(
        axA,
        "A"
    )

    # ========================================================
    # B. APOE-stratified slopes
    # ========================================================

    s = need(
        "slopes"
    )

    s = s[
        (
            s[
                "subset"
            ]
            == "prospective"
        )
        &
        (
            s[
                "method"
            ]
            == "MixedLM_random_intercept"
        )
    ]

    if "term" in s.columns:

        s = s[
            s[
                "term"
            ]
            .astype(str)
            .str.contains(
                "time",
                case=False
            )
        ]

    s = s.sort_values(
        "APOE4_carrier"
    )

    if len(s) != 2:

        sys.exit(
            "Panel B: expected 2 rows "
            f"(noncarrier, carrier), found {len(s)}"
        )

    for i, (
        _,
        r
    ) in enumerate(
        s.iterrows()
    ):

        c = (
            CA
            if r[
                "APOE4_carrier"
            ]
            == 1
            else NC
        )

        axB.errorbar(
            i,
            r[
                "beta"
            ],
            yerr=[
                [
                    r[
                        "beta"
                    ]
                    - r[
                        "CI_low"
                    ]
                ],
                [
                    r[
                        "CI_high"
                    ]
                    - r[
                        "beta"
                    ]
                ],
            ],
            fmt=(
                "s"
                if r[
                    "APOE4_carrier"
                ]
                == 1
                else "o"
            ),
            ms=5,
            color=c,
            capsize=3,
            lw=1.3
        )

        g = (
            "carrier"
            if r[
                "APOE4_carrier"
            ]
            == 1
            else "noncarrier"
        )

        check(
            (
                "B",
                g
            ),
            r[
                "beta"
            ]
        )

        plot_rows.append(
            {
                "panel":
                    "B",

                "group":
                    g,

                "estimate":
                    r[
                        "beta"
                    ],

                "ci_low":
                    r[
                        "CI_low"
                    ],

                "ci_high":
                    r[
                        "CI_high"
                    ],
            }
        )

    axB.axhline(
        0,
        color=GREY,
        lw=0.7,
        ls="--"
    )

    axB.set_xlim(
        -0.6,
        1.6
    )

    axB.set_xticks(
        [
            0,
            1
        ]
    )

    axB.set_xticklabels(
        [
            "APOE ε4\nnoncarriers",
            "APOE ε4\ncarriers"
        ]
    )

    axB.set_ylabel(
        "Slope of LA remodeling on annual\n"
        "ventricular change (SD/year per SD)"
    )

    style(
        axB
    )

    letter(
        axB,
        "B"
    )

    # ========================================================
    # C. Estimator sensitivity
    # ========================================================

    e = need(
        "effects"
    )

    if "term" in e.columns:

        e = e[
            e[
                "term"
            ]
            == TERM3
        ]

    def eff(
        subset,
        method
    ):

        z = e[
            (
                e[
                    "subset"
                ]
                == subset
            )
            &
            (
                e[
                    "method"
                ]
                == method
            )
        ]

        if len(z) != 1:

            sys.exit(
                "Panel C: expected one row for "
                f"{subset}/{method}, found {len(z)}"
            )

        r = z.iloc[
            0
        ]

        return (
            r[
                "beta"
            ],
            r[
                "CI_low"
            ],
            r[
                "CI_high"
            ],
        )

    rowsC = [
        (
            "Random intercept\nprospective",
            *eff(
                "prospective",
                "MixedLM_random_intercept"
            )
        )
    ]

    if IN[
        "random_slope"
    ].exists():

        rs = need(
            "random_slope"
        )

        rs = rs[
            (
                rs[
                    "cardiac"
                ]
                == "LA_dimension"
            )
            &
            (
                rs[
                    "outcome"
                ]
                == "lateral_ventricles"
            )
        ]

        if "subset" in rs.columns:

            rs = rs[
                rs[
                    "subset"
                ]
                == "prospective"
            ]

        if len(rs) != 1:

            sys.exit(
                "Panel C: expected one random-slope "
                f"LA/lateral-ventricle row, found {len(rs)}"
            )

        r = rs.iloc[
            0
        ]

        rowsC.append(
            (
                "Random slope\nprospective",
                r[
                    "beta"
                ],
                r[
                    "CI_low"
                ],
                r[
                    "CI_high"
                ],
            )
        )

    rowsC += [
        (
            "GEE\nprospective",
            *eff(
                "prospective",
                "GEE_exchangeable_robust"
            )
        ),
        (
            "Random intercept\n≥2-year gap",
            *eff(
                "gap_ge2y",
                "MixedLM_random_intercept"
            )
        ),
        (
            "GEE\n≥2-year gap",
            *eff(
                "gap_ge2y",
                "GEE_exchangeable_robust"
            )
        ),
    ]

    forest(
        axC,
        rowsC,
        [
            INK
        ]
        * len(
            rowsC
        )
    )

    for lab, b, lo, hi in rowsC:

        verify_lab = lab.replace("\n", ", ")
        check(
            (
                "C",
                verify_lab
            ),
            b
        )

        plot_rows.append(
            {
                "panel":
                    "C",

                "group":
                    lab,

                "estimate":
                    b,

                "ci_low":
                    lo,

                "ci_high":
                    hi,
            }
        )

    axC.set_xlabel(
        "LA remodeling × MRI time × APOE ε4\n"
        "(β, 95% CI)"
    )

    letter(
        axC,
        "C"
    )

    # ========================================================
    # D. Vascular / antihypertensive sensitivity
    # ========================================================

    v = need(
        "vascular"
    )

    v = v[
        v[
            "term"
        ]
        == "MRI_time_x_LA_x_APOE4"
    ].copy()

    # If a dedicated antihypertensive file exists, append it.
    # If script 161 wrote into the original vascular directory,
    # the M3/M4 rows are already present in v.
    if IN[
        "antihyp"
    ].exists():

        v2 = need(
            "antihyp"
        )

        v2 = v2[
            v2[
                "term"
            ]
            == "MRI_time_x_LA_x_APOE4"
        ]

        v = pd.concat(
            [
                v,
                v2
            ],
            ignore_index=True
        )

    # Remove exact duplicate rows if the same file was propagated
    # to both locations.
    keep_cols = [
        c
        for c in [
            "subset",
            "model",
            "term",
            "beta",
            "CI_low",
            "CI_high"
        ]
        if c in v.columns
    ]

    v = v.drop_duplicates(
        subset=keep_cols
    )

    order = [
        (
            "M0_minimal",
            "Age and sex"
        ),
        (
            "M1_early_vascular",
            "+ BMI, SBP, smoking,\n   diabetes"
        ),
        (
            "M2_early_vascular_plus_DBP",
            "+ DBP"
        ),
    ]

    # New matched-sample antihypertensive sensitivity from script 161.
    if (
        v[
            "model"
        ]
        == "M3_matched_no_HTNtx"
    ).any():

        order.append(
            (
                "M3_matched_no_HTNtx",
                "Matched sample,\n   no treatment adjustment"
            )
        )

    if (
        v[
            "model"
        ]
        == "M4_plus_HTNtx"
    ).any():

        order.append(
            (
                "M4_plus_HTNtx",
                "+ antihypertensive\n   treatment"
            )
        )

    # Backward compatibility only if an older dedicated output exists.
    elif (
        v[
            "model"
        ]
        == "M3_plus_antihypertensive"
    ).any():

        order.append(
            (
                "M3_plus_antihypertensive",
                "+ antihypertensive\n   treatment"
            )
        )

    rowsD = []
    colsD = []

    for subset, tag, c in (
        (
            "prospective",
            "",
            INK
        ),
        (
            "gap_ge2y",
            " (≥2-year gap)",
            GREY
        ),
    ):

        for m, lab in order:

            z = v[
                (
                    v[
                        "subset"
                    ]
                    == subset
                )
                &
                (
                    v[
                        "model"
                    ]
                    == m
                )
            ]

            if len(z) == 1:

                r = z.iloc[
                    0
                ]

                display_lab = (
                    lab + tag
                    if (
                        tag
                        and m
                        == "M0_minimal"
                    )
                    else lab
                )

                rowsD.append(
                    (
                        display_lab,
                        r[
                            "beta"
                        ],
                        r[
                            "CI_low"
                        ],
                        r[
                            "CI_high"
                        ],
                    )
                )

                colsD.append(
                    c
                )

                if subset == "prospective":

                    check(
                        (
                            "D",
                            m
                        ),
                        r[
                            "beta"
                        ]
                    )

                plot_rows.append(
                    {
                        "panel":
                            "D",

                        "group":
                            f"{subset}: {m}",

                        "estimate":
                            r[
                                "beta"
                            ],

                        "ci_low":
                            r[
                                "CI_low"
                            ],

                        "ci_high":
                            r[
                                "CI_high"
                            ],
                    }
                )

            elif (
                subset
                == "prospective"
                and m
                in (
                    "M0_minimal",
                    "M1_early_vascular",
                    "M2_early_vascular_plus_DBP",
                )
            ):

                sys.exit(
                    "Panel D: expected one row for "
                    f"{subset}/{m}, found {len(z)}"
                )

    forest(
        axD,
        rowsD,
        colsD
    )

    axD.set_xlabel(
        "LA remodeling × MRI time × APOE ε4\n"
        "(β, 95% CI)"
    )

    for ax in (
        axC,
        axD
    ):

        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(
                4
            )
        )

    axD.text(
        1.0,
        -0.30,
        "black: prospective; grey: ≥2-year gap",
        transform=axD.transAxes,
        ha="right",
        fontsize=5.8,
        color="#52606D"
    )

    letter(
        axD,
        "D"
    )

    # ========================================================
    # Save figure and verification/source data
    # ========================================================

    for ext, kw in (
        (
            "pdf",
            {}
        ),
        (
            "svg",
            {}
        ),
        (
            "png",
            {
                "dpi":
                    600
            }
        ),
    ):

        fig.savefig(
            out
            / f"Figure2.{ext}",
            bbox_inches="tight",
            **kw
        )

    pd.DataFrame(
        plot_rows
    ).to_csv(
        out
        / "Figure2_plot_data.csv",
        index=False
    )

    chk = pd.DataFrame(
        checks
    )

    chk.to_csv(
        out
        / "Figure2_verification.csv",
        index=False
    )

    print(
        "\nVERIFICATION"
    )

    print(
        chk.to_string(
            index=False
        )
    )

    if (
        a.strict
        and (
            chk.empty
            or not chk[
                "match"
            ].all()
        )
    ):

        sys.exit(
            "STRICT: plotted values differ from manuscript "
            "- see Figure2_verification.csv"
        )

    print(
        f"\nwritten to {out}"
    )


if __name__ == "__main__":
    main()
