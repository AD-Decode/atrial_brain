#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
import matplotlib.pyplot as plt

ROOT = Path("/data/qiallab/Framingham")

DATA = (
    ROOT
    / "data/longitudinal_derived/"
      "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

OUT = (
    ROOT
    / "results/longitudinal_heart_brain/"
      "temporal_prediction/APOE4_dose_sensitivity"
)

FIGDIR = (
    ROOT
    / "results/longitudinal_heart_brain/"
      "temporal_prediction/figures"
)

OUT.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(DATA, sep="\t")

# ============================================================
# EXACT PRIMARY LONGITUDINAL SAMPLE
# ============================================================

base = d[
    (d["la_dim_n"] == 3)
    & d["la_dim_slope"].notna()
    & d["APOE4_dose"].notna()
    & d["Lateralvent"].notna()
    & d["age6_c"].notna()
    & d["sex"].notna()
].copy()

n_lv = (
    base.groupby("shareid")["Lateralvent"]
    .count()
    .rename("n_lv")
)

base = base.merge(
    n_lv,
    left_on="shareid",
    right_index=True,
    how="left"
)

base = base[base["n_lv"] >= 2].copy()

# Match existing standardized longitudinal analysis
base["cardiac_z"] = (
    base["la_dim_slope"] - base["la_dim_slope"].mean()
) / base["la_dim_slope"].std(ddof=0)

base["brain_z"] = (
    base["Lateralvent"] - base["Lateralvent"].mean()
) / base["Lateralvent"].std(ddof=0)

base["APOE4_dose_int"] = base["APOE4_dose"].astype(int)

base["APOE4_dose_cat"] = pd.Categorical(
    base["APOE4_dose_int"],
    categories=[0, 1, 2],
    ordered=False
)

subsets = {
    "prospective": base[
        base["last_echo_to_first_mri_years"] > 0
    ].copy(),

    "gap_ge2y": base[
        base["last_echo_to_first_mri_years"] >= 2
    ].copy(),
}

formula = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * C(APOE4_dose_cat) "
    "+ age6_c + C(sex)"
)

rows = []
joint_rows = []

for subset_name, x in subsets.items():

    print("\n" + "=" * 80)
    print(subset_name.upper())
    print("=" * 80)

    model = smf.mixedlm(
        formula,
        data=x,
        groups=x["shareid"],
        re_formula="1"
    )

    res = model.fit(
        reml=False,
        method="lbfgs",
        maxiter=2000,
        disp=False
    )

    print("Converged:", res.converged)
    print("N subjects:", x["shareid"].nunique())
    print("N observations:", len(x))

    params = list(res.params.index)

    base_term = "mri_time_years:cardiac_z"

    dose_terms = {}

    for dose in [1, 2]:
        matches = [
            t for t in params
            if "mri_time_years" in t
            and "cardiac_z" in t
            and "APOE4_dose_cat" in t
            and f"T.{dose}" in t
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely identify dose {dose} interaction: {matches}"
            )

        dose_terms[dose] = matches[0]

    cov = res.cov_params()

    # Dose 0 simple slope
    b0 = float(res.params[base_term])
    v0 = float(cov.loc[base_term, base_term])

    for dose in [0, 1, 2]:

        if dose == 0:
            slope = b0
            var = v0

        else:
            term = dose_terms[dose]

            slope = (
                b0
                + float(res.params[term])
            )

            var = (
                cov.loc[base_term, base_term]
                + cov.loc[term, term]
                + 2 * cov.loc[base_term, term]
            )

        se = np.sqrt(max(float(var), 0))
        z = slope / se
        p = 2 * norm.sf(abs(z))

        n_subjects = (
            x.loc[
                x["APOE4_dose_int"] == dose,
                "shareid"
            ]
            .nunique()
        )

        rows.append({
            "subset": subset_name,
            "APOE4_dose": dose,
            "simple_slope": slope,
            "SE": se,
            "CI_low": slope - 1.96 * se,
            "CI_high": slope + 1.96 * se,
            "P": p,
            "N_subjects": int(n_subjects),
            "N_observations": int(
                (x["APOE4_dose_int"] == dose).sum()
            ),
            "converged": bool(res.converged),
        })

    # Joint 2-df test: are both categorical dose-modification terms zero?
    names = list(res.params.index)

    R = np.zeros((2, len(names)))

    R[0, names.index(dose_terms[1])] = 1
    R[1, names.index(dose_terms[2])] = 1

    wt = res.wald_test(R)

    joint_rows.append({
        "subset": subset_name,
        "test": "Joint categorical APOE4 dose interaction",
        "chi2": float(np.asarray(wt.statistic).squeeze()),
        "df": 2,
        "P": float(np.asarray(wt.pvalue).squeeze()),
        "N_subjects": int(x["shareid"].nunique()),
        "N_observations": int(len(x)),
    })

slopes = pd.DataFrame(rows)
joint = pd.DataFrame(joint_rows)

slopes.to_csv(
    OUT / "APOE4_categorical_dose_simple_slopes.tsv",
    sep="\t",
    index=False
)

joint.to_csv(
    OUT / "APOE4_categorical_dose_joint_test.tsv",
    sep="\t",
    index=False
)

print("\nCATEGORICAL DOSE SIMPLE SLOPES")
print(slopes.to_string(index=False))

print("\nJOINT CATEGORICAL DOSE TEST")
print(joint.to_string(index=False))

# ============================================================
# FIGURE 114
# Free categorical estimates: no forced linear dose spacing
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.8),
    sharey=True
)

specs = [
    ("prospective", "A  Prospective analysis"),
    ("gap_ge2y", "B  ≥2-year temporal gap"),
]

for ax, (subset_name, title) in zip(axes, specs):

    s = slopes[
        slopes["subset"] == subset_name
    ].sort_values("APOE4_dose")

    x = s["APOE4_dose"].to_numpy()
    y = s["simple_slope"].to_numpy()

    yerr = np.vstack([
        y - s["CI_low"].to_numpy(),
        s["CI_high"].to_numpy() - y,
    ])

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        capsize=4,
        linewidth=1.7,
        markersize=7
    )

    ax.plot(
        x,
        y,
        linestyle=":",
        linewidth=1.2
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        "0 ε4\nalleles",
        "1 ε4\nallele",
        "2 ε4\nalleles",
    ])

    ax.set_xlabel("APOE ε4 allele dose")
    ax.set_title(
        title,
        loc="left",
        fontweight="bold"
    )

    p_joint = float(
        joint.loc[
            joint["subset"] == subset_name,
            "P"
        ].iloc[0]
    )

    ax.text(
        0.03,
        0.97,
        f"Categorical dose interaction P={p_joint:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9
    )

axes[0].set_ylabel(
    "Effect of LA remodeling on annual\n"
    "ventricular-volume change"
)

fig.suptitle(
    "APOE ε4 categorical-dose sensitivity analysis",
    fontsize=14,
    fontweight="bold",
    y=1.02
)

fig.tight_layout()

for ext in ["png", "pdf", "svg"]:
    fig.savefig(
        FIGDIR / f"Figure114_SUPP_APOE4_categorical_dose.{ext}",
        dpi=300 if ext == "png" else None,
        bbox_inches="tight"
    )

plt.close(fig)

slopes.merge(
    joint[["subset", "P"]].rename(
        columns={"P": "joint_categorical_P"}
    ),
    on="subset",
    how="left"
).to_csv(
    FIGDIR / "Figure114_source_data.tsv",
    sep="\t",
    index=False
)

print("\nFIGURE 114 COMPLETE")
