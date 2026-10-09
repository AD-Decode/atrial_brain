#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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

# ============================================================
# LOAD
# ============================================================

d = pd.read_csv(DATA, sep="\t")

required = [
    "shareid",
    "mri_time_years",
    "Lateralvent",
    "la_dim_n",
    "la_dim_slope",
    "APOE4_dose",
    "age6_c",
    "sex",
    "last_echo_to_first_mri_years",
]

missing = [c for c in required if c not in d.columns]

if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# ============================================================
# EXACT PRIMARY LONGITUDINAL SAMPLE LOGIC
# ============================================================

base = d[
    (d["la_dim_n"] == 3)
    & d["la_dim_slope"].notna()
    & d["APOE4_dose"].notna()
    & d["Lateralvent"].notna()
    & d["age6_c"].notna()
    & d["sex"].notna()
].copy()

# At least two usable ventricular observations
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

# ============================================================
# STANDARDIZATION
#
# Match primary interpretation:
# cardiac_z = standardized LA trajectory
# brain_z   = standardized lateral ventricular volume
# ============================================================

base["cardiac_z"] = (
    base["la_dim_slope"]
    - base["la_dim_slope"].mean()
) / base["la_dim_slope"].std(ddof=0)

base["brain_z"] = (
    base["Lateralvent"]
    - base["Lateralvent"].mean()
) / base["Lateralvent"].std(ddof=0)

base["APOE4_dose"] = base["APOE4_dose"].astype(float)

# ============================================================
# ANALYSIS SUBSETS
# ============================================================

subsets = {
    "prospective": base[
        base["last_echo_to_first_mri_years"] > 0
    ].copy(),

    "gap_ge2y": base[
        base["last_echo_to_first_mri_years"] >= 2
    ].copy(),
}

# ============================================================
# MODEL
# ============================================================

formula = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_dose "
    "+ age6_c + C(sex)"
)

model_rows = []
slope_rows = []
count_rows = []

for subset_name, x in subsets.items():

    print("\n" + "=" * 80)
    print(subset_name.upper())
    print("=" * 80)

    # Counts
    u = x.drop_duplicates("shareid")

    counts = (
        u["APOE4_dose"]
        .value_counts()
        .sort_index()
    )

    for dose, n in counts.items():
        count_rows.append({
            "subset": subset_name,
            "APOE4_dose": int(dose),
            "N_subjects": int(n),
            "N_observations": int(
                (x["APOE4_dose"] == dose).sum()
            ),
        })

    print("Unique subjects:", u["shareid"].nunique())
    print(counts)

    # Random-intercept MixedLM, matching primary longitudinal model
    model = smf.mixedlm(
        formula,
        data=x,
        groups=x["shareid"],
        re_formula="1",
    )

    res = model.fit(
        reml=False,
        method="lbfgs",
        maxiter=2000,
        disp=False,
    )

    print("Converged:", res.converged)

    # --------------------------------------------------------
    # Formal ordinal allele-dose interaction
    # --------------------------------------------------------

    term = (
        "mri_time_years:"
        "cardiac_z:"
        "APOE4_dose"
    )

    beta = float(res.params[term])
    se = float(res.bse[term])
    p = float(res.pvalues[term])

    model_rows.append({
        "subset": subset_name,
        "term": term,
        "beta": beta,
        "SE": se,
        "CI_low": beta - 1.96 * se,
        "CI_high": beta + 1.96 * se,
        "P": p,
        "N_subjects": int(
            x["shareid"].nunique()
        ),
        "N_observations": int(len(x)),
        "converged": bool(res.converged),
    })

    # --------------------------------------------------------
    # Dose-specific simple slopes
    #
    # Effect of LA remodeling on annual ventricular change:
    #
    # beta(time × cardiac)
    # + dose * beta(time × cardiac × APOE4 dose)
    # --------------------------------------------------------

    b1_name = "mri_time_years:cardiac_z"
    b3_name = (
        "mri_time_years:"
        "cardiac_z:"
        "APOE4_dose"
    )

    b1 = float(res.params[b1_name])
    b3 = float(res.params[b3_name])

    cov = res.cov_params()

    for dose in [0, 1, 2]:

        slope = b1 + dose * b3

        var = (
            cov.loc[b1_name, b1_name]
            + (dose ** 2)
              * cov.loc[b3_name, b3_name]
            + 2 * dose
              * cov.loc[b1_name, b3_name]
        )

        slope_se = np.sqrt(max(var, 0))

        z = slope / slope_se
        # normal approximation
        from scipy.stats import norm
        slope_p = 2 * norm.sf(abs(z))

        slope_rows.append({
            "subset": subset_name,
            "APOE4_dose": dose,
            "simple_slope": slope,
            "SE": slope_se,
            "CI_low": slope - 1.96 * slope_se,
            "CI_high": slope + 1.96 * slope_se,
            "P": slope_p,
            "N_subjects": int(
                counts.get(float(dose), 0)
            ),
        })

# ============================================================
# SAVE TABLES
# ============================================================

model_df = pd.DataFrame(model_rows)
slope_df = pd.DataFrame(slope_rows)
count_df = pd.DataFrame(count_rows)

model_file = OUT / "APOE4_dose_interaction_results.tsv"
slope_file = OUT / "APOE4_dose_simple_slopes.tsv"
count_file = OUT / "APOE4_dose_sample_counts.tsv"

model_df.to_csv(
    model_file,
    sep="\t",
    index=False,
)

slope_df.to_csv(
    slope_file,
    sep="\t",
    index=False,
)

count_df.to_csv(
    count_file,
    sep="\t",
    index=False,
)

print("\nFORMAL DOSE INTERACTION")
print(model_df.to_string(index=False))

print("\nDOSE-SPECIFIC SIMPLE SLOPES")
print(slope_df.to_string(index=False))

# ============================================================
# SUPPLEMENTARY FIGURE 113
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.8),
    sharey=True,
)

panel_specs = [
    ("prospective", "A  Prospective analysis"),
    ("gap_ge2y", "B  ≥2-year temporal gap"),
]

for ax, (subset_name, title) in zip(
    axes,
    panel_specs
):

    s = slope_df[
        slope_df["subset"] == subset_name
    ].copy()

    s = s.sort_values("APOE4_dose")

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
        fmt="o-",
        capsize=4,
        linewidth=1.7,
        markersize=6,
    )

    ax.axhline(
        0,
        linewidth=1,
        linestyle="--",
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
        fontweight="bold",
    )

    # Add formal ordinal dose-interaction P value
    pval = float(
        model_df.loc[
            model_df["subset"] == subset_name,
            "P",
        ].iloc[0]
    )

    ax.text(
        0.03,
        0.97,
        f"Ordinal dose interaction P={pval:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

axes[0].set_ylabel(
    "Effect of LA remodeling on annual\n"
    "ventricular-volume change"
)

fig.suptitle(
    "APOE ε4 allele-dose sensitivity analysis",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

fig.tight_layout()

png = FIGDIR / "Figure113_SUPP_APOE4_dose_sensitivity.png"
pdf = FIGDIR / "Figure113_SUPP_APOE4_dose_sensitivity.pdf"
svg = FIGDIR / "Figure113_SUPP_APOE4_dose_sensitivity.svg"

fig.savefig(
    png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

fig.savefig(
    svg,
    bbox_inches="tight",
)

plt.close(fig)

# Combined source data for figure
source = slope_df.merge(
    model_df[
        [
            "subset",
            "beta",
            "SE",
            "CI_low",
            "CI_high",
            "P",
        ]
    ].rename(columns={
        "beta": "dose_interaction_beta",
        "SE": "dose_interaction_SE",
        "CI_low": "dose_interaction_CI_low",
        "CI_high": "dose_interaction_CI_high",
        "P": "dose_interaction_P",
    }),
    on="subset",
    how="left",
)

source.to_csv(
    FIGDIR / "Figure113_source_data.tsv",
    sep="\t",
    index=False,
)

print("\n" + "=" * 90)
print("FIGURE 113 COMPLETE")
print("=" * 90)

for f in [png, pdf, svg]:
    print(f)

print(
    FIGDIR
    / "Figure113_source_data.tsv"
)
