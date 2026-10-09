#!/usr/bin/env python3
"""
Figure 4 - Sensitivity of the regional ventricular interactions to the CMR-brain MRI interval.

Replaces 147 + 148 (whose archived figure file was 0 bytes). The analysis is identical to 147:
  * the four M3 models of 32_tier2_corrected_scaling.py (ICV-normalized volumes; total cortical
    area for surface area; age, sex, |CMR-MRI interval|, BMI, SBP, smoking, diabetes 1/2; APOE e4;
    cardiac x APOE e4; OLS with HC3);
  * full cohort, interval <= 3 years, interval <= 2 years (abs_delta_years);
  * FIXED scaling: means/SDs (ddof=1) of every continuous variable are taken once from the
    full-cohort complete-case sample of each model and applied unchanged to the restricted samples,
    so interaction coefficients share one scale.
Every estimate is checked against Results 3.5 (Figure4_verification.csv; --strict stops on mismatch).

Outputs: Figure4.{pdf,svg,png}, Figure4_plot_data.csv, Figure4_scaling_constants.csv,
         Figure4_verification.csv (summary statistics only).
Usage: python make_Figure4.py [--strict] [--outdir DIR]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path("/data/qiallab/Framingham")
DATAFILE = ROOT / "results/neurocardiac_metadata_analysis_ready_metabolic.csv"

TARGETS = [  # panel order
    ("LVEF", "lh_middletemporal_vol", "volume", "LVEF — left middle temporal volume"),
    ("LVESVi", "rh_superiorfrontal_area", "surface_area", "LVESVi — right superior frontal surface area"),
    ("LV_MASSi", "rh_lingual_vol", "volume", "LV mass index — right lingual volume"),
    ("LV_MASSi", "rh_lingual_area", "surface_area", "LV mass index — right lingual surface area"),
]
SAMPLES = [("Full cohort", None), ("≤3 years", 3.0), ("≤2 years", 2.0)]
EXPECTED_N = {"Full cohort": 765, "≤3 years": 418, "≤2 years": 185}

# Results 3.5: interaction beta (and CI where printed)
VERIFY = {
    ("lh_middletemporal_vol", "Full cohort"): (0.233, 0.087, 0.380),
    ("lh_middletemporal_vol", "≤3 years"): (0.165, -0.022, 0.351),
    ("lh_middletemporal_vol", "≤2 years"): (0.331, -0.007, 0.669),
    ("rh_superiorfrontal_area", "Full cohort"): (-0.170, -0.260, -0.081),
    ("rh_superiorfrontal_area", "≤3 years"): (-0.246, -0.360, -0.131),
    ("rh_superiorfrontal_area", "≤2 years"): (-0.296, -0.480, -0.113),
    ("rh_lingual_vol", "Full cohort"): (0.305, 0.144, 0.467),
    ("rh_lingual_vol", "≤3 years"): (0.227, None, None),
    ("rh_lingual_vol", "≤2 years"): (-0.032, None, None),
    ("rh_lingual_area", "Full cohort"): (0.226, 0.095, 0.357),
    ("rh_lingual_area", "≤3 years"): (0.195, None, None),
    ("rh_lingual_area", "≤2 years"): (-0.021, None, None),
}
TOL = 0.002

_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in _AVAIL), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "font.size": 7, "axes.linewidth": 0.7,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
SHADE = {"Full cohort": "#1F2933", "≤3 years": "#52606D", "≤2 years": "#9AA5B1"}
MARK = {"Full cohort": "o", "≤3 years": "s", "≤2 years": "^"}


def num(x):
    return pd.to_numeric(x, errors="coerce")


def prepare(df):                                     # identical to 32 / 147
    h = num(df["height_in"]) * 2.54; w = num(df["weight_lb"]) * 0.45359237
    df["BSA_m2"] = np.sqrt(h * w / 3600.0)
    for raw, idx in (("LVEDV", "LVEDVi"), ("LVESV", "LVESVi"), ("LV_MASS", "LV_MASSi")):
        df[idx] = num(df[raw]) / df["BSA_m2"]
    df["TotalCorticalSurfaceArea"] = num(df["lh_WhiteSurfArea_DesKil_area"]) + num(df["rh_WhiteSurfArea_DesKil_area"])
    df["_sex"] = num(df["sex_clinical"])
    dm = num(df["diabetes_history_nearest_exam"])
    df["diabetes_history_any"] = np.where(dm.isna(), np.nan, dm.isin([1, 2]).astype(float))
    return df


def analysis_data(df, heart, region, metric):
    y = num(df[region])
    if metric == "volume":
        y = y / num(df["IntraCranialVol"])
    d = pd.DataFrame({"y": y, "heart": num(df[heart]), "age": num(df["age_at_mri"]), "sex": num(df["_sex"]),
                      "interval": num(df["abs_delta_years"]), "BMI": num(df["BMI_nearest_exam"]),
                      "SBP": num(df["SBP_nearest_exam"]), "smoking": num(df["current_smoker_nearest_exam"]),
                      "diabetes": num(df["diabetes_history_any"]), "APOE4": num(df["APOE4_carrier"]),
                      "total_area": num(df["TotalCorticalSurfaceArea"])})
    need = ["y", "heart", "age", "sex", "interval", "BMI", "SBP", "smoking", "diabetes", "APOE4"]
    if metric == "surface_area":
        need.append("total_area")
    return d[need].dropna().copy()


def fit(sub, scales, metric):
    s = sub.copy()
    for v, (m, sd) in scales.items():
        s[v + "_z"] = (s[v] - m) / sd
    s["heart_x_APOE4"] = s["heart_z"] * s["APOE4"]
    x = ["heart_z", "age_z", "sex", "interval_z", "BMI_z", "SBP_z", "smoking", "diabetes", "APOE4", "heart_x_APOE4"]
    if metric == "surface_area":
        x.insert(4, "total_area_z")
    f = sm.OLS(s["y_z"], sm.add_constant(s[x])).fit(cov_type="HC3")
    b, se = f.params["heart_x_APOE4"], f.bse["heart_x_APOE4"]
    return b, b - 1.96 * se, b + 1.96 * se, f.pvalues["heart_x_APOE4"], len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--outdir", default=str(ROOT / "results/final_rebuild_20261007/Figure4"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    df = prepare(pd.read_csv(DATAFILE, low_memory=False))
    rows, scale_rows, checks = [], [], []
    for heart, region, metric, label in TARGETS:
        full = analysis_data(df, heart, region, metric)
        cont = ["y", "heart", "age", "interval", "BMI", "SBP"] + (["total_area"] if metric == "surface_area" else [])
        scales = {v: (float(full[v].mean()), float(full[v].std())) for v in cont}   # ddof=1, as in 32
        scale_rows += [{"region": region, "cardiac": heart, "variable": v, "mean": m, "sd": sd} for v, (m, sd) in scales.items()]
        for name, cut in SAMPLES:
            sub = full if cut is None else full[full["interval"] <= cut]
            b, lo, hi, p, n = fit(sub, scales, metric)
            rows.append({"label": label, "region": region, "cardiac": heart, "sample": name, "N": n,
                         "beta_interaction": b, "ci_low": lo, "ci_high": hi, "P": p})
            exp = VERIFY[(region, name)]
            for q, got, e in zip(("beta", "ci_low", "ci_high"), (b, lo, hi), exp):
                if e is not None:
                    checks.append({"region": region, "sample": name, "quantity": q, "published": e,
                                   "computed": round(got, 4), "match": abs(got - e) <= TOL})
            checks.append({"region": region, "sample": name, "quantity": "N", "published": EXPECTED_N[name],
                           "computed": n, "match": n == EXPECTED_N[name]})
    R = pd.DataFrame(rows)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6), gridspec_kw=dict(hspace=0.55, wspace=0.45))
    for ax, (heart, region, metric, label), letter in zip(axs.ravel(), TARGETS, "ABCD"):
        z = R[R["region"] == region].reset_index(drop=True)
        y = np.arange(len(z))[::-1]
        for yi, (_, r) in zip(y, z.iterrows()):
            ax.errorbar(r["beta_interaction"], yi, xerr=[[r["beta_interaction"] - r["ci_low"]], [r["ci_high"] - r["beta_interaction"]]],
                        fmt=MARK[r["sample"]], ms=4.2, color=SHADE[r["sample"]], capsize=2.5, lw=1.2)
        ax.axvline(0, color="#7B8794", lw=0.7, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([f"{s} (N = {n})" for s, n in zip(z["sample"], z["N"])])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Cardiac × APOE ε4 interaction (β, 95% CI)")
        ax.set_title(label, fontsize=7, loc="left", pad=4)
        ax.text(-0.42, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.6, len(z) - 0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(out / f"Figure4.{ext}", bbox_inches="tight", **kw)
    R.to_csv(out / "Figure4_plot_data.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(out / "Figure4_scaling_constants.csv", index=False)
    chk = pd.DataFrame(checks); chk.to_csv(out / "Figure4_verification.csv", index=False)
    print(R.round(4).to_string(index=False))
    print("\nVERIFICATION (Results 3.5)\n" + chk.to_string(index=False))
    if a.strict and not chk["match"].all():
        sys.exit("STRICT: values differ from Results 3.5 - see Figure4_verification.csv")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
