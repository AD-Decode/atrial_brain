#!/usr/bin/env python3
"""
New Figure 5 - APOE e4 modifies the association between left atrial volume and left middle-occipital
cortical-thickness variability. Combines the former Figure 6 (A, B) with the LAVI part of the former
Figure 5 (C); the ventricular allele-dose results remain in Table 3.

  A  APOE e4-stratified LAVI slopes (model of 39_atrial_APOE4_exact.py, Exam-8 primary sample).
     Carrier slope = beta_LAVI + beta_interaction with HC3 covariance.
  B  Interaction after adding LV mass index, LVEF, or both (model of 42_atrial_independence_LV.py,
     common complete-case sample standardized once).
  C  Genotype-specific LAVI slopes in e3/e3, e3/e4, e4/e4 (sample of 51_APOE_genotype_dose_validation.py)
     with the per-allele interaction printed; e4/e4 marked as imprecise.

All three use: outcome lh_G_occipital_middle_tksd, exposure LAVI_max, covariates age at MRI, sex,
|CMR-MRI interval| (abs_years_exam_mri), BMI, SBP, current smoking, diabetes (codes 1/2), OLS + HC3,
z-scores with ddof=0 (as in 39/42/51). Every value is checked against Table 4 / Table 3 / Results 3.7.

Outputs: Figure5.{pdf,svg,png}, Figure5_plot_data.csv, Figure5_verification.csv
Usage: python make_Figure5_LAVI.py [--strict] [--outdir DIR]
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
INFILE = ROOT / "results/neurocardiac_metadata_with_atria.csv"
OUTCOME, ATRIAL = "lh_G_occipital_middle_tksd", "LAVI_max"
COVS = ["age_at_mri", "_sex", "abs_years_exam_mri", "BMI_nearest_exam", "SBP_nearest_exam",
        "current_smoker_nearest_exam", "diabetes_history_any"]
Z_COVS = {"age_at_mri": "age_z", "abs_years_exam_mri": "interval_z", "BMI_nearest_exam": "BMI_z", "SBP_nearest_exam": "SBP_z"}
BIN_COVS = ["_sex", "current_smoker_nearest_exam", "diabetes_history_any"]

VERIFY = {  # published values (Table 4, Results 3.7, Table 3)
    "A_N": 451, "A_noncarriers": 355, "A_carriers": 96,
    "A_nc_beta": 0.045, "A_car_beta": -0.500, "A_int_beta": -0.545, "A_int_ci_low": -0.792, "A_int_ci_high": -0.298,
    "B_baseline": -0.545, "B_LVMI": -0.549, "B_LVEF": -0.545, "B_both": -0.549,
    "C_N": 387, "C_e33": 296, "C_e34": 86, "C_e44": 5,
    "C_dose_beta": -0.473, "C_dose_ci_low": -0.722, "C_dose_ci_high": -0.225,
}
TOL_BETA = 0.002

_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in _AVAIL), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "font.size": 7, "axes.linewidth": 0.7, "pdf.fonttype": 42, "svg.fonttype": "none"})
NC, CA, INK, GREY = "#3B6FB6", "#D9822B", "#1F2933", "#7B8794"


def fmt_p(p):
    """P value in manuscript style, e.g. 1.56×10^-5 (mathtext, so no special glyphs are needed)."""
    if p >= 0.001:
        return f"{p:.2g}"
    m, e = f"{p:.2e}".split("e")
    return f"{m}×10$^{{{int(e)}}}$"


def num(x):
    return pd.to_numeric(x, errors="coerce")


def z(x):
    x = num(x); return (x - x.mean()) / x.std(ddof=0)


def load():
    df = pd.read_csv(INFILE, low_memory=False)
    df = df[num(df["nearest_exam"]) == 8].copy()                       # primary atrial sample
    df["_sex"] = num(df["sex_clinical"])
    dm = num(df["diabetes_history_nearest_exam"])
    df["diabetes_history_any"] = np.where(dm == 0, 0.0, np.where(dm.isin([1, 2]), 1.0, np.nan))
    if "height_cm" in df.columns and "weight_kg" in df.columns:
        h, w = num(df["height_cm"]), num(df["weight_kg"])
    else:
        h, w = num(df["height_in"]) * 2.54, num(df["weight_lb"]) * 0.45359237
    df["LV_MASSi_recomputed"] = num(df["LV_MASS"]) / np.sqrt(h * w / 3600.0)   # as in 42
    if "APOE_genotype" in df.columns:
        df["APOE_genotype"] = df["APOE_genotype"].astype("string").str.strip().str.upper()
    return df


def design(d, exposure_cols):
    for c, n in Z_COVS.items():
        d[n] = z(d[c])
    return list(exposure_cols) + list(Z_COVS.values()) + BIN_COVS


def lincomb(fit, w):
    """estimate and 95% CI of sum(w[k]*beta[k]) using the HC3 covariance."""
    names = list(w); b = fit.params[names].to_numpy(); v = fit.cov_params().loc[names, names].to_numpy()
    c = np.array([w[k] for k in names]); est = c @ b; se = np.sqrt(c @ v @ c)
    return est, est - 1.96 * se, est + 1.96 * se


def panel_a(df):
    d = df[[OUTCOME, ATRIAL, "APOE4_carrier"] + COVS].apply(num).dropna().copy()
    d["y_z"], d["a_z"] = z(d[OUTCOME]), z(d[ATRIAL]); d["a_x_e4"] = d["a_z"] * d["APOE4_carrier"]
    X = design(d, ["a_z", "APOE4_carrier", "a_x_e4"])
    f = sm.OLS(d["y_z"], sm.add_constant(d[X].astype(float))).fit(cov_type="HC3")
    return {"N": len(d), "noncarriers": int((d.APOE4_carrier == 0).sum()), "carriers": int((d.APOE4_carrier == 1).sum()),
            "nc": lincomb(f, {"a_z": 1}), "car": lincomb(f, {"a_z": 1, "a_x_e4": 1}),
            "int": lincomb(f, {"a_x_e4": 1}), "p_int": f.pvalues["a_x_e4"]}


def panel_b(df):
    common = [OUTCOME, ATRIAL, "APOE4_carrier", "LV_MASSi_recomputed", "LVEF"] + COVS
    d = df[common].apply(num).dropna().copy()
    d["y_z"], d["a_z"] = z(d[OUTCOME]), z(d[ATRIAL]); d["a_x_e4"] = d["a_z"] * d["APOE4_carrier"]
    d["LVMI_z"], d["LVEF_z"] = z(d["LV_MASSi_recomputed"]), z(d["LVEF"])
    base = design(d, ["a_z", "APOE4_carrier", "a_x_e4"])
    out = []
    for name, extra in (("Baseline", []), ("+ LV mass index", ["LVMI_z"]), ("+ LVEF", ["LVEF_z"]),
                        ("+ LV mass index + LVEF", ["LVMI_z", "LVEF_z"])):
        f = sm.OLS(d["y_z"], sm.add_constant(d[base + extra].astype(float))).fit(cov_type="HC3")
        out.append((name, *lincomb(f, {"a_x_e4": 1}), f.pvalues["a_x_e4"], len(d)))
    return out


def panel_c(df):
    dose_map = {"E3/E3": 0, "E3/E4": 1, "E4/E4": 2}
    d = df[df["APOE_genotype"].isin(dose_map)][[OUTCOME, ATRIAL, "APOE_genotype"] + COVS].copy()
    for c in [OUTCOME, ATRIAL] + COVS:
        d[c] = num(d[c])
    d = d.dropna().copy()
    d["dose"] = d["APOE_genotype"].map(dose_map).astype(float)
    d["y_z"], d["a_z"] = z(d[OUTCOME]), z(d[ATRIAL])
    cov = design(d, [])
    # per-allele trend (as in 51)
    d["a_x_dose"] = d["a_z"] * d["dose"]
    fd = sm.OLS(d["y_z"], sm.add_constant(d[["a_z", "dose", "a_x_dose"] + cov].astype(float))).fit(cov_type="HC3")
    dose = lincomb(fd, {"a_x_dose": 1})
    # genotype-specific slopes (categorical, e3/e3 reference)
    for g in ("E3/E4", "E4/E4"):
        k = g.replace("/", "")
        d[k] = (d["APOE_genotype"] == g).astype(float); d["a_x_" + k] = d["a_z"] * d[k]
    fg = sm.OLS(d["y_z"], sm.add_constant(d[["a_z", "E3E4", "E4E4", "a_x_E3E4", "a_x_E4E4"] + cov].astype(float))).fit(cov_type="HC3")
    slopes = [("ε3/ε3", int((d.dose == 0).sum()), *lincomb(fg, {"a_z": 1})),
              ("ε3/ε4", int((d.dose == 1).sum()), *lincomb(fg, {"a_z": 1, "a_x_E3E4": 1})),
              ("ε4/ε4", int((d.dose == 2).sum()), *lincomb(fg, {"a_z": 1, "a_x_E4E4": 1}))]
    return {"N": len(d), "dose": dose, "p_dose": fd.pvalues["a_x_dose"], "slopes": slopes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--width", type=float, default=8.0, help="figure width in inches (journal double column is ~7.2)")
    ap.add_argument("--outdir", default=str(ROOT / "results/final_rebuild_20261007/Figure5_LAVI"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df = load()
    A, B, C = panel_a(df), panel_b(df), panel_c(df)

    got = {"A_N": A["N"], "A_noncarriers": A["noncarriers"], "A_carriers": A["carriers"],
           "A_nc_beta": A["nc"][0], "A_car_beta": A["car"][0], "A_int_beta": A["int"][0],
           "A_int_ci_low": A["int"][1], "A_int_ci_high": A["int"][2],
           "B_baseline": B[0][1], "B_LVMI": B[1][1], "B_LVEF": B[2][1], "B_both": B[3][1],
           "C_N": C["N"], "C_e33": C["slopes"][0][1], "C_e34": C["slopes"][1][1], "C_e44": C["slopes"][2][1],
           "C_dose_beta": C["dose"][0], "C_dose_ci_low": C["dose"][1], "C_dose_ci_high": C["dose"][2]}
    chk = pd.DataFrame([{"item": k, "published": v, "computed": round(got[k], 4) if isinstance(got[k], float) else got[k],
                         "match": (abs(got[k] - v) <= TOL_BETA) if isinstance(v, float) else got[k] == v}
                        for k, v in VERIFY.items()])

    fig, axs = plt.subplots(1, 3, figsize=(a.width, 3.0), gridspec_kw=dict(wspace=0.75, width_ratios=[0.8, 1.15, 1.0]))
    axA, axB, axC = axs
    # A
    for i, (k, c, mk, lab) in enumerate((("nc", NC, "o", f"Non-\ncarriers\n(n = {A['noncarriers']})"),
                                         ("car", CA, "s", f"ε4\ncarriers\n(n = {A['carriers']})"))):
        b, lo, hi = A[k]
        axA.errorbar(i, b, yerr=[[b - lo], [hi - b]], fmt=mk, ms=5, color=c, capsize=3, lw=1.3)
    axA.axhline(0, color=GREY, lw=0.7, ls="--")
    axA.set_xlim(-0.7, 1.7); axA.set_xticks([0, 1])
    axA.set_xticklabels([f"Non-\ncarriers\nn = {A['noncarriers']}", f"ε4\ncarriers\nn = {A['carriers']}"], fontsize=6)
    axA.set_ylabel("LAVI slope on thickness\nvariability (β, 95% CI)")
    axA.set_title(f"Interaction β = {A['int'][0]:.3f}\nP = {fmt_p(A['p_int'])}", fontsize=6.2, pad=3)
    # B
    yb = np.arange(len(B))[::-1]
    for yi, (name, b, lo, hi, p, n) in zip(yb, B):
        axB.errorbar(b, yi, xerr=[[b - lo], [hi - b]], fmt="o", ms=4, color=CA, capsize=2.5, lw=1.2)
    axB.axvline(0, color=GREY, lw=0.7, ls="--")
    axB.set_yticks(yb); axB.set_yticklabels([f"{r[0]}\nP = {fmt_p(r[4])}" for r in B], fontsize=6)
    axB.tick_params(axis="y", length=0)
    axB.set_ylim(-0.6, len(B) - 0.4)
    axB.set_xlabel("LAVI × APOE ε4 interaction\n(β, 95% CI)")
    # C
    colsC = [NC, CA, CA]; mk = ["o", "s", "D"]
    inf = [x for x in C["slopes"] if x[0] != "ε4/ε4"]
    lo_lim = min(x[3] for x in inf) - 0.25
    hi_lim = max(x[4] for x in inf) + 0.25
    for i, ((g, n, b, lo, hi), c, m) in enumerate(zip(C["slopes"], colsC, mk)):
        if g == "ε4/ε4":                     # n is tiny: show the point, truncate the interval at the axis
            clo, chi = max(lo, lo_lim), min(hi, hi_lim)
            axC.plot([i, i], [clo, chi], color=c, alpha=0.45, lw=1.2)
            axC.plot(i, b, m, ms=4.5, color=c, alpha=0.45)
            if lo < lo_lim:
                axC.annotate("", xy=(i, lo_lim), xytext=(i, lo_lim + 0.12),
                             arrowprops=dict(arrowstyle="-|>", color=c, alpha=0.6, lw=1.0))
            if hi > hi_lim:
                axC.annotate("", xy=(i, hi_lim), xytext=(i, hi_lim - 0.12),
                             arrowprops=dict(arrowstyle="-|>", color=c, alpha=0.6, lw=1.0))
            continue
        axC.errorbar(i, b, yerr=[[b - lo], [hi - b]], fmt=m, ms=4.5, color=c, capsize=3, lw=1.2)
    axC.set_ylim(lo_lim, hi_lim)
    axC.axhline(0, color=GREY, lw=0.7, ls="--")
    axC.set_xlim(-0.7, 2.7); axC.set_xticks([0, 1, 2])
    axC.set_xticklabels([f"{g}\nn = {n}" for g, n, *_ in C["slopes"]], fontsize=6)
    axC.set_ylabel("LAVI slope (β, 95% CI)")
    db, dlo, dhi = C["dose"]
    axC.set_title(f"Per ε4 allele: β = {db:.3f}\n(95% CI {dlo:.3f} to {dhi:.3f})", fontsize=6.2, pad=3)
    axC.text(2.0, hi_lim, "CI truncated", ha="center", va="top", fontsize=5.2, color=GREY)
    for ax in axs:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bbs = [ax.get_tightbbox(r).transformed(fig.transFigure.inverted()) for ax in axs]
    ytop = max(bb.y1 for bb in bbs) + 0.01          # one baseline for all panel letters
    for bb, L in zip(bbs, "ABC"):
        fig.text(bb.x0, ytop, L, fontsize=10, fontweight="bold", ha="left", va="bottom")

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(out / f"Figure5.{ext}", bbox_inches="tight", **kw)
    rows = [{"panel": "A", "group": g, "estimate": A[k][0], "ci_low": A[k][1], "ci_high": A[k][2], "N": A["N"]}
            for g, k in (("noncarriers", "nc"), ("carriers", "car"), ("interaction", "int"))]
    rows += [{"panel": "B", "group": n, "estimate": b, "ci_low": lo, "ci_high": hi, "P": p, "N": nn} for n, b, lo, hi, p, nn in B]
    rows += [{"panel": "C", "group": g, "estimate": b, "ci_low": lo, "ci_high": hi, "N": n} for g, n, b, lo, hi in C["slopes"]]
    rows += [{"panel": "C", "group": "per-allele interaction", "estimate": db, "ci_low": dlo, "ci_high": dhi, "P": C["p_dose"], "N": C["N"]}]
    pd.DataFrame(rows).to_csv(out / "Figure5_plot_data.csv", index=False)
    chk.to_csv(out / "Figure5_verification.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nVERIFICATION (Table 4 / Results 3.7 / Table 3)\n" + chk.to_string(index=False))
    if a.strict and not chk["match"].all():
        sys.exit("STRICT: values differ from the manuscript - see Figure5_verification.csv")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
