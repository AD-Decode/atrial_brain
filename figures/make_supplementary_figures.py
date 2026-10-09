#!/usr/bin/env python3
"""
Supplementary Figures S1-S4 (S5 = AHBA enrichment, drawn by script 89; S6 = make_FigureS_APOE_dose.py).

  S1  Alternative morphometric specifications of the four regional ventricular interactions
      (refit here with the make_Figure3 data preparation): primary specification vs
      volume -> raw volume with intracranial volume as a covariate; surface area -> raw area without
      the total-cortical-area covariate.
  S2  Plasma-biomarker adjustment (Aβ42/40, p-tau181, GFAP, NfL, combinations) of all four interactions,
      matched complete-case baseline vs adjusted - from 40_APOE_AD_completecase_proper_CI.py output.
  S3  Atrial fibrillation/flutter: original, + AF history, excluding AF - from 61_AF_flutter_sensitivity.py.
  S4  HbA1c x APOE e4 -> LVEDVi (exploratory): HbA1c slopes by APOE e4 status and the interaction across
      models M0-M3 - from 56_validate_LVEDVi_APOE4_HbA1c.py. Summary-level only (no participant scatter).

All panels are summary statistics. Each figure is checked against the manuscript/supplement
(FigureS_verification.csv; --strict stops on mismatch). Source files that moved are found by name under
results/.

Usage: python make_supplementary_figures.py [--strict] [--outdir DIR] [--only S1 S3]
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path("/data/qiallab/Framingham")
R = ROOT / "results"
SRC = {
    "S2": R / "APOE_AD_biomarker_completecase/APOE_AD_biomarker_completecase_proper_CI.csv",
    "S3": R / "AF_flutter_sensitivity/AF_flutter_primary4_sensitivity.csv",
    "S4": R / "LVEDVi_APOE4_HbA1c_validation/LVEDVi_APOE4_HbA1c_validation_models.csv",
}
ASSOC = [  # region, cardiac, metric, label (order of Table 2)
    ("lh_middletemporal_vol", "LVEF", "volume", "LVEF — left middle temporal volume"),
    ("rh_superiorfrontal_area", "LVESVi", "surface_area", "LVESVi — right superior frontal surface area"),
    ("rh_lingual_vol", "LV_MASSi", "volume", "LV mass index — right lingual volume"),
    ("rh_lingual_area", "LV_MASSi", "surface_area", "LV mass index — right lingual surface area"),
]
TABLE2 = {"lh_middletemporal_vol": 0.233, "rh_superiorfrontal_area": -0.170, "rh_lingual_vol": 0.305, "rh_lingual_area": 0.226}
S16 = {"int": (-2.89, -5.21, -0.56), "nc": (-0.17, -1.55, 1.20), "car": (-3.06, -4.94, -1.18)}   # Table S16, M3
TOL = 0.002

_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in _AVAIL), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "font.size": 7, "axes.linewidth": 0.7, "pdf.fonttype": 42, "svg.fonttype": "none"})
NC, CA, INK, GREY = "#3B6FB6", "#D9822B", "#1F2933", "#7B8794"
CHECKS = []


def check(fig, item, published, got, tol=TOL):
    CHECKS.append({"figure": fig, "item": item, "published": published, "computed": round(float(got), 4),
                   "match": abs(float(got) - published) <= tol})


def locate(p):
    p = Path(p)
    if p.exists():
        return p
    hits = sorted(R.rglob(p.name), key=lambda x: x.stat().st_mtime)
    hits = [h for h in hits if "final_rebuild" not in str(h)]
    if not hits:
        sys.exit(f"Source not found: {p.name} (looked under {R})")
    print(f"   located {p.name} -> {hits[-1]}")
    return hits[-1]


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    sys.exit(f"None of {names} in columns {list(df.columns)}")


def forest(ax, rows, colors, markers=None):
    """rows: (label, beta, lo, hi) top-to-bottom"""
    y = np.arange(len(rows))[::-1]
    for i, (yi, (lab, b, lo, hi)) in enumerate(zip(y, rows)):
        ax.errorbar(b, yi, xerr=[[b - lo], [hi - b]], fmt=(markers[i] if markers else "o"), ms=3.8,
                    color=colors[i], capsize=2, lw=1.0)
    ax.axvline(0, color=GREY, lw=0.7, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def letters(fig, axs):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for ax, L in zip(axs, "ABCDEFGH"):
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        fig.text(bb.x0, bb.y1 + 0.005, L, fontsize=10, fontweight="bold", va="bottom")


def save(fig, out, name):
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


# ------------------------------------------------------------------ S1
def figure_S1(out, figdir):
    spec = importlib.util.spec_from_file_location("mf3", figdir / "make_Figure3.py")
    F3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(F3)
    df = F3.prepare(pd.read_csv(F3.DATAFILE, low_memory=False))

    def fit(heart, region, metric, variant):
        y = pd.to_numeric(df[region], errors="coerce")
        icv = pd.to_numeric(df["IntraCranialVol"], errors="coerce")
        if metric == "volume" and variant == "primary":
            y = y / icv
        d = pd.DataFrame({"y": y, "heart": df[heart], "age": df["age_at_mri"], "sex": df["_sex"], "interval": df["abs_delta_years"],
                          "BMI": df["BMI_nearest_exam"], "SBP": df["SBP_nearest_exam"], "smoking": df["current_smoker_nearest_exam"],
                          "diabetes": df["diabetes_history_any"], "APOE4": df["APOE4_carrier"],
                          "total_area": df["TotalCorticalSurfaceArea"], "icv": icv}).apply(pd.to_numeric, errors="coerce")
        need = ["y", "heart", "age", "sex", "interval", "BMI", "SBP", "smoking", "diabetes", "APOE4"]
        extra = []
        if metric == "surface_area" and variant == "primary":
            need.append("total_area"); extra = ["total_area"]
        if metric == "volume" and variant == "alternative":
            need.append("icv"); extra = ["icv"]
        d = d[need].dropna()
        for c in ["y", "heart", "age", "interval", "BMI", "SBP"] + extra:
            d[c + "_z"] = F3.z(d[c])
        d["hx"] = d["heart_z"] * d["APOE4"]
        X = ["heart_z", "age_z", "sex", "interval_z", "BMI_z", "SBP_z", "smoking", "diabetes", "APOE4", "hx"] + [c + "_z" for c in extra]
        f = sm.OLS(d["y_z"], sm.add_constant(d[X])).fit(cov_type="HC3")
        b, se = f.params["hx"], f.bse["hx"]
        return b, b - 1.96 * se, b + 1.96 * se, len(d)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 3.8), gridspec_kw=dict(hspace=0.75, wspace=0.75))
    rows_out = []
    for ax, (region, heart, metric, label) in zip(axs.ravel(), ASSOC):
        p = fit(heart, region, metric, "primary"); q = fit(heart, region, metric, "alternative")
        check("S1", f"{region} primary = Table 2", TABLE2[region], p[0])
        alt = "Raw volume + ICV covariate" if metric == "volume" else "Raw area, no total-area covariate"
        prim = "ICV-normalized volume (primary)" if metric == "volume" else "Area + total-area covariate (primary)"
        forest(ax, [(f"{prim}\nN = {p[3]}", *p[:3]), (f"{alt}\nN = {q[3]}", *q[:3])], [INK, GREY], ["o", "s"])
        ax.set_title(label, fontsize=6.5, loc="left"); ax.set_xlabel("Cardiac × APOE ε4 interaction (β, 95% CI)", fontsize=6)
        rows_out += [{"association": label, "specification": prim, "beta": p[0], "ci_low": p[1], "ci_high": p[2], "N": p[3]},
                     {"association": label, "specification": alt, "beta": q[0], "ci_low": q[1], "ci_high": q[2], "N": q[3]}]
    letters(fig, axs.ravel()); save(fig, out, "FigureS1")
    pd.DataFrame(rows_out).to_csv(out / "FigureS1_plot_data.csv", index=False)


# ------------------------------------------------------------------ S2
SPEC_LABEL = {"AMYLOID": "Aβ42/40", "AB": "Aβ42/40", "PTAU": "p-tau181", "TAU": "p-tau181", "GFAP": "GFAP", "NFL": "NfL"}


def spec_label(s):
    s = str(s).upper()
    if "ALL" in s or "FULL" in s:
        return "+ Aβ42/40 + p-tau181 + GFAP + NfL"
    found = []
    for k, v in SPEC_LABEL.items():
        if k in s and v not in found:
            found.append(v)
    return "+ " + " + ".join(found) if found else str(s)


def figure_S2(out):
    d = pd.read_csv(locate(SRC["S2"]))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.5, wspace=0.95))
    rows_out = []
    for ax, (region, heart, metric, label) in zip(axs.ravel(), ASSOC):
        z = d[(d["region"] == region) & (d["cardiac"] == heart)]
        rows, cols, mk = [], [], []
        for _, r in z.iterrows():
            lab = spec_label(r["spec"])
            rows.append((f"{lab} (N = {int(r['N'])})", r["base_beta_interaction"], r["base_CI_interaction_low"], r["base_CI_interaction_high"]))
            rows.append(("adjusted", r[pick(z, "adjusted_beta_interaction")], r[pick(z, "adjusted_CI_interaction_low")], r[pick(z, "adjusted_CI_interaction_high")]))
            cols += [GREY, CA]; mk += ["o", "s"]
            rows_out.append({"association": label, "adjustment": lab, "N": int(r["N"]), "baseline_beta": r["base_beta_interaction"],
                             "adjusted_beta": r["adjusted_beta_interaction"], "change_pct": r.get("attenuation_pct", np.nan)})
        forest(ax, rows, cols, mk)
        ax.set_yticklabels([lab if lab != "adjusted" else "" for lab, *_ in rows], fontsize=5.6)
        ax.set_title(label, fontsize=6.5, loc="left"); ax.set_xlabel("Interaction (β, 95% CI)", fontsize=6)
    fig.legend(handles=[plt.Line2D([], [], color=GREY, marker="o", ls="", label="Matched complete-case baseline"),
                        plt.Line2D([], [], color=CA, marker="s", ls="", label="Biomarker-adjusted")],
               loc="lower center", ncol=2, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    letters(fig, axs.ravel()); save(fig, out, "FigureS2")
    R2 = pd.DataFrame(rows_out); R2.to_csv(out / "FigureS2_plot_data.csv", index=False)
    if "change_pct" in R2 and R2["change_pct"].notna().any():
        CHECKS.append({"figure": "S2", "item": "max |change| ≤ 3.2% (Results 3.8)", "published": 3.2,
                       "computed": round(R2["change_pct"].abs().max(), 2), "match": R2["change_pct"].abs().max() <= 3.25})


# ------------------------------------------------------------------ S3
def figure_S3(out):
    d = pd.read_csv(locate(SRC["S3"]))
    order = ["Original M3", "M3 + AF history", "Exclude AF positive"]
    nice = {"Original M3": "Primary model", "M3 + AF history": "+ AF/flutter history", "Exclude AF positive": "Excluding AF/flutter"}
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 3.6), gridspec_kw=dict(hspace=0.75, wspace=0.7))
    rows_out = []
    for ax, (region, heart, metric, label) in zip(axs.ravel(), ASSOC):
        z = d[(d["region"] == region) & (d["cardiac"] == heart)].set_index("analysis")
        rows = [(f"{nice[a]}\nN = {int(z.loc[a, 'N'])}", z.loc[a, "beta_interaction"], z.loc[a, "CI_low"], z.loc[a, "CI_high"]) for a in order]
        forest(ax, rows, [INK, CA, GREY], ["o", "s", "^"])
        ax.set_title(label, fontsize=6.5, loc="left"); ax.set_xlabel("Interaction (β, 95% CI)", fontsize=6)
        b0 = z.loc["Original M3", "beta_interaction"]
        check("S3", f"{region} primary = Table 2", TABLE2[region], b0)
        for a in order[1:]:
            ch = 100 * (abs(z.loc[a, "beta_interaction"]) - abs(b0)) / abs(b0)
            rows_out.append({"association": label, "analysis": nice[a], "beta": z.loc[a, "beta_interaction"], "change_pct": ch})
    letters(fig, axs.ravel()); save(fig, out, "FigureS3")
    R3 = pd.DataFrame(rows_out); R3.to_csv(out / "FigureS3_plot_data.csv", index=False)
    adj = R3[R3.analysis == "+ AF/flutter history"]["change_pct"].abs().max()
    exc = R3[R3.analysis == "Excluding AF/flutter"]["change_pct"].abs().max()
    CHECKS.append({"figure": "S3", "item": "max |change| with AF adjustment ≤ 6% (Results 3.8)", "published": 6.0, "computed": round(adj, 2), "match": adj <= 6.05})
    CHECKS.append({"figure": "S3", "item": "max |change| with AF exclusion < 8% (Results 3.8)", "published": 8.0, "computed": round(exc, 2), "match": exc < 8.0})


# ------------------------------------------------------------------ S4
def figure_S4(out):
    d = pd.read_csv(locate(SRC["S4"])).set_index("model")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.9, width_ratios=[0.8, 1.2]))
    m3 = d.loc["M3_VASCULAR"]
    for i, (lab, b, lo, hi, c, mk) in enumerate((("APOE ε4\nnoncarriers", m3["beta_HbA1c_APOE0"], m3["CI_APOE0_low"], m3["CI_APOE0_high"], NC, "o"),
                                                  ("APOE ε4\ncarriers", m3["beta_HbA1c_APOE1"], m3["CI_APOE1_low"], m3["CI_APOE1_high"], CA, "s"))):
        axA.errorbar(i, b, yerr=[[b - lo], [hi - b]], fmt=mk, ms=4.5, color=c, capsize=3, lw=1.2)
    axA.axhline(0, color=GREY, lw=0.7, ls="--"); axA.set_xlim(-0.6, 1.6); axA.set_xticks([0, 1])
    axA.set_xticklabels(["APOE ε4\nnoncarriers", "APOE ε4\ncarriers"], fontsize=6)
    axA.set_ylabel("LVEDVi per 1-SD HbA1c\n(mL/m², 95% CI)"); axA.set_title(f"Fully adjusted model (N = {int(m3['N'])})", fontsize=6.2)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    names = {"M0_BASE": "Age and sex", "M1_INTERVAL": "+ HbA1c–CMR interval", "M2_BMI": "+ BMI", "M3_VASCULAR": "+ SBP, antihypertensive treatment"}
    rows = [(f"{names[k]} (N = {int(d.loc[k, 'N'])})", d.loc[k, "beta_interaction"], d.loc[k, "CI_interaction_low"], d.loc[k, "CI_interaction_high"])
            for k in names if k in d.index]
    forest(axB, rows, [INK] * len(rows)); axB.set_xlabel("HbA1c × APOE ε4 interaction (mL/m², 95% CI)", fontsize=6)
    letters(fig, [axA, axB]); save(fig, out, "FigureS4")
    d.reset_index().to_csv(out / "FigureS4_plot_data.csv", index=False)
    for k, col in (("int", ("beta_interaction", "CI_interaction_low", "CI_interaction_high")),
                   ("nc", ("beta_HbA1c_APOE0", "CI_APOE0_low", "CI_APOE0_high")), ("car", ("beta_HbA1c_APOE1", "CI_APOE1_low", "CI_APOE1_high"))):
        for pub, c in zip(S16[k], col):
            check("S4", f"M3 {k} {c}", pub, m3[c], tol=0.006)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--only", nargs="*", default=["S1", "S2", "S3", "S4"])
    ap.add_argument("--figures-dir", default=str(here), help="folder containing make_Figure3.py")
    ap.add_argument("--outdir", default=str(R / "final_rebuild_20261007/SupplementaryFigures"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    for s in a.only:
        print(f"--- {s}")
        {"S1": lambda: figure_S1(out, Path(a.figures_dir)), "S2": lambda: figure_S2(out),
         "S3": lambda: figure_S3(out), "S4": lambda: figure_S4(out)}[s]()
    C = pd.DataFrame(CHECKS); C.to_csv(out / "FigureS_verification.csv", index=False)
    print("\nVERIFICATION\n" + C.to_string(index=False))
    if a.strict and len(C) and not C["match"].all():
        sys.exit("STRICT: values differ from the manuscript - see FigureS_verification.csv")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
