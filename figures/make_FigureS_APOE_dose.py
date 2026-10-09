#!/usr/bin/env python3
"""
Supplementary figure - APOE e4 allele-dose sensitivity analyses of the regional cardiac-brain
associations (former main Figure 5, moved to the supplement and retitled; replaces 61).

Plots the per-allele change in the standardized cardiac-brain slope (95% CI) from Table 3
(e3/e3, e3/e4, e4/e4 participants). The four ventricular associations are drawn in grey and
labelled exploratory (the regional screen was consistent with chance in the permutation test);
the LAVI association is drawn in the carrier colour. Dose and genotype-omnibus q values sit in
their own columns. No in-figure title (no "validation").

Input : Table3_APOE_genotype_dose_validation.csv (summary-level; written by 57/51)
Output: FigureS_APOE_dose.{pdf,svg,png}, FigureS_APOE_dose_plot_data.csv, FigureS_APOE_dose_verification.csv
Usage : python make_FigureS_APOE_dose.py [--strict] [--outdir DIR]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path("/data/qiallab/Framingham")
INFILE = ROOT / "results/paper_figures_tables/APOE_genotype_dose/Table3_APOE_genotype_dose_validation.csv"

# Row order, label, and published per-allele beta (Table 3) - matched by keywords in the Finding column
ROWS = [
    (("LVEF", "middle temporal"), "LVEF — left middle temporal volume", 0.161, False),
    (("LVESV", "superior frontal"), "LVESVi — right superior frontal surface area", -0.187, False),
    (("mass", "lingual vol"), "LV mass index — right lingual volume", 0.294, False),
    (("mass", "lingual area"), "LV mass index — right lingual surface area", 0.236, False),
    (("LAVI", "occipital"), "LAVI — left middle-occipital thickness variability", -0.473, True),
]
EXPECTED_N = [644, 644, 644, 644, 387]
TOL = 0.002

_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in _AVAIL), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "font.size": 7, "axes.linewidth": 0.7, "pdf.fonttype": 42, "svg.fonttype": "none"})
CA, GREY = "#D9822B", "#7B8794"


def fmt_q(q):
    if q >= 0.001:
        return f"{q:.2g}" if q < 0.01 else f"{q:.3f}"
    m, e = f"{q:.2e}".split("e")
    return f"{m}×10$^{{{int(e)}}}$"


def find_row(df, keys):
    txt = df["Finding"].astype(str).str.lower().str.replace("_", " ")
    hit = df[np.logical_and.reduce([txt.str.contains(k.lower()) for k in keys])]
    if len(hit) != 1:
        sys.exit(f"Could not match exactly one Table 3 row for {keys}; Findings are:\n{df['Finding'].tolist()}")
    return hit.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--outdir", default=str(ROOT / "results/final_rebuild_20261007/FigureS_APOE_dose"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INFILE)

    recs, checks = [], []
    for (keys, label, pub, is_lavi), n_exp in zip(ROWS, EXPECTED_N):
        r = find_row(df, keys)
        rec = {"label": label, "N": int(r["N"]), "e33": int(r["E3/E3 n"]), "e34": int(r["E3/E4 n"]), "e44": int(r["E4/E4 n"]),
               "beta_dose": r["β heart×ε4 dose"], "ci_low": r["95% CI low"], "ci_high": r["95% CI high"],
               "dose_q": r["Dose q"], "genotype_q": r["Genotype omnibus q"], "atrial": is_lavi}
        recs.append(rec)
        checks += [{"row": label, "item": "beta per allele", "published": pub, "file": round(rec["beta_dose"], 4),
                    "match": abs(rec["beta_dose"] - pub) <= TOL},
                   {"row": label, "item": "N", "published": n_exp, "file": rec["N"], "match": rec["N"] == n_exp}]
    R = pd.DataFrame(recs)

    fig = plt.figure(figsize=(7.2, 2.9))
    ax = fig.add_axes([0.36, 0.17, 0.40, 0.76])
    y = np.arange(len(R))[::-1]
    for yi, (_, r) in zip(y, R.iterrows()):
        c = CA if r["atrial"] else GREY
        ax.errorbar(r["beta_dose"], yi, xerr=[[r["beta_dose"] - r["ci_low"]], [r["ci_high"] - r["beta_dose"]]],
                    fmt="o" if r["atrial"] else "s", ms=4.2, color=c, capsize=2.5, lw=1.2)
    ax.axvline(0, color=GREY, lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\nN = {n} (ε3/ε3 {a_}, ε3/ε4 {b_}, ε4/ε4 {c_})"
                        for l, n, a_, b_, c_ in zip(R.label, R.N, R.e33, R.e34, R.e44)], fontsize=6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Change in standardized slope per APOE ε4 allele\n(β, 95% CI)")
    ax.set_ylim(-0.6, len(R) - 0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # q-value columns in figure coordinates (cannot collide with the plot)
    for x, head, col in ((0.80, "Dose q", "dose_q"), (0.91, "Genotype\nomnibus q", "genotype_q")):
        fig.text(x, 0.95, head, ha="center", va="bottom", fontsize=6, fontweight="bold")
        for yi, (_, r) in zip(y, R.iterrows()):
            yy = ax.transData.transform((0, yi))[1]
            fig.text(x, fig.transFigure.inverted().transform((0, yy))[1], fmt_q(r[col]), ha="center", va="center", fontsize=6)
    fig.text(0.36, -0.06, "Grey: regional ventricular associations (exploratory). Orange: atrial association.\n"
             "Same participants as the original analyses; ε4/ε4 estimates are imprecise.",
             fontsize=5.6, color="#52606D")

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(out / f"FigureS_APOE_dose.{ext}", bbox_inches="tight", **kw)
    R.to_csv(out / "FigureS_APOE_dose_plot_data.csv", index=False)
    chk = pd.DataFrame(checks); chk.to_csv(out / "FigureS_APOE_dose_verification.csv", index=False)
    print(R.round(4).to_string(index=False)); print("\nVERIFICATION (Table 3)\n" + chk.to_string(index=False))
    if a.strict and not chk["match"].all():
        sys.exit("STRICT: values differ from Table 3")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
