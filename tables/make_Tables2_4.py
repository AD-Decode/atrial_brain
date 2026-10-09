#!/usr/bin/env python3
"""
Tables 2-4 of the manuscript, rebuilt from the models (same code paths as the verified figures).

  Table 2  Regional ventricular cardiac x APOE e4 interactions (exploratory)
           - refits the four M3 models with make_Figure3.fit_m3 (= script 32); within-family q from the
             full-cohort tier2_corrected_all.csv (M3_APOE4 rows).
  Table 3  APOE e4 allele-dose and genotype sensitivity analyses
           - formatted from the Table 3 summary file (written by 51/57), as Figure S6.
  Table 4  LAVI x APOE e4 interaction after adjustment for ventricular structure and function
           - refits the four nested models of script 42 with the make_Figure5_LAVI data preparation
             (adds beta noncarrier, beta carrier and R^2).

Every value is checked against the manuscript tables (Tables_verification.csv; --strict stops on mismatch).
Outputs (OUTDIR): Table2.csv, Table3.csv, Table4.csv, Tables2_4.xlsx, Tables2_4.docx (if python-docx is
installed), Tables_verification.csv. Summary statistics only.

Usage (from the repository, with the figure scripts in ../figures):
    python make_Tables2_4.py --strict [--outdir DIR] [--figures-dir DIR]
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/data/qiallab/Framingham")
TABLE3_SRC = ROOT / "results/paper_figures_tables/APOE_genotype_dose/Table3_APOE_genotype_dose_validation.csv"
MINUS = "−"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def f3(x, d=3):
    return f"{x:.{d}f}".replace("-", MINUS)


def fp(p):
    """manuscript P/q style: 3 significant figures; scientific as a×10⁻ᵇ below 0.001"""
    if p >= 0.001:
        return f"{p:.3g}"
    m, e = f"{p:.2e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m}×10{str(int(e)).translate(sup)}"


def ci(lo, hi):
    return f"{f3(lo)} to {f3(hi)}"


# ------------------------------------------------------------------ Table 2
def table2(F3):
    df = F3.prepare(pd.read_csv(F3.DATAFILE, low_memory=False))
    t2 = pd.read_csv(F3.TIER2)
    t2 = t2[t2["model"] == "M3_APOE4"]
    lab = {"LVEF": "LVEF", "LVESVi": "LVESVi", "LV_MASSi": "LV mass index"}
    brain = {"lh_middletemporal_vol": "Left middle temporal volume", "rh_superiorfrontal_area": "Right superior frontal surface area",
             "rh_lingual_vol": "Right lingual volume", "rh_lingual_area": "Right lingual surface area"}
    order = ["lh_middletemporal_vol", "rh_superiorfrontal_area", "rh_lingual_vol", "rh_lingual_area"]
    find = {r: (h, m) for h, r, m, _ in F3.FINDINGS}
    rows, raw = [], []
    for region in order:
        heart, metric = find[region]
        r = F3.fit_m3(df, heart, region, metric)
        q = t2[(t2["region"] == region) & (t2["cardiac"] == heart)]["q_interaction"]
        if len(q) != 1:
            sys.exit(f"Table 2: no unique tier2 q for {region}/{heart}")
        q = float(q.iloc[0])
        raw.append({"region": region, "N": r["N"], "nc": r["nc"][0], "int": r["int"][0], "lo": r["int"][1],
                    "hi": r["int"][2], "p": r["p_int"], "q": q})
        rows.append({"Cardiac phenotype": lab[heart], "Brain phenotype": brain[region], "N": r["N"],
                     "β, noncarriers": f3(r["nc"][0]), "β, interaction": f3(r["int"][0]),
                     "95% CI": ci(r["int"][1], r["int"][2]), "P": fp(r["p_int"]), "FDR q (within family)": fp(q)})
    return pd.DataFrame(rows), pd.DataFrame(raw)


PUB2 = {  # manuscript Table 2
    "lh_middletemporal_vol": dict(N=765, nc=-0.046, int=0.233, lo=0.087, hi=0.380, p=0.00176, q=0.0352),
    "rh_superiorfrontal_area": dict(N=765, nc=0.035, int=-0.170, lo=-0.260, hi=-0.081, p=1.97e-4, q=0.00433),
    "rh_lingual_vol": dict(N=765, nc=-0.092, int=0.305, lo=0.144, hi=0.467, p=2.16e-4, q=0.00604),
    "rh_lingual_area": dict(N=765, nc=-0.106, int=0.226, lo=0.095, hi=0.357, p=7.09e-4, q=0.0198),
}


# ------------------------------------------------------------------ Table 3
KEYS3 = [(("LVEF", "middle temporal"), "LVEF → left middle temporal volume"),
         (("LVESV", "superior frontal"), "LVESVi → right superior frontal surface area"),
         (("mass", "lingual vol"), "LV mass index → right lingual volume"),
         (("mass", "lingual area"), "LV mass index → right lingual surface area"),
         (("LAVI", "occipital"), "LAVI → left middle-occipital cortical-thickness variability")]
PUB3 = [(644, 0.161, 0.0066), (644, -0.187, 4.45e-6), (644, 0.294, 1.26e-4), (644, 0.236, 6.19e-4), (387, -0.473, 3.13e-4)]


def table3():
    src = pd.read_csv(TABLE3_SRC)
    txt = src["Finding"].astype(str).str.lower()
    rows, raw = [], []
    for keys, label in KEYS3:
        hit = src[np.logical_and.reduce([txt.str.contains(k.lower()) for k in keys])]
        if len(hit) != 1:
            sys.exit(f"Table 3: could not match {keys}; Findings: {src['Finding'].tolist()}")
        r = hit.iloc[0]
        raw.append({"N": int(r["N"]), "b": r["β heart×ε4 dose"], "q": r["Dose q"]})
        rows.append({"Cardiac–brain relationship": label, "N": int(r["N"]), "ε3/ε3, n": int(r["E3/E3 n"]),
                     "ε3/ε4, n": int(r["E3/E4 n"]), "ε4/ε4, n": int(r["E4/E4 n"]),
                     "β per ε4 allele (95% CI)": f"{f3(r['β heart×ε4 dose'])} ({ci(r['95% CI low'], r['95% CI high'])})",
                     "P value": fp(r["Dose p"]), "FDR q": fp(r["Dose q"]), "Genotype omnibus FDR q": fp(r["Genotype omnibus q"])})
    return pd.DataFrame(rows), pd.DataFrame(raw)


# ------------------------------------------------------------------ Table 4
def table4(F5):
    df = F5.load()
    common = [F5.OUTCOME, F5.ATRIAL, "APOE4_carrier", "LV_MASSi_recomputed", "LVEF"] + F5.COVS
    d = df[common].apply(F5.num).dropna().copy()
    d["y_z"], d["a_z"] = F5.z(d[F5.OUTCOME]), F5.z(d[F5.ATRIAL]); d["a_x_e4"] = d["a_z"] * d["APOE4_carrier"]
    d["LVMI_z"], d["LVEF_z"] = F5.z(d["LV_MASSi_recomputed"]), F5.z(d["LVEF"])
    base = F5.design(d, ["a_z", "APOE4_carrier", "a_x_e4"])
    rows, raw = [], []
    for name, extra in (("Baseline", []), ("+ LV mass index", ["LVMI_z"]), ("+ LVEF", ["LVEF_z"]),
                        ("+ LV mass index + LVEF", ["LVMI_z", "LVEF_z"])):
        f = sm.OLS(d["y_z"], sm.add_constant(d[base + extra].astype(float))).fit(cov_type="HC3")
        nc = F5.lincomb(f, {"a_z": 1}); it = F5.lincomb(f, {"a_x_e4": 1}); car = F5.lincomb(f, {"a_z": 1, "a_x_e4": 1})
        raw.append({"model": name, "N": len(d), "int": it[0], "p": f.pvalues["a_x_e4"]})
        rows.append({"Model": name, "N": len(d), "APOE ε4 noncarriers, n": int((d.APOE4_carrier == 0).sum()),
                     "APOE ε4 carriers, n": int((d.APOE4_carrier == 1).sum()), "β noncarrier": f3(nc[0]),
                     "β interaction": f3(it[0]), "95% CI": ci(it[1], it[2]), "Interaction P value": fp(f.pvalues["a_x_e4"]),
                     "β carrier": f3(car[0]), "R²": f3(f.rsquared)})
    return pd.DataFrame(rows), pd.DataFrame(raw)


PUB4 = {"Baseline": (-0.545, 1.56e-5), "+ LV mass index": (-0.549, 1.76e-5), "+ LVEF": (-0.545, 1.76e-5),
        "+ LV mass index + LVEF": (-0.549, 1.97e-5)}


def close(a, b, rel=0.02, abs_=0.002):
    return abs(a - b) <= max(abs_, rel * abs(b))


def write_docx(tables, path):
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        print("python-docx not installed - skipping DOCX (pip install python-docx)"); return
    doc = Document()
    for title, t in tables:
        doc.add_paragraph().add_run(title).bold = True
        tb = doc.add_table(rows=1, cols=len(t.columns)); tb.style = "Table Grid"
        for i, c in enumerate(t.columns):
            tb.rows[0].cells[i].text = str(c)
        for _, r in t.iterrows():
            cells = tb.add_row().cells
            for i, c in enumerate(t.columns):
                cells[i].text = str(r[c])
        for row in tb.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8)
        doc.add_paragraph()
    doc.save(path)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--outdir", default=str(ROOT / "results/final_rebuild_20261007/Tables2_4"))
    ap.add_argument("--figures-dir", default=str(here.parent / "figures"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    figdir = Path(a.figures_dir)
    F3 = load_module(figdir / "make_Figure3.py", "make_Figure3")
    F5 = load_module(figdir / "make_Figure5_LAVI.py", "make_Figure5_LAVI")

    T2, r2 = table2(F3); T3, r3 = table3(); T4, r4 = table4(F5)

    chk = []
    for _, r in r2.iterrows():
        pub = PUB2[r["region"]]
        for k in ("nc", "int", "lo", "hi"):
            chk.append({"table": 2, "row": r["region"], "item": k, "published": pub[k], "computed": round(r[k], 4), "match": close(r[k], pub[k])})
        for k in ("p", "q"):
            chk.append({"table": 2, "row": r["region"], "item": k, "published": pub[k], "computed": float(f"{r[k]:.3g}"), "match": close(r[k], pub[k], rel=0.01, abs_=0)})
        chk.append({"table": 2, "row": r["region"], "item": "N", "published": pub["N"], "computed": r["N"], "match": r["N"] == pub["N"]})
    for (_, r), (n, b, q), (_, lab) in zip(r3.iterrows(), PUB3, KEYS3):
        chk += [{"table": 3, "row": lab, "item": "N", "published": n, "computed": r["N"], "match": r["N"] == n},
                {"table": 3, "row": lab, "item": "beta", "published": b, "computed": round(r["b"], 4), "match": close(r["b"], b)},
                {"table": 3, "row": lab, "item": "q", "published": q, "computed": float(f"{r['q']:.3g}"), "match": close(r["q"], q, rel=0.01, abs_=0)}]
    for _, r in r4.iterrows():
        b, p = PUB4[r["model"]]
        chk += [{"table": 4, "row": r["model"], "item": "N", "published": 451, "computed": r["N"], "match": r["N"] == 451},
                {"table": 4, "row": r["model"], "item": "beta interaction", "published": b, "computed": round(r["int"], 4), "match": close(r["int"], b)},
                {"table": 4, "row": r["model"], "item": "P", "published": p, "computed": float(f"{r['p']:.3g}"), "match": close(r["p"], p, rel=0.01, abs_=0)}]
    C = pd.DataFrame(chk)

    for name, t in (("Table2", T2), ("Table3", T3), ("Table4", T4)):
        t.to_csv(out / f"{name}.csv", index=False)
    with pd.ExcelWriter(out / "Tables2_4.xlsx") as w:
        T2.to_excel(w, sheet_name="Table 2", index=False); T3.to_excel(w, sheet_name="Table 3", index=False)
        T4.to_excel(w, sheet_name="Table 4", index=False)
    write_docx([("Table 2. APOE ε4 modification of regional ventricular cardiac–brain associations (exploratory)", T2),
                ("Table 3. APOE ε4 allele-dose and genotype sensitivity analyses of cardiac–brain associations", T3),
                ("Table 4. LAVI × APOE ε4 interaction after adjustment for ventricular structure and function", T4)],
               out / "Tables2_4.docx")
    C.to_csv(out / "Tables_verification.csv", index=False)
    for name, t in (("TABLE 2", T2), ("TABLE 3", T3), ("TABLE 4", T4)):
        print(f"\n{name}\n" + t.to_string(index=False))
    print("\nVERIFICATION\n" + C.to_string(index=False))
    if a.strict and not C["match"].all():
        sys.exit("STRICT: values differ from the manuscript tables - see Tables_verification.csv")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
