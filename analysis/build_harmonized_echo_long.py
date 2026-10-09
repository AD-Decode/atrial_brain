#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/data/qiallab/Framingham/downloads/longitudinal_20260924")
OUT = ROOT / "harmonized"
OUT.mkdir(exist_ok=True)

def getfile(pht):
    return next(ROOT.glob(f"*{pht}*.HMB-IRB-MDS.txt.gz"))

def read(pht):
    d = pd.read_csv(
        getfile(pht),
        sep="\t",
        dtype=str,
        low_memory=False,
        comment="#"
    )
    d.columns = [c.strip() for c in d.columns]
    d["shareid"] = d["shareid"].astype(str).str.strip()
    return d

def num(x):
    return pd.to_numeric(x, errors="coerce")

# ----------------------------------------------------------
# READ FOUR EXAMS
# ----------------------------------------------------------
e4 = read("pht000300")
e5 = read("pht000301")
e6 = read("pht000150")
e8 = read("pht002572")

# ----------------------------------------------------------
# STANDARDIZE EACH EXAM
# ----------------------------------------------------------
def make_exam(df, exam, mapping):
    z = pd.DataFrame()
    z["shareid"] = df["shareid"]
    z["exam"] = exam

    for canonical, raw in mapping.items():
        if raw in df.columns:
            z[canonical] = num(df[raw])
        else:
            z[canonical] = np.nan

    return z

m4 = {
    "ivs_d":   "IVSD4",
    "pwt_d":   "LVPD4",
    "lvdd":    "LVDD4",
    "lvds":    "LVDS4",
    "la_dim":  "LAD4",
    "ao_root": "AOR4",
}

m5 = {
    "ivs_d":   "IVSD5",
    "pwt_d":   "LVPD5",
    "lvdd":    "LVDD5",
    "lvds":    "LVDS5",
    "la_dim":  "LAD5",
    "ao_root": "AOR5",
}

m6 = {
    "ivs_d":   "LMIVS_DV",
    "pwt_d":   "LMLVP_DV",
    "lvdd":    "LMLVD_DV",
    "lvds":    "LMLVD_SV",
    "la_dim":  "LMLADM_V",
    "ao_root": "LMAORT_V",
}

m8 = {
    "ivs_d":   "x62",
    "pwt_d":   "x69",
    "lvdd":    "x76",
    "lvds":    "x83",
    "la_dim":  "x23",
    "ao_root": "x16",
}

long = pd.concat(
    [
        make_exam(e4, 4, m4),
        make_exam(e5, 5, m5),
        make_exam(e6, 6, m6),
        make_exam(e8, 8, m8),
    ],
    ignore_index=True
)

# ----------------------------------------------------------
# DERIVED TRAITS — use same equations at every exam
# ----------------------------------------------------------
long["fs_derived_pct"] = (
    (long["lvdd"] - long["lvds"]) / long["lvdd"] * 100
)

long["wall_sum"] = long["ivs_d"] + long["pwt_d"]

# ----------------------------------------------------------
# QC: biologically impossible / suspicious values
# Do not delete yet; simply flag.
# ----------------------------------------------------------
long["flag_lvdd"] = ~long["lvdd"].between(2.5, 9.0)
long["flag_lvds"] = ~long["lvds"].between(1.0, 8.0)
long["flag_ivs"] = ~long["ivs_d"].between(0.3, 3.0)
long["flag_pwt"] = ~long["pwt_d"].between(0.3, 3.0)
long["flag_la"] = ~long["la_dim"].between(1.0, 8.0)
long["flag_ao"] = ~long["ao_root"].between(1.0, 6.0)
long["flag_fs"] = ~long["fs_derived_pct"].between(5, 70)

# NaNs should not count as outliers
for v, flag in [
    ("lvdd","flag_lvdd"),
    ("lvds","flag_lvds"),
    ("ivs_d","flag_ivs"),
    ("pwt_d","flag_pwt"),
    ("la_dim","flag_la"),
    ("ao_root","flag_ao"),
    ("fs_derived_pct","flag_fs"),
]:
    long.loc[long[v].isna(), flag] = False

# ----------------------------------------------------------
# DISTRIBUTION TABLE BY EXAM
# ----------------------------------------------------------
traits = [
    "ivs_d",
    "pwt_d",
    "lvdd",
    "lvds",
    "la_dim",
    "ao_root",
    "fs_derived_pct",
    "wall_sum",
]

rows = []

for exam in [4,5,6,8]:
    d = long[long["exam"] == exam]

    for v in traits:
        x = d[v].dropna()

        if len(x) == 0:
            continue

        rows.append({
            "exam": exam,
            "variable": v,
            "N": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "p01": x.quantile(.01),
            "p05": x.quantile(.05),
            "median": x.median(),
            "p95": x.quantile(.95),
            "p99": x.quantile(.99),
            "min": x.min(),
            "max": x.max(),
        })

qc = pd.DataFrame(rows)

print("\n" + "="*100)
print("HARMONIZED ECHO DISTRIBUTIONS")
print("="*100)

print(
    qc.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

# ----------------------------------------------------------
# SUBJECT COVERAGE
# ----------------------------------------------------------
print("\n" + "="*100)
print("LONGITUDINAL COVERAGE BY CARDIAC TRAIT")
print("="*100)

coverage = []

for v in traits:
    d = long.dropna(subset=[v])

    counts = d.groupby("shareid")[v].count()

    row = {
        "variable": v,
        "N_any": int(counts.size),
        "N_ge2": int((counts >= 2).sum()),
        "N_ge3": int((counts >= 3).sum()),
        "N_ge4": int((counts >= 4).sum()),
    }

    coverage.append(row)
    print(row)

coverage = pd.DataFrame(coverage)

# ----------------------------------------------------------
# COMPLETE TRAJECTORY PATTERNS
# ----------------------------------------------------------
core = ["lvdd","lvds","ivs_d","pwt_d","la_dim","ao_root"]

print("\n" + "="*100)
print("EXAM-SPECIFIC NONMISSING N")
print("="*100)

for exam in [4,5,6,8]:
    d = long[long["exam"] == exam]
    print(f"\nExam {exam}")
    for v in core:
        print(f"  {v:12s}: {d[v].notna().sum()}")

# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------
long.to_csv(
    OUT / "echo_harmonized_long.tsv",
    sep="\t",
    index=False
)

qc.to_csv(
    OUT / "echo_harmonized_distribution_qc.tsv",
    sep="\t",
    index=False
)

coverage.to_csv(
    OUT / "echo_harmonized_longitudinal_coverage.tsv",
    sep="\t",
    index=False
)

print("\nSaved:")
print(OUT / "echo_harmonized_long.tsv")
print(OUT / "echo_harmonized_distribution_qc.tsv")
print(OUT / "echo_harmonized_longitudinal_coverage.tsv")
