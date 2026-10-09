#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")

SOURCE = (
    ROOT
    / "data/longitudinal_derived"
    / "attrition_atrisk_source.tsv"
)

OUTDIR = (
    ROOT
    / "results/longitudinal_heart_brain/temporal_prediction"
    / "attrition_IPW"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

ID = "shareid"
APOE = "APOE4_carrier"
LA = "la_dim_slope"
MRI_N = "n_eligible_vent_mri"
INCLUDED = "included"

print("=" * 100)
print("153: DESCRIPTIVE ATTRITION FROM VALIDATED AT-RISK SOURCE")
print("=" * 100)
print("SOURCE:", SOURCE)

df = pd.read_csv(SOURCE, sep="\t", low_memory=False)

required = [ID, APOE, LA, MRI_N, INCLUDED]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns:\n{list(df.columns)}"
    )

print("\nRows in validated source:", len(df))
print("Unique participants      :", df[ID].nunique())

# Exact QC against script 152
if df[ID].nunique() != 2522:
    raise RuntimeError(
        f"Expected 2522 source participants; observed {df[ID].nunique()}"
    )

if int(df[INCLUDED].sum()) != 1305:
    raise RuntimeError(
        f"Expected 1305 included participants; observed {int(df[INCLUDED].sum())}"
    )

# -----------------------------------------------------------------------------
# MRI availability
# -----------------------------------------------------------------------------

def status(n):
    if n >= 2:
        return ">=2 eligible MRI"
    elif n == 1:
        return "1 eligible MRI"
    else:
        return "0 eligible MRI"

df["MRI_status"] = df[MRI_N].map(status)

status_order = [
    ">=2 eligible MRI",
    "1 eligible MRI",
    "0 eligible MRI",
]

df["MRI_status"] = pd.Categorical(
    df["MRI_status"],
    categories=status_order,
    ordered=True,
)

overall = (
    df["MRI_status"]
    .value_counts(sort=False)
    .reindex(status_order, fill_value=0)
    .rename_axis("MRI availability")
    .reset_index(name="N")
)

overall["Percent"] = 100 * overall["N"] / len(df)

print("\n" + "=" * 100)
print("OVERALL MRI AVAILABILITY")
print("=" * 100)
print(overall.to_string(index=False))

# -----------------------------------------------------------------------------
# LA remodeling tertiles
# -----------------------------------------------------------------------------

df["LA_tertile"] = pd.qcut(
    df[LA].rank(method="first"),
    q=3,
    labels=[
        "T1: lower remodeling",
        "T2: middle",
        "T3: greater remodeling",
    ],
)

df["APOE_group"] = np.where(
    pd.to_numeric(df[APOE], errors="coerce") == 1,
    "APOE ε4 carrier",
    "APOE ε4 noncarrier",
)

# -----------------------------------------------------------------------------
# Cross-tab
# -----------------------------------------------------------------------------

group_n = (
    df.groupby(
        ["APOE_group", "LA_tertile"],
        observed=False
    )
    .size()
    .rename("group_N")
    .reset_index()
)

counts = (
    df.groupby(
        ["APOE_group", "LA_tertile", "MRI_status"],
        observed=False
    )
    .size()
    .rename("n")
    .reset_index()
)

counts = counts.merge(
    group_n,
    on=["APOE_group", "LA_tertile"],
    how="left",
)

counts["percent"] = 100 * counts["n"] / counts["group_N"]

counts["n_percent"] = counts.apply(
    lambda r: f"{int(r['n'])} ({r['percent']:.1f}%)",
    axis=1,
)

wide = (
    counts.pivot_table(
        index=["APOE_group", "LA_tertile", "group_N"],
        columns="MRI_status",
        values="n_percent",
        aggfunc="first",
        observed=False,
    )
    .reset_index()
)

wide.columns.name = None

wide = wide.rename(
    columns={
        "APOE_group": "APOE ε4 status",
        "LA_tertile": "LA remodeling tertile",
        "group_N": "N",
    }
)

print("\n" + "=" * 100)
print("MRI AVAILABILITY BY APOE4 × LA-REMODELING TERTILE")
print("=" * 100)
print(wide.to_string(index=False))

# -----------------------------------------------------------------------------
# Included vs excluded
# -----------------------------------------------------------------------------

selection = (
    df.groupby(
        ["APOE_group", "LA_tertile"],
        observed=False
    )
    .agg(
        N=(ID, "size"),
        included=(INCLUDED, "sum"),
    )
    .reset_index()
)

selection["excluded"] = selection["N"] - selection["included"]
selection["included_percent"] = (
    100 * selection["included"] / selection["N"]
)

print("\n" + "=" * 100)
print("PRIMARY LONGITUDINAL INCLUSION BY APOE4 × LA-REMODELING TERTILE")
print("=" * 100)
print(selection.to_string(index=False))

# -----------------------------------------------------------------------------
# Check for explicit event/status variables
# -----------------------------------------------------------------------------

keywords = [
    "death", "dead", "deceased", "mort",
    "dement", "alz",
    "stroke", "tia", "cva",
]

possible = [
    c for c in df.columns
    if any(k in c.lower() for k in keywords)
]

print("\n" + "=" * 100)
print("POSSIBLE DEATH / DEMENTIA / STROKE VARIABLES")
print("=" * 100)

if possible:
    for c in possible:
        print(c)
else:
    print("None present in validated source table.")

# -----------------------------------------------------------------------------
# QC
# -----------------------------------------------------------------------------

N_total = len(df)
N_ge2 = int((df[MRI_N] >= 2).sum())
N_one = int((df[MRI_N] == 1).sum())
N_zero = int((df[MRI_N] == 0).sum())

print("\n" + "=" * 100)
print("QC")
print("=" * 100)
print("Source cohort          :", N_total)
print(">=2 eligible MRI       :", N_ge2)
print("Exactly 1 eligible MRI :", N_one)
print("0 eligible MRI         :", N_zero)
print("Check sum              :", N_ge2 + N_one + N_zero)

assert N_ge2 == 1305
assert N_ge2 + N_one + N_zero == N_total

# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------

overall.to_csv(
    OUTDIR / "TableS18_PanelC_overall_MRI_availability.csv",
    index=False,
)

counts.to_csv(
    OUTDIR / "TableS18_PanelC_APOE4_LAtertile_long.csv",
    index=False,
)

wide.to_csv(
    OUTDIR / "TableS18_PanelC_APOE4_LAtertile.csv",
    index=False,
)

selection.to_csv(
    OUTDIR / "TableS18_selection_APOE4_LAtertile.csv",
    index=False,
)

xlsx = OUTDIR / "TableS18_attrition_descriptive.xlsx"

with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    overall.to_excel(
        writer,
        sheet_name="Overall MRI availability",
        index=False,
    )
    wide.to_excel(
        writer,
        sheet_name="Panel C",
        index=False,
    )
    counts.to_excel(
        writer,
        sheet_name="Panel C long",
        index=False,
    )
    selection.to_excel(
        writer,
        sheet_name="Selection",
        index=False,
    )

print("\nSaved:")
print(" ", OUTDIR / "TableS18_PanelC_APOE4_LAtertile.csv")
print(" ", xlsx)
print("\nDONE.")
