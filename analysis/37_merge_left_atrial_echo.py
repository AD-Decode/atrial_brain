#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
RES = ROOT / "results"

BRAINFILE = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"
ATRIAFILE = RES / "left_atrial_echo_pht015145_clean.csv"

OUTFILE = RES / "neurocardiac_metadata_with_atria.csv"
SUMMARY = RES / "left_atrial_overlap_summary.csv"

brain = pd.read_csv(BRAINFILE)
atria = pd.read_csv(ATRIAFILE)

print("=" * 100)
print("LEFT ATRIAL / NEUROCARDIAC OVERLAP")
print("=" * 100)

print(f"\nNeurocardiac cohort N = {len(brain)}")
print(f"Atrial table N       = {len(atria)}")


# ------------------------------------------------------------------
# Inspect identifiers
# ------------------------------------------------------------------

print("\nBrain ID candidates:")
print([
    c for c in brain.columns
    if "share" in c.lower()
    or "dbgap" in c.lower()
    or c.lower() in ["id", "idtype"]
])

print("\nAtrial ID candidates:")
print([
    c for c in atria.columns
    if "share" in c.lower()
    or "dbgap" in c.lower()
    or c.lower() in ["id", "idtype"]
])


if "shareid" not in brain.columns:
    raise RuntimeError("shareid not found in neurocardiac analysis-ready file.")

if "shareid" not in atria.columns:
    raise RuntimeError("shareid not found in atrial table.")


# Normalize IDs
for d in [brain, atria]:
    d["shareid"] = (
        d["shareid"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

if "idtype" in brain.columns:
    brain["idtype"] = pd.to_numeric(
        brain["idtype"],
        errors="coerce"
    )

if "idtype" in atria.columns:
    atria["idtype"] = pd.to_numeric(
        atria["idtype"],
        errors="coerce"
    )


# ------------------------------------------------------------------
# Check duplicates
# ------------------------------------------------------------------

print("\nDuplicate shareid counts:")
print("brain:", brain["shareid"].duplicated().sum())
print("atria:", atria["shareid"].duplicated().sum())

if brain["shareid"].duplicated().any():
    print("\nBrain duplicate shareids:")
    print(
        brain.loc[
            brain["shareid"].duplicated(False),
            ["shareid"] + (
                ["idtype"] if "idtype" in brain.columns else []
            )
        ].sort_values("shareid").to_string(index=False)
    )

if atria["shareid"].duplicated().any():
    print("\nAtrial duplicate shareids:")
    print(
        atria.loc[
            atria["shareid"].duplicated(False),
            ["shareid", "idtype"]
        ].sort_values("shareid").to_string(index=False)
    )


# ------------------------------------------------------------------
# Use shareid + idtype if available in both; otherwise shareid
# ------------------------------------------------------------------

if (
    "idtype" in brain.columns
    and "idtype" in atria.columns
):
    keys = ["shareid", "idtype"]
else:
    keys = ["shareid"]

print("\nMerge keys:", keys)


# ------------------------------------------------------------------
# Keep desired atrial measures
# ------------------------------------------------------------------

atrial_vars = [
    "TwoD_SingPlVol_LA4chSVolume",
    "TwoD_SingPlVol_LA4chDVolume",
    "TwoD_SingPlVol_LA2chSVolume",
    "TwoD_SingPlVol_LA2chDVolume",
    "Left_atrial_volume_max_average",
    "Left_atrial_volume_min_average",
    "LA_total_emptying_volume",
    "LA_total_emptying_fraction",
]

keep = keys + atrial_vars

a = atria[keep].copy()

merged = brain.merge(
    a,
    on=keys,
    how="left",
    validate="one_to_one"
)

print("\nMerged N:", len(merged))


# ------------------------------------------------------------------
# Compute BSA exactly as in Tier-2 model if needed
# ------------------------------------------------------------------

if "BSA_m2" not in merged.columns:

    if not {"height_in", "weight_lb"}.issubset(merged.columns):
        raise RuntimeError(
            "Need either BSA_m2 or height_in + weight_lb."
        )

    height_cm = (
        pd.to_numeric(
            merged["height_in"],
            errors="coerce"
        ) * 2.54
    )

    weight_kg = (
        pd.to_numeric(
            merged["weight_lb"],
            errors="coerce"
        ) * 0.45359237
    )

    merged["BSA_m2"] = np.sqrt(
        height_cm * weight_kg / 3600.0
    )


# ------------------------------------------------------------------
# Derived atrial phenotypes
# ------------------------------------------------------------------

merged["LAVI_max"] = (
    pd.to_numeric(
        merged["Left_atrial_volume_max_average"],
        errors="coerce"
    )
    /
    pd.to_numeric(
        merged["BSA_m2"],
        errors="coerce"
    )
)

merged["LAVI_min"] = (
    pd.to_numeric(
        merged["Left_atrial_volume_min_average"],
        errors="coerce"
    )
    /
    pd.to_numeric(
        merged["BSA_m2"],
        errors="coerce"
    )
)

merged["LA_emptying_volume_index"] = (
    pd.to_numeric(
        merged["LA_total_emptying_volume"],
        errors="coerce"
    )
    /
    pd.to_numeric(
        merged["BSA_m2"],
        errors="coerce"
    )
)


# ------------------------------------------------------------------
# Summaries
# ------------------------------------------------------------------

vars_to_report = [
    "Left_atrial_volume_max_average",
    "Left_atrial_volume_min_average",
    "LA_total_emptying_volume",
    "LA_total_emptying_fraction",
    "LAVI_max",
    "LAVI_min",
    "LA_emptying_volume_index",
]

rows = []

print("\n")
print("=" * 100)
print("OVERLAP / COMPLETENESS IN 785-PERSON COHORT")
print("=" * 100)

for c in vars_to_report:

    x = pd.to_numeric(
        merged[c],
        errors="coerce"
    )

    rows.append({
        "variable": c,
        "N_nonmissing": int(x.notna().sum()),
        "percent_of_785": 100 * x.notna().mean(),
        "mean": x.mean(),
        "sd": x.std(),
        "min": x.min(),
        "median": x.median(),
        "max": x.max(),
    })

summary = pd.DataFrame(rows)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4g}"
    )
)


# ------------------------------------------------------------------
# APOE4 overlap
# ------------------------------------------------------------------

if "APOE4_carrier" in merged.columns:

    print("\n")
    print("=" * 100)
    print("ATRIAL DATA BY APOE4 STATUS")
    print("=" * 100)

    has_atria = merged["LA_total_emptying_fraction"].notna()

    tmp = (
        merged.loc[has_atria, "APOE4_carrier"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print(tmp.to_string())


# ------------------------------------------------------------------
# Correlation between atrial size and function
# ------------------------------------------------------------------

pair = merged[
    ["LAVI_max", "LA_total_emptying_fraction"]
].dropna()

if len(pair) > 2:
    r = pair.corr().iloc[0, 1]

    print("\nCorrelation:")
    print(
        f"LAVI_max vs LA emptying fraction: "
        f"r = {r:.4f}, N = {len(pair)}"
    )


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

merged.to_csv(
    OUTFILE,
    index=False
)

summary.to_csv(
    SUMMARY,
    index=False
)

print("\nSaved merged analysis file:")
print(OUTFILE)

print("\nSaved overlap summary:")
print(SUMMARY)
