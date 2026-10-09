#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")

SRC = ROOT / "results/tier2_corrected/tier2_corrected_all.csv"
OUTDIR = ROOT / "results/final_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "FINAL_master_core4_interactions.csv"

d = pd.read_csv(SRC)

targets = [
    {
        "order": 1,
        "association": "LVEF × APOE ε4 → L middle temporal volume",
        "cardiac": "LVEF",
        "region": "lh_middletemporal_vol",
        "metric_type": "volume",
        "atlas": "Desikan-Killiany",
    },
    {
        "order": 2,
        "association": "LVESVi × APOE ε4 → R superior frontal area",
        "cardiac": "LVESVi",
        "region": "rh_superiorfrontal_area",
        "metric_type": "surface_area",
        "atlas": "Desikan-Killiany",
    },
    {
        "order": 3,
        "association": "LV mass index × APOE ε4 → R lingual volume",
        "cardiac": "LV_MASSi",
        "region": "rh_lingual_vol",
        "metric_type": "volume",
        "atlas": "Desikan-Killiany",
    },
    {
        "order": 4,
        "association": "LV mass index × APOE ε4 → R lingual area",
        "cardiac": "LV_MASSi",
        "region": "rh_lingual_area",
        "metric_type": "surface_area",
        "atlas": "Desikan-Killiany",
    },
]

rows = []

for t in targets:

    hit = d[
        (d["model"] == "M3_APOE4")
        & (d["cardiac"] == t["cardiac"])
        & (d["region"] == t["region"])
        & (d["metric_type"] == t["metric_type"])
    ].copy()

    if len(hit) != 1:
        raise RuntimeError(
            f"{t['association']}: expected exactly 1 row, "
            f"found {len(hit)}"
        )

    r = hit.iloc[0]

    rows.append({
        "order": t["order"],
        "association": t["association"],

        "cardiac_metric": t["cardiac"],
        "brain_metric": t["region"],
        "metric_type": t["metric_type"],
        "atlas": t["atlas"],

        # NOTE: "family" in script 32 means anatomical
        # hypothesis family, NOT atlas family.
        "anatomical_FDR_family": r["family"],

        "model": "M3_APOE4",

        "N": int(r["N"]),

        # Verified from the original M3 tables
        "APOE4_noncarriers_n": 592,
        "APOE4_carriers_n": 173,

        "interaction_beta": float(r["beta_interaction"]),
        "HC3_SE": float(r["SE_interaction"]),
        "CI_low": float(r["CI_interaction_low"]),
        "CI_high": float(r["CI_interaction_high"]),
        "P": float(r["p_interaction"]),
        "q": float(r["q_interaction"]),

        "FDR_method": "Benjamini-Hochberg",

        "FDR_family_definition":
            "BH-FDR within imaging metric type × anatomical "
            "region family × cardiac phenotype; "
            "Desikan-Killiany and Destrieux parcels jointly included",

        "estimate_source":
            "results/tier2_corrected/tier2_corrected_all.csv",

        "analysis_source":
            "code/32_tier2_corrected_scaling.py",
    })


out = (
    pd.DataFrame(rows)
    .sort_values("order")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# QC
# ------------------------------------------------------------

assert (out["N"] == 765).all()

assert (
    out["APOE4_noncarriers_n"]
    + out["APOE4_carriers_n"]
    == out["N"]
).all()

expected = {
    "lh_middletemporal_vol": 0.233332,
    "rh_superiorfrontal_area": -0.170334,
    "rh_lingual_vol": 0.305335,
    "rh_lingual_area": 0.225921,
}

for _, r in out.iterrows():
    exp = expected[r["brain_metric"]]

    if abs(r["interaction_beta"] - exp) > 1e-5:
        raise RuntimeError(
            f"Unexpected beta for {r['brain_metric']}: "
            f"{r['interaction_beta']} vs expected {exp}"
        )


print("\nFINAL AUTHORITATIVE CORE-4 TABLE")
print("=" * 110)

cols = [
    "association",
    "N",
    "APOE4_noncarriers_n",
    "APOE4_carriers_n",
    "interaction_beta",
    "HC3_SE",
    "CI_low",
    "CI_high",
    "P",
    "q",
    "anatomical_FDR_family",
]

print(out[cols].to_string(index=False))

out.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)
