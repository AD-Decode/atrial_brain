#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/data/qiallab/Framingham")
RES = ROOT / "results"

FILE = RES / "tier2_corrected" / "tier2_corrected_all.csv"
OUTDIR = RES / "tier2_followup"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FILE)

# ============================================================
# 1. Model-to-model stability table
# ============================================================

key = [
    "metric_type",
    "family",
    "region",
    "cardiac"
]

m1 = (
    df[df["model"] == "M1_basic"]
    [key + ["N", "beta_cardiac", "p_cardiac", "q_cardiac"]]
    .rename(columns={
        "N": "N_M1",
        "beta_cardiac": "beta_M1",
        "p_cardiac": "p_M1",
        "q_cardiac": "q_M1"
    })
)

m2 = (
    df[df["model"] == "M2_vascular"]
    [key + ["N", "beta_cardiac", "p_cardiac", "q_cardiac"]]
    .rename(columns={
        "N": "N_M2",
        "beta_cardiac": "beta_M2",
        "p_cardiac": "p_M2",
        "q_cardiac": "q_M2"
    })
)

m3 = (
    df[df["model"] == "M3_APOE4"]
    [
        key + [
            "N",
            "beta_cardiac",
            "p_cardiac",
            "q_cardiac",
            "beta_interaction",
            "p_interaction",
            "q_interaction"
        ]
    ]
    .rename(columns={
        "N": "N_M3",
        "beta_cardiac": "beta_M3",
        "p_cardiac": "p_M3",
        "q_cardiac": "q_M3"
    })
)

out = (
    m1
    .merge(m2, on=key, how="outer")
    .merge(m3, on=key, how="outer")
)

out["same_direction_M1_M2"] = (
    np.sign(out["beta_M1"])
    == np.sign(out["beta_M2"])
)

out["same_direction_M1_M2_M3"] = (
    (np.sign(out["beta_M1"]) == np.sign(out["beta_M2"]))
    &
    (np.sign(out["beta_M2"]) == np.sign(out["beta_M3"]))
)

# cardiac effect in APOE4 noncarrier
out["slope_APOE4_noncarrier"] = out["beta_M3"]

# cardiac effect in APOE4 carrier
out["slope_APOE4_carrier"] = (
    out["beta_M3"]
    + out["beta_interaction"]
)

out["slope_difference_carrier_minus_noncarrier"] = (
    out["beta_interaction"]
)

# ============================================================
# 2. Mark the primary surviving results
# ============================================================

out["M2_FDR_sig"] = (
    out["q_M2"] < 0.05
)

out["APOE_interaction_FDR_sig"] = (
    out["q_interaction"] < 0.05
)

primary = out[
    out["M2_FDR_sig"]
    |
    out["APOE_interaction_FDR_sig"]
].copy()

primary = primary.sort_values(
    [
        "APOE_interaction_FDR_sig",
        "M2_FDR_sig",
        "q_interaction",
        "q_M2"
    ],
    ascending=[
        False,
        False,
        True,
        True
    ]
)

out.to_csv(
    OUTDIR / "tier2_model_stability_all.csv",
    index=False
)

primary.to_csv(
    OUTDIR / "tier2_primary_followup.csv",
    index=False
)

# ============================================================
# 3. APOE4-specific interaction summary
# ============================================================

apoe = primary[
    primary["APOE_interaction_FDR_sig"]
].copy()

cols = [
    "metric_type",
    "family",
    "region",
    "cardiac",
    "N_M3",
    "beta_M3",
    "beta_interaction",
    "p_interaction",
    "q_interaction",
    "slope_APOE4_noncarrier",
    "slope_APOE4_carrier"
]

apoe[cols].to_csv(
    OUTDIR / "APOE4_significant_slopes.csv",
    index=False
)

# ============================================================
# 4. Simple APOE slope figure
# ============================================================

if len(apoe):

    plot = apoe.copy()

    labels = (
        plot["cardiac"]
        + " → "
        + plot["region"]
    )

    y = np.arange(len(plot))

    fig, ax = plt.subplots(
        figsize=(9, max(4, 0.8 * len(plot)))
    )

    ax.scatter(
        plot["slope_APOE4_noncarrier"],
        y,
        label="APOE4 noncarrier"
    )

    ax.scatter(
        plot["slope_APOE4_carrier"],
        y,
        marker="s",
        label="APOE4 carrier"
    )

    for i in range(len(plot)):

        ax.plot(
            [
                plot.iloc[i]["slope_APOE4_noncarrier"],
                plot.iloc[i]["slope_APOE4_carrier"]
            ],
            [i, i],
            linewidth=1
        )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlabel(
        "Standardized cardiac slope"
    )

    ax.set_title(
        "APOE4 modification of cardiac–brain associations"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTDIR / "APOE4_significant_slopes.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTDIR / "APOE4_significant_slopes.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================
# 5. Print concise results
# ============================================================

print("\nPRIMARY TIER 2 FOLLOW-UP")
print("========================")

print(
    primary[
        [
            "metric_type",
            "family",
            "region",
            "cardiac",
            "beta_M1",
            "beta_M2",
            "beta_M3",
            "q_M2",
            "beta_interaction",
            "q_interaction",
            "slope_APOE4_noncarrier",
            "slope_APOE4_carrier",
            "same_direction_M1_M2_M3"
        ]
    ]
    .to_string(index=False)
)

print("\nAPOE4-SPECIFIC SLOPES")
print("=====================")

if len(apoe):

    print(
        apoe[cols]
        .to_string(index=False)
    )

else:
    print("None.")

print("\nSaved:")
print(OUTDIR)
