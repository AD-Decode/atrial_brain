#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import chi2

ROOT = Path("/data/qiallab/Framingham")

PHENO = ROOT / "results/genetics_inventory/neurocardiac_TOPMed_WGS_analysis_ready_EXACT.csv"
GENETICS = ROOT / "results/genetics_pathways/sample_metadata/TOPMed_541_genetics_analysis_ready.tsv"

OUTDIR = ROOT / "results/genetics_pathways/pathway_heart_brain_models"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# FOUR VALIDATED HEART-BRAIN COUPLINGS
# ------------------------------------------------------------
PAIRS = [
    {
        "pair_id": "LVEF_L_middletemporal_vol",
        "label": "LVEF × genetic pathway → L middle temporal volume",
        "cardiac": "LVEF",
        "outcome": "lh_middletemporal_vol",
        "metric_type": "volume",
    },
    {
        "pair_id": "LVESVi_R_superiorfrontal_area",
        "label": "LVESVi × genetic pathway → R superior frontal area",
        "cardiac": "LVESVi",
        "outcome": "rh_superiorfrontal_area",
        "metric_type": "surface_area",
    },
    {
        "pair_id": "LVMASSi_R_lingual_vol",
        "label": "LV mass index × genetic pathway → R lingual volume",
        "cardiac": "LV_MASSi",
        "outcome": "rh_lingual_vol",
        "metric_type": "volume",
    },
    {
        "pair_id": "LVMASSi_R_lingual_area",
        "label": "LV mass index × genetic pathway → R lingual area",
        "cardiac": "LV_MASSi",
        "outcome": "rh_lingual_area",
        "metric_type": "surface_area",
    },
]

PATHWAYS = [
    "adiponectin_adipocyte",
    "cardiac_remodeling_fibrosis",
    "endothelial_vascular",
    "inflammation_innate",
    "insulin_glucose",
    "lipid_APOE_cholesterol",
    "mitochondrial_energetics",
]

GLOBAL_PCS = [f"PC{i}" for i in range(1, 6)]

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)

    ok = np.isfinite(p)
    vals = p[ok]

    if len(vals) == 0:
        return q

    order = np.argsort(vals)
    ranked = vals[order]

    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)

    tmp = np.empty(len(vals))
    tmp[order] = adj
    q[ok] = tmp

    return q


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=1)

    if not np.isfinite(sd) or sd == 0:
        return s * np.nan

    return (s - s.mean()) / sd


def binaryize(s):
    """Convert common binary representations to 0/1."""
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")

        # preserve 0/1
        u = set(vals.dropna().unique())
        if u.issubset({0, 1}):
            return vals.astype(float)

        # Framingham sex coding 1/2 -> 0/1
        if u.issubset({1, 2}):
            return (vals == 2).astype(float)

        return vals.astype(float)

    x = s.astype(str).str.strip().str.lower()

    mapping = {
        "0": 0.0,
        "1": 1.0,
        "no": 0.0,
        "yes": 1.0,
        "false": 0.0,
        "true": 1.0,
        "noncarrier": 0.0,
        "carrier": 1.0,
        "male": 0.0,
        "m": 0.0,
        "female": 1.0,
        "f": 1.0,
    }

    return x.map(mapping)


def detect_merge_key(a, b):
    preferred = [
        "NWD_sample",
        "NWD",
        "wgs_sample",
        "WGS_sample",
        "shareid",
        "dbGaP_Subject_ID",
    ]

    for c in preferred:
        if c in a.columns and c in b.columns:
            return c

    # Find any common column populated with NWD identifiers
    for c in set(a.columns).intersection(b.columns):
        aa = a[c].astype(str)
        bb = b[c].astype(str)

        if aa.str.startswith("NWD").sum() > 400 and bb.str.startswith("NWD").sum() > 400:
            return c

    raise RuntimeError(
        "Could not identify common merge key.\n"
        f"PHENO columns: {list(a.columns)}\n"
        f"GENETICS columns: {list(b.columns)}"
    )


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise RuntimeError(f"None of these columns found: {candidates}")

    return None


# ------------------------------------------------------------
# LOAD + MERGE
# ------------------------------------------------------------
ph = pd.read_csv(PHENO)
ge = pd.read_csv(GENETICS, sep="\t")

print("Phenotype:", ph.shape)
print("Genetics:", ge.shape)

merge_key = detect_merge_key(ph, ge)
print("Merge key:", merge_key)

# only bring genetics-specific columns to avoid duplicate phenotype columns
gen_cols = [merge_key, "family_cluster"] + GLOBAL_PCS

for pathway in PATHWAYS:
    gen_cols += [f"{pathway}_PC{i}" for i in range(1, 6)]

gen_cols = [c for c in gen_cols if c in ge.columns]

d = ph.merge(
    ge[gen_cols],
    on=merge_key,
    how="inner",
    validate="one_to_one"
)

print("Merged:", d.shape)

# ------------------------------------------------------------
# IDENTIFY ORIGINAL M3 VARIABLES
# ------------------------------------------------------------
age_col = pick_col(d, ["age_at_mri"])
sex_col = pick_col(d, ["sex_clinical", "_sex", "sex"])
interval_col = pick_col(d, ["abs_delta_years"])
bmi_col = pick_col(d, ["BMI_nearest_exam"])
sbp_col = pick_col(d, ["SBP_nearest_exam"])
smoke_col = pick_col(d, ["current_smoker_nearest_exam"])
diab_col = pick_col(
    d,
    ["diabetes_history_any", "diabetes_history_nearest_exam"]
)
apoe_col = pick_col(d, ["APOE4_carrier"])

icv_col = pick_col(d, ["IntraCranialVol"])

area_col = pick_col(
    d,
    ["TotalCorticalSurfaceArea"],
    required=False
)

if area_col is None:
    lh_area = pick_col(
        d,
        ["lh_WhiteSurfArea_DesKil_area"],
        required=False
    )
    rh_area = pick_col(
        d,
        ["rh_WhiteSurfArea_DesKil_area"],
        required=False
    )

    if lh_area is not None and rh_area is not None:
        d["TotalCorticalSurfaceArea"] = (
            pd.to_numeric(d[lh_area], errors="coerce") +
            pd.to_numeric(d[rh_area], errors="coerce")
        )
        area_col = "TotalCorticalSurfaceArea"
    else:
        raise RuntimeError("Could not construct TotalCorticalSurfaceArea.")

family_col = pick_col(d, ["family_cluster"])

# binary variables
d["_sex_binary"] = binaryize(d[sex_col])
d["_smoking"] = binaryize(d[smoke_col])
d["_diabetes"] = binaryize(d[diab_col])
d["_APOE4"] = binaryize(d[apoe_col])

print("\nResolved columns:")
print(" age:", age_col)
print(" sex:", sex_col)
print(" interval:", interval_col)
print(" BMI:", bmi_col)
print(" SBP:", sbp_col)
print(" smoking:", smoke_col)
print(" diabetes:", diab_col)
print(" APOE4:", apoe_col)
print(" ICV:", icv_col)
print(" total area:", area_col)
print(" family:", family_col)

# ------------------------------------------------------------
# MODEL FUNCTION
# ------------------------------------------------------------
def run_one_model(data, pair, pathway):

    cardiac = pair["cardiac"]
    outcome = pair["outcome"]

    pathway_pcs = [f"{pathway}_PC{i}" for i in range(1, 6)]

    needed = [
        cardiac, outcome,
        age_col, interval_col, bmi_col, sbp_col,
        "_sex_binary", "_smoking", "_diabetes", "_APOE4",
        family_col,
        *GLOBAL_PCS,
        *pathway_pcs,
    ]

    if pair["metric_type"] == "volume":
        needed.append(icv_col)
    else:
        needed.append(area_col)

    m = data[needed].copy()
    m = m.dropna()

    # --------------------------------------------------------
    # outcome preprocessing exactly following original logic
    # --------------------------------------------------------
    if pair["metric_type"] == "volume":
        raw_outcome = (
            pd.to_numeric(m[outcome], errors="coerce") /
            pd.to_numeric(m[icv_col], errors="coerce")
        )
    else:
        raw_outcome = pd.to_numeric(m[outcome], errors="coerce")

    m["_Y"] = zscore(raw_outcome)
    m["_cardiac"] = zscore(m[cardiac])

    # original continuous covariates standardized in model subset
    m["_age"] = zscore(m[age_col])
    m["_interval"] = zscore(m[interval_col])
    m["_BMI"] = zscore(m[bmi_col])
    m["_SBP"] = zscore(m[sbp_col])

    # population-structure PCs
    for pc in GLOBAL_PCS:
        m[f"_{pc}"] = zscore(m[pc])

    # pathway PCs
    for i, pc in enumerate(pathway_pcs, 1):
        m[f"_PPC{i}"] = zscore(m[pc])

    if pair["metric_type"] == "surface_area":
        m["_total_area"] = zscore(m[area_col])

    # existing APOE4 modifier
    m["_cardiac_x_APOE4"] = m["_cardiac"] * m["_APOE4"]

    # pathway modifier interactions
    for i in range(1, 6):
        m[f"_cardiac_x_PPC{i}"] = m["_cardiac"] * m[f"_PPC{i}"]

    # --------------------------------------------------------
    # BASE MODEL: original M3 + genetic structure PCs
    # --------------------------------------------------------
    base_cols = [
        "_cardiac",
        "_APOE4",
        "_cardiac_x_APOE4",
        "_age",
        "_sex_binary",
        "_interval",
        "_BMI",
        "_SBP",
        "_smoking",
        "_diabetes",
    ] + [f"_{pc}" for pc in GLOBAL_PCS]

    if pair["metric_type"] == "surface_area":
        base_cols.append("_total_area")

    # --------------------------------------------------------
    # PATHWAY MODEL
    # add pathway PC main effects + cardiac interactions
    # --------------------------------------------------------
    ppc_main = [f"_PPC{i}" for i in range(1, 6)]
    ppc_int = [f"_cardiac_x_PPC{i}" for i in range(1, 6)]

    full_cols = base_cols + ppc_main + ppc_int

    X0 = sm.add_constant(m[base_cols], has_constant="add")
    X1 = sm.add_constant(m[full_cols], has_constant="add")
    y = m["_Y"]

    # same subjects in baseline and pathway models
    base_ols = sm.OLS(y, X0)
    full_ols = sm.OLS(y, X1)

    # HC3
    base_hc3 = base_ols.fit(cov_type="HC3")
    full_hc3 = full_ols.fit(cov_type="HC3")

    # family-cluster robust
    groups = m[family_col].astype(str)

    base_cluster = base_ols.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True
        }
    )

    full_cluster = full_ols.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True
        }
    )

    # --------------------------------------------------------
    # 5-df omnibus Wald test for pathway interactions
    # --------------------------------------------------------
    def omnibus(result):

        names = list(result.params.index)
        R = np.zeros((5, len(names)))

        for j, term in enumerate(ppc_int):
            R[j, names.index(term)] = 1

        test = result.wald_test(R, scalar=True)

        return float(test.statistic), float(test.pvalue)

    stat_cluster, p_cluster = omnibus(full_cluster)
    stat_hc3, p_hc3 = omnibus(full_hc3)

    # baseline APOE4 interaction coefficient
    b0 = float(base_cluster.params["_cardiac_x_APOE4"])
    b1 = float(full_cluster.params["_cardiac_x_APOE4"])

    attenuation = np.nan
    if b0 != 0:
        attenuation = 100 * (abs(b1) - abs(b0)) / abs(b0)

    summary = {
        "pair_id": pair["pair_id"],
        "pair_label": pair["label"],
        "cardiac": cardiac,
        "outcome": outcome,
        "metric_type": pair["metric_type"],
        "pathway": pathway,
        "N": len(m),
        "N_clusters": groups.nunique(),
        "N_APOE4_carriers": int(m["_APOE4"].sum()),
        "omnibus_df": 5,
        "omnibus_Wald_cluster": stat_cluster,
        "omnibus_P_cluster": p_cluster,
        "omnibus_Wald_HC3": stat_hc3,
        "omnibus_P_HC3": p_hc3,
        "baseline_cardiac_APOE4_beta": b0,
        "pathway_model_cardiac_APOE4_beta": b1,
        "change_abs_APOE4_beta_pct": attenuation,
        "R2_base": base_cluster.rsquared,
        "R2_pathway": full_cluster.rsquared,
        "delta_R2": full_cluster.rsquared - base_cluster.rsquared,
    }

    details = []

    for i, term in enumerate(ppc_int, 1):
        details.append({
            "pair_id": pair["pair_id"],
            "pathway": pathway,
            "component": f"PC{i}",
            "term": term,
            "N": len(m),

            "beta_cluster": float(full_cluster.params[term]),
            "SE_cluster": float(full_cluster.bse[term]),
            "P_cluster": float(full_cluster.pvalues[term]),

            "beta_HC3": float(full_hc3.params[term]),
            "SE_HC3": float(full_hc3.bse[term]),
            "P_HC3": float(full_hc3.pvalues[term]),
        })

    return summary, details


# ------------------------------------------------------------
# RUN 7 x 4 = 28 PRIMARY TESTS
# ------------------------------------------------------------
summaries = []
details = []

for pair in PAIRS:
    for pathway in PATHWAYS:

        print(
            f"\nRunning {pair['pair_id']} | {pathway}"
        )

        s, det = run_one_model(d, pair, pathway)
        summaries.append(s)
        details.extend(det)

summary = pd.DataFrame(summaries)
detail = pd.DataFrame(details)

# FDR across exactly 28 prespecified pathway x coupling tests
summary["q_cluster_primary28"] = bh_fdr(
    summary["omnibus_P_cluster"]
)

summary["q_HC3_primary28"] = bh_fdr(
    summary["omnibus_P_HC3"]
)

summary = summary.sort_values(
    ["q_cluster_primary28", "omnibus_P_cluster"]
)

# descriptive component-level FDR -- secondary only
detail["q_component_cluster_all140"] = bh_fdr(
    detail["P_cluster"]
)

detail["q_component_HC3_all140"] = bh_fdr(
    detail["P_HC3"]
)

SUMMARY_OUT = OUTDIR / "pathway_omnibus_primary28.tsv"
DETAIL_OUT = OUTDIR / "pathway_component_interactions_secondary.tsv"

summary.to_csv(SUMMARY_OUT, sep="\t", index=False)
detail.to_csv(DETAIL_OUT, sep="\t", index=False)

print("\n============================================================")
print("PRIMARY 28 PATHWAY × HEART-BRAIN OMNIBUS TESTS")
print("============================================================")

show = [
    "pair_id",
    "pathway",
    "N",
    "N_APOE4_carriers",
    "omnibus_Wald_cluster",
    "omnibus_P_cluster",
    "q_cluster_primary28",
    "delta_R2",
    "baseline_cardiac_APOE4_beta",
    "pathway_model_cardiac_APOE4_beta",
    "change_abs_APOE4_beta_pct",
]

print(
    summary[show]
    .round(6)
    .to_string(index=False)
)

print("\nFDR-significant pathway modifiers (q < 0.05):")
sig = summary[summary["q_cluster_primary28"] < 0.05]

if len(sig):
    print(
        sig[show]
        .round(6)
        .to_string(index=False)
    )
else:
    print("None.")

print("\nNominal pathway signals (P < 0.05):")
nom = summary[summary["omnibus_P_cluster"] < 0.05]

if len(nom):
    print(
        nom[show]
        .round(6)
        .to_string(index=False)
    )
else:
    print("None.")

print("\nWrote:")
print(" ", SUMMARY_OUT)
print(" ", DETAIL_OUT)
