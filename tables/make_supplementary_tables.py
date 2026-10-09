#!/usr/bin/env python3
"""
Reproducible build of Supplementary Tables S1-S25 (manuscript numbering).

What it does
  1. (--run) runs the analysis script(s) that produce each table, in order, from the code directory
     (skips a table whose sources already exist unless --force). Slow analyses (spatial nulls, WGS)
     are run only when named with --only.
  2. Collects each table's source file(s) into OUTDIR/TableSxx/ under manuscript numbering
     (the producing scripts still write files with superseded numbers - the mapping lives here),
     and writes one workbook with a sheet per table/panel.
  3. Builds three tables directly so they cannot drift from the verified analyses:
       S2  = 16-test longitudinal family + random-slope columns (script 155 output)
       S8  = regional associations meeting within-family q<0.05, from the full-cohort tier2 file
       S9  = stratified slopes from Figure3A_plot_data.csv (make_Figure3.py, verified 36/36)
  4. Verifies every table against the Supplementary Materials document: each number printed in the
     document's Table Sx must be found (to its printed precision) in the collected source(s).
     Coverage < 100% is reported per table with the unmatched numbers.
  5. Refuses to copy any file with a participant-identifier column.

Usage
  python make_supplementary_tables.py --supplement Framingham_Supplementary_Materials_100726_clean.docx
  python make_supplementary_tables.py --run --only S10 S14      # rerun producers for selected tables
  python make_supplementary_tables.py --strict ...              # exit 1 unless every table verifies 100%
"""
import argparse, glob, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
CODE = ROOT / "code"
R = ROOT / "results"
REPO_OLD = ROOT / "FINAL_MANUSCRIPT_REPO_20260928/SupplementaryTables"
TP = R / "longitudinal_heart_brain/temporal_prediction"
TIER2 = R / "tier2_corrected/tier2_corrected_all.csv"
REBUILD = R / "final_rebuild_20261007"
SLOW = {"S19", "S20", "S21", "S22", "S23", "S24", "S25"}

# table -> (title, producer scripts in run order, {panel: source path or glob})
MANIFEST = {
 "S1": ("LA remodeling × APOE ε4: principal interaction, carrier-stratified slopes, genotype",
        ["104_temporal_mixedlm_random_intercept.py", "108_temporal_GEE_and_simple_slopes.py",
         "113_APOE4_dose_longitudinal_sensitivity.py", "114_APOE4_categorical_dose_sensitivity.py",
         "115_finalize_longitudinal_tables.py", "159_longitudinal_genotype_restricted_audit.py"],
        {"SL1_interaction": "Table_SL1_Primary_longitudinal_interaction_and_GEE.tsv",
         "SL2_stratified": "Table_SL2_APOE4_stratified_simple_slopes.tsv",
         "SL4A_dose": "Table_SL4A_APOE4_ordinal_dose_interaction.tsv",
         "SL4B_dose_slopes": "Table_SL4B_APOE4_ordinal_dose_simple_slopes.tsv",
         "SL5A_genotype": "Table_SL5A_APOE4_categorical_dose_joint_test.tsv",
         "SL5B_genotype_slopes": "Table_SL5B*.tsv",
         "E_genotype_restricted_slopes": TP / "genotype_restricted_audit/restricted_E3E3_E3E4_E4E4_slopes.csv",
         "E_genotype_restricted_terms": TP / "genotype_restricted_audit/restricted_E3E3_E3E4_E4E4_model_terms.csv"}),
 "S2": ("Complete 16-test longitudinal testing family (random intercept and random slope)",
        ["154_build_missing_supplementary_tables.py", "155_longitudinal_random_time_slope_all16.py"],
        {"BUILD": "S2"}),
 "S3": ("Longitudinal model-specification, exposure, head-size, vascular and antihypertensive sensitivity",
        ["156_make_TableS17_longitudinal_robustness.py", "157_patch_TableS17_add_vascular.py"],
        {"main": REPO_OLD / "TableS17/TableS17_LongitudinalRobustness.csv",
         "antihypertensive": "CONTENT:M4_plus_HTNtx"}),
 "S4": ("Availability of subsequent brain MRI in the pre-MRI at-risk cohort",
        ["152_build_attrition_source.py", "153_longitudinal_LA_attrition_descriptive.py"],
        {"overall": TP / "attrition_IPW/TableS18_PanelC_overall_MRI_availability.csv",
         "by_APOE_LA": TP / "attrition_IPW/TableS18_PanelC_APOE4_LAtertile.csv",
         "descriptive": TP / "attrition_IPW/TableS18_attrition_descriptive.xlsx"}),
 "S5": ("Selection model and inverse-probability-of-selection weighting",
        ["152_build_attrition_source.py", "153_attrition_IPSW_GEE.py", "154_make_S17_attrition_IPSW_table.py"],
        {"A_selection": TP / "attrition_IPSW/Table_S17A_attrition_selection_model.tsv",
         "B_IPSW": TP / "attrition_IPSW/Table_S17B_IPSW_effect_comparison.tsv"}),
 "S6": ("Antecedent cardiac trajectory × APOE ε4 → later plasma biomarkers",
        ["158_build_TableS23_S24_biomarker_cognition.py"],
        {"main": REPO_OLD / "TableS23/TableS23_CardiacTrajectories_PlasmaBiomarkers.csv"}),
 "S7": ("Antecedent cardiac trajectory × APOE ε4 → later cognition",
        ["158_build_TableS23_S24_biomarker_cognition.py"],
        {"A_tests": REPO_OLD / "TableS24/TableS24A_CardiacTrajectories_LaterCognition.csv",
         "B_MMSE": REPO_OLD / "TableS24/TableS24B_CardiacTrajectories_LongitudinalMMSE.csv"}),
 "S8": ("Regional associations meeting the within-family FDR criterion", ["32_tier2_corrected_scaling.py"], {"BUILD": "S8"}),
 "S9": ("APOE ε4-stratified slopes for the four regional ventricular associations", [], {"BUILD": "S9"}),
 "S10": ("Metabolic sensitivity of the four regional ventricular interactions",
         ["62_make_TableS7_metabolic_robustness.py"],
         {"main": R / "paper_figures_tables/SUBMISSION_PACKAGE/04_Supplementary_Tables/TableS7_metabolic_robustness.xlsx"}),
 "S11": ("LAVI × APOE ε4 in all available atrial participants",
         ["39_atrial_APOE4_exact.py", "42_atrial_independence_LV.py", "58_make_Table4_LAVI_independence_FINAL.py"],
         {"main": R / "paper_figures_tables/atrial_APOE4/TableS2_LAVI_APOE4_independence_all_sample.csv"}),
 "S12": ("Plasma-biomarker adjustment of the regional ventricular interactions",
         ["62_matched_completecase_AD_biomarkers.py", "160_build_TableS25_biomarker_adjustment.py"],
         {"main": REPO_OLD / "TableS25/TableS25_BiomarkerAdjustment_PrimaryCardiacBrainInteractions.csv"}),
 "S13": ("Cross-sectional cardiac phenotype × APOE ε4 → plasma biomarkers",
         ["159_rebuild_TableS20_direct_biomarkers.py"],
         {"main": REPO_OLD / "TableS20/TableS20_DirectCardiac_APOE4_PlasmaBiomarkers.csv"}),
 "S14": ("Atrial fibrillation/flutter sensitivity",
         ["61_AF_flutter_sensitivity.py", "62_make_AF_flutter_supplementary.py"],
         {"main": R / "supplementary_AF_flutter/SupplementaryTableS4_AF_flutter_sensitivity_formatted.csv",
          "raw": R / "AF_flutter_sensitivity/AF_flutter_primary4_sensitivity.csv"}),
 "S15": ("Pedigree-family cluster-robust inference",
         ["156_corrected_pedigree_cluster_robust_sensitivity.py", "157_rebuild_corrected_TableS3_pedigree_cluster.py",
          "158_corrected_HbA1c_pedigree_cluster_sensitivity.py"],
         {"A_ventricular": R / "kinship_sensitivity_corrected/TableS3_corrected_primary_pedigree_cluster.csv",
          "B_LAVI": R / "kinship_sensitivity_corrected/LAVI_middle_occipital_corrected_pedigree_sensitivity.csv",
          "C_HbA1c": R / "kinship_sensitivity_corrected/LVEDVi_APOE4_HbA1c_M3_corrected_pedigree_sensitivity.csv",
          "diagnostics": R / "kinship_sensitivity_corrected/TableS3_corrected_primary_pedigree_diagnostics.csv"}),
 "S16": ("HbA1c × APOE ε4 → LV end-diastolic volume index", ["56_validate_LVEDVi_APOE4_HbA1c.py"],
         {"main": R / "LVEDVi_APOE4_HbA1c_validation/LVEDVi_APOE4_HbA1c_validation_models.csv"}),
 "S17": ("Exploratory APOE ε4 × metabolic-modifier analyses", ["55_APOE4_metabolic_discovery_heart_wholebrain.py"],
         {"A_family_summary": R / "APOE4_metabolic_discovery_heart_wholebrain/APOE4_metabolic_discovery_family_summary.csv",
          "B_top": R / "APOE4_metabolic_discovery_heart_wholebrain/APOE4_metabolic_discovery_top20_per_family.csv"}),
 "S18": ("Exploratory whole-brain analyses of three-dimensional myocardial strain", ["41_LV_RV_3D_strain_analysis.py"],
         {"main": R / "LV_RV_3D_strain/LV_RV_3D_analysis_summary.csv"}),
 "S19": ("AHBA regional enrichment", ["82_build_AHBA_DK_expression.py", "89_primary_regions_nonranked_enrichment_table_figure.py"],
         {"main": R / "transcriptomic_enrichment/Table_AHBA_primary_regions_nonranked_enrichment.csv"}),
 "S20": ("Summary of molecular analyses", ["79_pathway_omnibus_heart_brain.py", "91_make_mechanistic_triangulation_table.py"],
         {"main": R / "paper_figures_tables/TableS8_mechanistic_triangulation.csv",
          "primary_interactions": REBUILD / "Tables2_4/Table2.csv"}),
 "S21": ("Targeted AHBA pathway spatial correspondence", ["80_make_wholebrain_coupling_maps.py", "81_split_coupling_maps_by_atlas.py",
         "83b_build_DK56_distance_cache.py", "84b_spatial_null_FAST.py"],
         {"main": R / "transcriptomic_enrichment/pathway_expression_vs_primary_interaction_maps_SPATIAL_NULL_FAST.tsv"}),
 "S22": ("GO Biological Process and Reactome spatial correspondence", ["90b_GO_Reactome_spatial_coupling_FAST.py"],
         {"main": R / "transcriptomic_enrichment/GO_Reactome_spatial_coupling_FAST_N*_TOP25.tsv"}),
 "S23": ("MSigDB Hallmark spatial correspondence", ["92_MSigDB_Hallmark_spatial.py"],
         {"main": R / "transcriptomic_enrichment/MSigDB_Hallmark_spatial_N5000_ALL.tsv"}),
 "S24": ("Brain cell-class spatial correspondence", ["93b_select_C8_brain_specific.py", "94_MSigDB_C8_brain_spatial.py"],
         {"main": R / "transcriptomic_enrichment/MSigDB_C8_cellclass_N5000_ALL.tsv"}),
 "S25": ("Cell-type signature spatial correspondence", ["93b_select_C8_brain_specific.py", "94_MSigDB_C8_brain_spatial.py"],
         {"main": R / "transcriptomic_enrichment/MSigDB_C8_brain_individual_N5000_ALL.tsv"}),
}
# ---- manuscript numbering after the 8 Oct 2026 consolidation: new table -> analysis units above (old numbering)
FINAL = {"S1": ["S1"], "S2": ["S2"], "S3": ["S3"], "S4": ["S4", "S5"], "S5": ["S6", "S7"], "S6": ["S8"],
         "S7": ["S10", "S15"], "S8": ["S16"], "S9": ["S17"], "S10": ["S19"], "S11": ["S20"]}
DATA_S2 = {"Stratified_slopes": "S9", "LAVI_all_available": "S11", "Biomarker_adjustment": "S12",
           "XS_cardiac_biomarkers": "S13", "AF_flutter": "S14", "Strain_3D": "S18", "Spatial_targeted": "S21",
           "Spatial_GO_Reactome": "S22", "Spatial_Hallmark": "S23", "Spatial_cell_class": "S24", "Spatial_cell_signature": "S25"}
ID = re.compile(r"^(shareid|share_id|dbgap_subject_id|subject_id|subjid|framid|pid)$", re.I)


# ------------------------------------------------------------------ reading
def read_any(p):
    p = Path(p)
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=None)
    sep = "\t" if p.suffix in (".tsv", ".txt") else ","
    if p.stat().st_size == 0:
        print(f"   EMPTY file skipped: {p}")
        return {}
    try:
        return {"": pd.read_csv(p, sep=sep)}
    except pd.errors.EmptyDataError:
        print(f"   EMPTY file skipped: {p}")
        return {}


_FOUND = {}
SEARCH_BASES = None


def resolve_content(token):
    """tsv/csv files under the results tree whose text contains `token` (most recent first)"""
    hits = []
    for p in list(R.rglob("*.tsv")) + list(R.rglob("*.csv")):
        try:
            if p.stat().st_size and p.stat().st_size < 5_000_000 and token in p.read_text(errors="ignore"):
                hits.append(p)
        except OSError:
            pass
    hits = sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[:1]
    if hits:
        print(f"   located content '{token}' -> {hits[0]}")
    return hits


def resolve(src):
    """Paths for src: the given path/glob if it exists, otherwise a search by file name under ROOT
    (files from the September package moved). The location used is printed once."""
    src = str(src)
    if src.startswith("CONTENT:"):
        return resolve_content(src[len("CONTENT:"):])
    hits = sorted(glob.glob(src))
    if hits:
        return [Path(h) for h in hits]
    name = Path(src).name
    if name.startswith("*"):
        return []                      # a bare wildcard is not a file name - do not search the tree for it
    if name in _FOUND:
        return _FOUND[name]
    found = []
    # search results/ and the manuscript-package folders only (data/ and downloads/ hold large raw files)
    bases = SEARCH_BASES or ([R] + sorted(p for p in ROOT.glob("*") if p.is_dir() and ("REPO" in p.name.upper() or "MANUSCRIPT" in p.name.upper())))
    for base in bases:
        found = sorted(p for p in base.rglob(name)
                       if "final_rebuild" not in str(p) and "SupplementaryTables_S" not in str(p))
        if found:
            break
    if found:
        # prefer the most recently modified copy
        found = [max(found, key=lambda p: p.stat().st_mtime)]
        print(f"   located {name} -> {found[0]}")
    _FOUND[name] = found
    return found


def one(src, what):
    hits = resolve(src)
    if not hits:
        raise FileNotFoundError(f"{what}: {Path(str(src)).name} not found under {ROOT}")
    return hits[0]


# ------------------------------------------------------------------ built tables
def build_S2():
    card = {"LA_dimension": "LA dimension", "LV_mass": "LV mass", "LVDD": "LV end-diastolic dimension", "fractional_shortening": "Fractional shortening"}
    outc = {"hippocampus": "Hippocampal volume", "total_brain": "Total brain volume", "lateral_ventricles": "Lateral ventricular volume",
            "WMH": "White matter hyperintensity volume"}
    base = resolve(REPO_OLD / "TableS21/TableS21_CompleteLongitudinalTestingFamily.csv")
    if base:
        s2 = pd.read_csv(base[0])
    else:   # rebuild from the principal model output (script 104)
        pr = pd.read_csv(one(TP / "PRIMARY_APOE4_temporal_interactions.tsv", "S2 principal models"), sep="\t")
        pr = pr[(pr["subset"] == "prospective") & pr["term"].astype(str).str.contains("APOE4")]
        s2 = pd.DataFrame({"Cardiac trajectory": pr["cardiac"].map(card), "Brain outcome": pr["outcome"].map(outc),
                           "N subjects": pr["N_subjects"], "N MRI observations": pr["N_observations"],
                           "N APOE ε4 carriers": pr["N_APOE4_carrier"], "N APOE ε4 noncarriers": pr["N_APOE4_noncarrier"],
                           "β, trajectory × MRI time × APOE ε4": pr["beta"], "SE": pr["SE"], "95% CI lower": pr["CI_low"],
                           "95% CI upper": pr["CI_high"], "P": pr["P"], "FDR q": pr["FDR_q"]}).sort_values("P")
    rs = pd.read_csv(one(TP / "mixedlm_random_time_slope/PRIMARY_APOE4_temporal_interactions_random_time_slope.tsv", "S2 random slopes"), sep="\t")
    rs["Cardiac trajectory"] = rs["cardiac"].map(card); rs["Brain outcome"] = rs["outcome"].map(outc)
    for c in ("Cardiac trajectory", "Brain outcome"):
        miss = set(rs[c].dropna()) - set(s2[c]) | (set() if rs[c].notna().all() else {"<unmapped>"})
        if miss:
            raise ValueError(f"S2: label mismatch in {c}: {miss}; Table S2 has {sorted(set(s2[c]))}")
    rs = rs.rename(columns={"beta": "β, random slope", "CI_low": "95% CI lower, random slope", "CI_high": "95% CI upper, random slope",
                            "P": "P, random slope", "FDR_q_random_slope": "FDR q, random slope"})
    keep = ["Cardiac trajectory", "Brain outcome", "β, random slope", "95% CI lower, random slope",
            "95% CI upper, random slope", "P, random slope", "FDR q, random slope"]
    return {"main": s2.merge(rs[keep], on=["Cardiac trajectory", "Brain outcome"], how="left", validate="one_to_one")}


def build_S8():
    t = pd.read_csv(one(TIER2, "S8 tier2"))
    hit_main = t[t["q_cardiac"] < 0.05][["region", "cardiac"]]           # main effect in any model (M1, M2 or M3)
    hit_int = t[(t["model"] == "M3_APOE4") & (t["q_interaction"] < 0.05)][["region", "cardiac"]]
    pairs = pd.concat([hit_main, hit_int]).drop_duplicates()
    rows = t.merge(pairs, on=["region", "cardiac"])
    cols = ["model", "metric_type", "family", "region", "cardiac", "N", "beta_cardiac", "CI_low", "CI_high", "p_cardiac", "q_cardiac",
            "beta_interaction", "CI_interaction_low", "CI_interaction_high", "p_interaction", "q_interaction"]
    return {"all_models_for_listed_pairs": rows[[c for c in cols if c in rows]].sort_values(["region", "cardiac", "model"])}


def build_S9():
    f = REBUILD / "Figure3/Figure3A_plot_data.csv"
    if not f.exists():
        raise FileNotFoundError(f"S9 is built from {f}; run make_Figure3.py first")
    d = pd.read_csv(f)
    from scipy.stats import norm
    for k in ("nc", "car"):          # stratified-slope P values (Wald, from the HC3 CI) as printed in Table S9
        se = (d[f"{k}_ci_high"] - d[f"{k}_ci_low"]) / (2 * 1.959964)
        d[f"{k}_p"] = 2 * norm.sf(np.abs(d[f"{k}_beta"] / se))
    return {"main": d}


BUILDERS = {"S2": build_S2, "S8": build_S8, "S9": build_S9}


# ------------------------------------------------------------------ verification against the supplement
SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
NUM = re.compile(r"[−\-]?\d+(?:\.\d+)?(?:\s*[×x]\s*10\s*(?:\^)?\s*[⁻\-−]?[⁰¹²³⁴⁵⁶⁷⁸⁹\d]+)?")


def tokens(text):
    out = []
    for m in NUM.finditer(text):
        s = m.group(0).replace("−", "-").replace(" ", "")
        if "×" in s or "x10" in s:
            mant, exp = re.split(r"[×x]10\^?", s)
            exp = exp.translate(SUP).replace("−", "-")
            try:
                out.append((float(mant) * 10 ** int(exp), None, len(mant.split(".")[-1]) if "." in mant else 0))
            except ValueError:
                continue
        else:
            if "." not in s:
                continue                                  # skip integers (Ns, counts, years) - checked via decimals
            dec = len(s.split(".")[1])          # 1-decimal numbers (percentages) included
            out.append((float(s), dec, None))
    return out


def supplement_tables(docx_path):
    """{'S1': [text of each docx table belonging to Table S1], ...}
    Reads the .docx XML directly (standard library only; python-docx is not needed)."""
    import zipfile
    import xml.etree.ElementTree as ET
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(docx_path) as z:
        body = ET.fromstring(z.read("word/document.xml")).find(W + "body")

    def text(el):
        return "".join(t.text or "" for t in el.iter(W + "t"))

    current, res = None, {}
    for el in body:
        if el.tag == W + "p":
            m = re.match(r"\s*Table (S\d+)\.", text(el))
            if m:
                current = m.group(1)
        elif el.tag == W + "tbl":
            cells = [text(tc) for tc in el.iter(W + "tc")]
            m = re.match(r"\s*Table (S\d+)\.", cells[0]) if cells else None
            key = m.group(1) if m else current
            if key:
                res.setdefault(key, []).append("\n".join(cells))
    return res


def verify(doc_texts, frames):
    vals = []
    for df in frames:
        vals.append(pd.to_numeric(df.stack(), errors="coerce").dropna().to_numpy(float))
        for col in df.select_dtypes(exclude="number").columns:      # numbers inside formatted strings
            for s in df[col].dropna().astype(str):
                vals.append(np.array([v for v, _, _ in tokens(s)], float))
    v = np.concatenate(vals) if vals else np.array([])
    found, missing = 0, []
    toks = [t for txt in doc_texts for t in tokens(txt)]
    for val, dec, mdec in toks:
        if dec is not None:
            ok = np.any(np.abs(np.round(v, dec) - val) <= 0.5 * 10 ** -dec + 1e-12) or np.any(np.abs(np.round(v * 100, dec) - val) <= 0.5 * 10 ** -dec)
        else:
            ok = np.any(np.isclose(v, val, rtol=0.5 * 10 ** -max(mdec, 1) * 2, atol=0))
        found += ok
        if not ok:
            missing.append(val)
    return len(toks), found, missing


def set_root(new, keep_original=()):
    """Point manifest paths at the workspace; molecular units (not rerun) and final_rebuild paths keep their location."""
    global ROOT, R, REPO_OLD, TP, TIER2, SEARCH_BASES
    old = str(ROOT)
    def sw(p):
        if not isinstance(p, Path) or "final_rebuild" in str(p):
            return p
        return Path(str(p).replace(old, str(new), 1))
    for k, (title, prod, src) in MANIFEST.items():
        if k in keep_original:
            continue
        MANIFEST[k] = (title, prod, {pn: sw(v) for pn, v in src.items()})
    ROOT, R, REPO_OLD, TP, TIER2 = new, sw(R), sw(REPO_OLD), sw(TP), sw(TIER2)
    SEARCH_BASES = [new]
    print(f"Collecting from analysis root {new}; existing outputs used for {sorted(keep_original)}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supplement", help="Supplementary Materials .docx used for verification")
    ap.add_argument("--outdir", default=str(REBUILD / "SupplementaryTables"))
    ap.add_argument("--code-dir", default=str(CODE))
    ap.add_argument("--run", action="store_true", help="run producer scripts")
    ap.add_argument("--force", action="store_true", help="rerun even if sources exist")
    ap.add_argument("--only", nargs="*", help="limit to these final tables, e.g. S2 S7")
    ap.add_argument("--no-data-s2", dest="data_s2", action="store_false", help="skip the Data S2 workbook")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--existing-ok", nargs="*", default=[], help="units taken from the original results tree")
    ap.add_argument("--root", help="analysis root to collect from (sandbox written by run_supplementary_pipeline.py)")
    ap.add_argument("--fresh-since", type=float, default=None,
                    help="epoch seconds; sources modified before this are marked STALE (set by run_supplementary_pipeline.py)")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    if a.root:
        set_root(Path(a.root), keep_original=set(a.existing_ok))
    tables = a.only or list(FINAL)
    if a.supplement and not Path(a.supplement).exists():
        sys.exit(f"Supplement not found: {Path(a.supplement).resolve()}\n"
                 "Upload the Supplementary Materials .docx to this folder or pass its full path.")
    doc = supplement_tables(a.supplement) if a.supplement else {}
    report, sheets = [], {}

    units = {}                                   # analysis unit (old number) -> (frames, status)
    needed = sorted({u for t in tables for u in FINAL[t]} | (set(DATA_S2.values()) if a.data_s2 else set()),
                    key=lambda k: int(k[1:]))
    for t in needed:
        title, producers, sources = MANIFEST[t]
        if a.run and (a.force or not all(resolve(s) for s in sources.values() if not isinstance(s, str))):
            if t in SLOW and not a.only:
                print(f"{t}: slow analysis - run explicitly with --run --only")
            else:
                for scr in producers:
                    print(f"{t}: running {scr}")
                    r = subprocess.run([sys.executable, str(Path(a.code_dir) / scr)], cwd=a.code_dir)
                    if r.returncode != 0:
                        sys.exit(f"{t}: {scr} failed")
        frames, status = {}, "ok"
        if "BUILD" in sources:
            try:
                frames = BUILDERS[t]()
                if a.fresh_since:
                    stale = [n for n, ps in _FOUND.items() for p in ps if p.stat().st_mtime < a.fresh_since]
                    if t == "S2" and any("TableS21" in n or "random_time_slope" in n for n in stale):
                        status = "STALE: S2 inputs not regenerated"
                    if t == "S8" and (one(TIER2, "S8").stat().st_mtime < a.fresh_since):
                        status = "STALE: tier2 not regenerated"
            except Exception as e:
                status = f"FAILED: {e}"
        else:
            for panel, src in sources.items():
                hits = resolve(src)
                if not hits:
                    status = f"missing: {Path(str(src)).name}"; continue
                for h in hits:
                    if a.fresh_since and h.stat().st_mtime < a.fresh_since and t not in a.existing_ok:
                        status = f"STALE: {h.name} not regenerated in this run"
                    for sheet, df in read_any(h).items():
                        frames[f"{panel}{'_' + sheet if sheet else ''}"] = df
        for k, df in frames.items():
            if any(ID.match(str(c)) for c in df.columns):
                sys.exit(f"{t}/{k}: participant-identifier column found - not copied")
        units[t] = (frames, status)

    for t in tables:                              # final numbering
        frames, statuses = {}, []
        for u in FINAL[t]:
            fr, st = units[u]
            frames.update({f"{u}_{k}": v for k, v in fr.items()}); statuses.append(st if st == "ok" else f"{u} {st}")
        status = "ok" if all(s == "ok" for s in statuses) else "; ".join(s for s in statuses if s != "ok")
        dest = out / f"Table{t}"; dest.mkdir(exist_ok=True)
        for k, df in frames.items():
            df.to_csv(dest / f"Table{t}_{k}.csv", index=False)
            sheets[f"{t}_{k}"[:31]] = df
        n_tok, n_found, missing = verify(doc.get(t, []), list(frames.values())) if doc else (0, 0, [])
        cov = (n_found / n_tok) if n_tok else np.nan
        report.append({"table": t, "built_from": "+".join(FINAL[t]), "panels": len(frames), "status": status, "doc_numbers": n_tok,
                       "matched": n_found, "coverage": round(cov, 3) if n_tok else "",
                       "unmatched_examples": ", ".join(f"{m:g}" for m in missing[:8])})
        print(f"{t:4s} ({'+'.join(FINAL[t]):8s}) panels={len(frames)} {status[:60]:10s} coverage={report[-1]['coverage']} {report[-1]['unmatched_examples']}")

    if a.data_s2:                                  # Supplementary Data S2: tables moved out of the print supplement
        with pd.ExcelWriter(out / "Supplementary_Data_S2_tables.xlsx") as w:
            pd.DataFrame([{"sheet": k, "former table": v} for k, v in DATA_S2.items()]).to_excel(w, sheet_name="README", index=False)
            for name, u in DATA_S2.items():
                fr, st = units[u]
                for i, (k, df) in enumerate(fr.items()):
                    df.to_excel(w, sheet_name=(name if len(fr) == 1 else f"{name}_{i+1}")[:31], index=False)
        print("Data S2 tables written (figure values come from the figure scripts' *_plot_data.csv files)")

    with pd.ExcelWriter(out / "SupplementaryTables_S1-S11.xlsx") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    rep = pd.DataFrame(report); rep.to_csv(out / "SupplementaryTables_verification.csv", index=False)
    print(f"\nwritten to {out}")
    bad = rep[(rep["status"] != "ok") | (rep["coverage"].astype(str).ne("1.0") if doc else False)]
    if a.strict and len(bad):
        sys.exit(f"STRICT: {len(bad)} table(s) missing or not fully matched - see SupplementaryTables_verification.csv")


if __name__ == "__main__":
    main()
