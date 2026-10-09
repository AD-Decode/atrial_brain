#!/usr/bin/env python3
"""
Clean regeneration of all analyses behind Supplementary Tables S1-S25, then collection + verification.

Stages (run in this order; each script's output is logged to OUTDIR/logs/):
  data        dbGaP-derived analysis datasets           (only with --from-raw)
  regional    cross-sectional CMR-brain MRI analyses     (S8-S18)
  long        longitudinal echo -> repeated MRI analyses (S1-S7)
  molecular   WGS pathway and AHBA analyses              (S19-S25; only with --molecular; hours)

Order notes (from the code audit):
  * 141 rewrites the Figure109 panel files, so 141-146 run BEFORE 109.
  * 154_build_missing_supplementary_tables writes Table S17 (= S3) and S21 (= S2); the dedicated S3 builder
    156 and its patch 157 run after it, so the final S3 is 156+157.
  * SLURM/shell genetics steps (74-78) are printed, not run; submit them separately if the WGS extracts
    must be rebuilt.

Freshness: the start time is recorded and passed to make_supplementary_tables.py (--fresh-since), which
marks any collected source file older than the start as STALE, so nothing from an earlier run is used
silently.

Usage
  python run_supplementary_pipeline.py --supplement Framingham_Supplementary_Materials_100726_clean.docx
  python run_supplementary_pipeline.py --supplement ... --molecular          # include S19-S25 (slow)
  python run_supplementary_pipeline.py --supplement ... --from-raw           # also rebuild datasets
  python run_supplementary_pipeline.py --list                                 # show the run order only
"""
import argparse, glob, re, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path("/data/qiallab/Framingham")
CODE = ROOT / "code"

STAGES = {
    "data": ["09_build_cognitive_outcomes_base.py", "11_build_cognitive_domains.py",
             "12_finalize_cognition_and_merge_metadata.py", "15_merge_APOE_into_metadata.py",
             "16_build_exam8_10_clinical_backbone.py", "18_build_nearest_exam_clinical_metadata.py",
             "23_merge_metabolic_variables.py", "37_merge_left_atrial_echo.py",
             "build_harmonized_echo_long.py", "97_build_cardiac_trajectories.py",
             "98_recover_echo_dates_and_define_windows.py", "100_build_longitudinal_APOE.py"],
    "regional": ["30_build_tier2_regional_families.py", "32_tier2_corrected_scaling.py",
                 "33_tier2_stability_APOE_slopes.py", "56_make_TableS1_complete_Tier2_results.py",
                 "38_APOE_AD_biomarker_completecase.py", "40_APOE_AD_completecase_proper_CI.py",
                 "43_plot_FINAL_biomarker_robustness.py", "44_direct_cardiac_APOE4_biomarker_prediction.py",
                 "41_LV_RV_3D_strain_analysis.py", "42_atrial_independence_LV.py",
                 "55_APOE4_metabolic_discovery_heart_wholebrain.py", "56_validate_LVEDVi_APOE4_HbA1c.py",
                 "57_kinship_cluster_robust_sensitivity.py", "58_make_Table4_LAVI_independence_FINAL.py",
                 "61_AF_flutter_sensitivity.py", "62_make_AF_flutter_supplementary.py",
                 "61_primary_pairs_metabolic_modifiers.py", "62_make_TableS7_metabolic_robustness.py",
                 "156_corrected_pedigree_cluster_robust_sensitivity.py",
                 "157_rebuild_corrected_TableS3_pedigree_cluster.py",
                 "158_corrected_HbA1c_pedigree_cluster_sensitivity.py",
                 "159_rebuild_TableS20_direct_biomarkers.py", "160_build_TableS25_biomarker_adjustment.py"],
    "long": ["104_temporal_mixedlm_random_intercept.py", "108_temporal_GEE_and_simple_slopes.py",
             "141_longitudinal_age_sex_time_sensitivity.py", "142_longitudinal_random_time_slope_sensitivity.py",
             "143_longitudinal_baseline_LA_sensitivity.py", "144_longitudinal_LA_annualized_change_sensitivity.py",
             "145_longitudinal_LA_exam4_exam6_change_sensitivity.py", "146_longitudinal_ICV_sensitivity.py",
             "146_longitudinal_ICV_sensitivity_STANDALONE.py", "146b_longitudinal_ICV_random_intercept.py",
             "146c_longitudinal_ICV_main_effect.py",
             "109_make_LA_ventricle_validation_figure.py",                 # after 141 (overwrite hazard)
             "111_longitudinal_LA_ventricle_vascular_adjustment.py",
             "161_longitudinal_LA_antihypertensive_sensitivity.py",   # Table S3 matched M3/M4 rows
             "113_APOE4_dose_longitudinal_sensitivity.py", "114_APOE4_categorical_dose_sensitivity.py",
             "120_longitudinal_cardiac_to_AD_biomarkers.py", "122_multicardiac_to_AD_biomarkers.py",
             "124_cardiac_to_later_cognition.py", "130_cardiac_trajectory_to_longitudinal_MMSE.py",
             "152_build_attrition_source.py",
             "153_attrition_IPSW_GEE.py", "153_longitudinal_LA_attrition_descriptive.py",
             "154_make_S17_attrition_IPSW_table.py",
             "155_longitudinal_random_time_slope_all16.py",
             "156_make_TableS17_longitudinal_robustness.py",               # ... then the dedicated S3 builder
             "157_patch_TableS17_add_vascular.py",
             "158_build_TableS23_S24_biomarker_cognition.py", "115_finalize_longitudinal_tables.py",
             "159_longitudinal_genotype_restricted_audit.py"],             # S1 genotype-restricted panel
    "molecular": ["63_build_FINAL_master_core4.py", "72_make_pathway_bed.py",
                  "74_unpack_TOPMed_c1.sbatch", "75_extract_TOPMed_pathway_autosomes.sbatch",
                  "76_merge_common_pathway_plink2.sh", "77_build_TOPMed_541_metadata.py",
                  "78_make_genomewide_PCA_panel.sbatch", "79_pathway_omnibus_heart_brain.py",
                  "80_make_wholebrain_coupling_maps.py", "81_split_coupling_maps_by_atlas.py",
                  "82_build_AHBA_DK_expression.py", "83_pathway_expression_vs_coupling.py",
                  "83b_build_DK56_distance_cache.py", "84b_spatial_null_FAST.py",
                  "84b_spatial_null_pathway_enrichment_fast.py",
                  "89_primary_regions_nonranked_enrichment_table_figure.py",
                  "90b_GO_Reactome_spatial_coupling_FAST.py", "91_make_mechanistic_triangulation_table.py",
                  "92_MSigDB_Hallmark_spatial.py", "93_select_C8_brain_signatures.py",
                  "93b_select_C8_brain_specific.py", "94_MSigDB_C8_brain_spatial.py"],
}


EXTERNAL_INPUTS = [  # files read but not written by the planned scripts (static code audit); linked read-only
    "/data/qiallab/Framingham/downloads/clinical/phs000007.v35.pht000183.v14.p16.Framingham_Pedigree.MULTI.txt.gz",
    "/data/qiallab/Framingham/data/cardiac_cmr/phs000007.v35.pht015153.v1.p16.c1.t_mrcvstr_2006_1b_1397s.HMB-IRB-MDS.txt.gz",
    "/data/qiallab/Framingham/data/longitudinal_20260924/longitudinal_analysis/E4_E6_cardiac_MRI_with_APOE.tsv",
    "/data/qiallab/Framingham/data/longitudinal_20260924/phs000007.v35.pht004374.v9.p16.c1.vr_npd_2023_a_1528s.HMB-IRB-MDS.txt.gz",
    "/data/qiallab/Framingham/downloads/clinical/phs000007.v35.pht004368.v6.p16.c1.vr_demrevd_2022_a_1470s.HMB-IRB-MDS.txt.gz",
    "/data/qiallab/Framingham/downloads/longitudinal_20260924/harmonized/echo_harmonized_long.tsv",
    "/data/qiallab/Framingham/downloads/longitudinal_20260924/longitudinal_analysis/APOE_full_offspring_subjects.tsv",
    "/data/qiallab/Framingham/downloads/longitudinal_20260924/longitudinal_analysis/cardiac_subject_trajectories.tsv",
    "/data/qiallab/Framingham/downloads/longitudinal_20260924/phs000007.v35.pht004364.v3.p16.c1.t_mrbrfs_2010_1_0900s.HMB-IRB-MDS.txt.gz",
    "/data/qiallab/Framingham/results/neurocardiac_metadata_analysis_ready_metabolic.csv",
    "/data/qiallab/Framingham/results/neurocardiac_metadata_with_atria.csv",
]
EXTERNAL_INPUTS_MOLECULAR = [
    "/data/qiallab/Framingham/downloads/clinical/phs000007.v35.pht000183.v14.p16.Framingham_Pedigree.MULTI.txt.gz",
    "/data/qiallab/Framingham/reference_data/msigdb/c8.all.v2026.1.Hs.symbols.gmt",
    "/data/qiallab/Framingham/results/genetics_inventory/neurocardiac_TOPMed_WGS_analysis_ready_EXACT.csv",
    "/data/qiallab/Framingham/results/genetics_pathways/gene_panel_v1.tsv",
    "/data/qiallab/Framingham/results/genetics_pathways/sample_metadata/TOPMed_541_genetics_analysis_ready.tsv",
    "/data/qiallab/Framingham/results/genetics_pathways/sample_metadata/TOPMed_c1_sample_info.txt",
    "/data/qiallab/Framingham/results/transcriptomic_enrichment/Framingham_DK56_partial_parcellation.nii.gz",
    "/data/qiallab/Framingham/results/transcriptomic_enrichment/wholebrain_M3_coupling_maps_DesikanKilliany.tsv",
]


def rewrite_code(src_dir, dst_dir, scripts, new_root):
    """Copy the planned scripts with every Framingham root path pointed at the sandbox root."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for s in scripts:
        txt = (src_dir / s).read_text(errors="ignore")
        for old in ("/shared" + str(ROOT), str(ROOT)):
            txt = txt.replace(old, str(new_root))
        (dst_dir / s).write_text(txt)
    # helper modules imported by the scripts (non-planned .py in the code dir) are copied the same way
    for p in src_dir.glob("*.py"):
        if p.name not in scripts and not (dst_dir / p.name).exists():
            txt = p.read_text(errors="ignore")
            for old in ("/shared" + str(ROOT), str(ROOT)):
                txt = txt.replace(old, str(new_root))
            (dst_dir / p.name).write_text(txt)


RAW = re.compile(r"(phs000\d+\.v\d+\.pht\d+.*\.txt\.gz$|Framingham_Pedigree.*\.txt\.gz$)", re.I)   # phenotype tables only (no genotype files)
PRODUCED_HERE = {"E4_E6_repeated_MRI_temporal_analysis.tsv", "attrition_atrisk_source.tsv"}
FROZEN_OK = {"TableS17_LongitudinalRobustness_BEFORE_VASCULAR_ROWS.xlsx", "ICV_main_146_longitudinal_ICV_matched_models.csv", "ICV_time_146_longitudinal_ICV_matched_models.csv"}
ALLOWED_RESULTS = {"neurocardiac_metadata_analysis_ready_metabolic.csv", "neurocardiac_metadata_with_atria.csv"}


def linkable(p):
    """Only raw dbGaP files, data-stage inputs and the two analysis-ready datasets may be linked.
    Intermediate results and old manuscript-package files must be regenerated, never borrowed."""
    p = Path(p); rel = str(p).replace(str(ROOT), "", 1)
    if p.name in PRODUCED_HERE:
        return False
    if rel.startswith("/results/"):
        return p.name in ALLOWED_RESULTS
    if p.name in FROZEN_OK:
        return True
    if "MANUSCRIPT" in rel.upper() or "REPO" in rel.upper() or "final_rebuild" in rel:
        return False
    return rel.startswith(("/data/", "/downloads/", "/reference_data/", "/templates/"))


def link_raw_tree(new_root, log):
    """Copy every raw phenotype table under data/ and downloads/ (read-only), keeping the LOGICAL path the scripts
    use - directory symlinks are followed (os.walk followlinks=True)."""
    import os
    n = 0
    for base in (ROOT / "data", ROOT / "downloads"):
        if not base.exists():
            continue
        for d, _, files in os.walk(base, followlinks=True):
            for f in files:
                if RAW.search(f):
                    n += link_input(Path(d) / f, new_root, log)
    return n


MAX_COPY = 2 * 1024**3          # never copy files larger than 2 GB into the workspace


def link_input(orig, new_root, log):
    """COPY one original input into the sandbox as a READ-ONLY file (no symlinks, ever).
    A script that tries to write to an input path then fails with 'Permission denied' instead of
    touching the original. Name kept for compatibility; nothing is linked."""
    orig = Path(orig)
    hits = [Path(h) for h in glob.glob(str(orig).replace("{*}", "*"))] if ("*" in str(orig) or "{" in str(orig)) else [orig]
    n = 0
    for h in hits:
        if not h.is_file() or not str(h).startswith(str(ROOT)) or not linkable(h):
            if h.is_file() and not linkable(h):
                log.write(f"REFUSED (must be regenerated, not copied): {h}\n")
            continue
        if h.stat().st_size > MAX_COPY:
            log.write(f"SKIPPED (larger than 2 GB): {h}\n"); continue
        dst = Path(str(h).replace(str(ROOT), str(new_root), 1))
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(h, dst)
        dst.chmod(0o444)                                   # read-only copy
        n += 1
        log.write(f"COPIED read-only: {h} -> {dst}\n")
    return n


MISSING = re.compile(r"(?:No such file or directory|FileNotFoundError|does not exist)[^/]*(/[^'\"\s:]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supplement", help="Supplementary Materials .docx for verification")
    ap.add_argument("--code-dir", default=str(CODE))
    ap.add_argument("--outdir", default=str(ROOT / "results/tables"))
    ap.add_argument("--sandbox", default=str(ROOT / "rebuild_workspace"),
                    help="separate analysis root; scripts are copied with paths rewritten so nothing in the old results/ is overwritten")
    ap.add_argument("--in-place", action="store_true", help="(not recommended) run the original scripts on the original results/")
    ap.add_argument("--from-raw", action="store_true", help="also rebuild the analysis datasets (data stage)")
    ap.add_argument("--molecular", action="store_true", help="include the slow S19-S25 analyses")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing script")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    stages = (["data"] if a.from_raw else []) + ["regional", "long"] + (["molecular"] if a.molecular else [])
    plan = [(st, s) for st in stages for s in STAGES[st]]
    if a.list:
        for i, (st, s) in enumerate(plan, 1):
            print(f"{i:3d}  {st:9s} {s}")
        return

    src_code = Path(a.code_dir); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    work = Path(a.sandbox) if not a.in_place else out          # logs and intermediate files stay out of the deliverables folder
    logs = work / "logs"; logs.mkdir(parents=True, exist_ok=True)
    missing = [s for _, s in plan if not (src_code / s).exists()]
    if missing:
        sys.exit("Scripts not found in code dir:\n  " + "\n  ".join(missing))
    if a.in_place:
        code, sandbox = src_code, None
    else:
        sandbox = Path(a.sandbox); code = sandbox / "code"
        rewrite_code(src_code, code, [s for _, s in plan], sandbox)
        linklog = open(work / "copied_inputs.log", "a")
        n = sum(link_input(p, sandbox, linklog) for p in EXTERNAL_INPUTS + (EXTERNAL_INPUTS_MOLECULAR if a.molecular else []))
        n += link_raw_tree(sandbox, linklog)
        links = [p for p in sandbox.rglob("*") if p.is_symlink()]
        if links:
            sys.exit(f"STOP: the workspace contains {len(links)} symlinks (e.g. {links[0]}). Remove the old workspace "
                     f"(rm -rf {sandbox}) and rerun - inputs must be copies, never links.")
        print(f"Sandbox {sandbox}: scripts rewritten to write here; {n} input files copied read-only (no links)")
    start = time.time()
    stamp = datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S")
    (work / "RUN_STARTED.txt").write_text(stamp + "\n")
    print(f"Clean run started {stamp}; {len(plan)} scripts; logs in {logs}")
    failed, skipped = [], []
    for i, (st, s) in enumerate(plan, 1):
        if s.endswith((".sbatch", ".sh")):
            print(f"[{i:3d}] {s}: SLURM/shell step - not run here (submit with sbatch if needed)")
            skipped.append(s); continue
        t0 = time.time()
        for attempt in range(6):
            logf = logs / f"{i:03d}_{Path(s).stem}.log"
            with open(logf, "w") as lf:
                r = subprocess.run([sys.executable, str(code / s)], cwd=code, stdout=lf, stderr=subprocess.STDOUT)
            if r.returncode == 0 or sandbox is None:
                break
            m = MISSING.findall(logf.read_text(errors="ignore"))
            cand = [Path(p) for p in m if p.startswith(str(sandbox))]
            orig = [Path(str(c).replace(str(sandbox), str(ROOT), 1)) for c in cand]
            new_links = sum(link_input(o, sandbox, linklog) for o in orig if o.is_file())
            for tok in re.findall(r"Could not find (\S+)", logf.read_text(errors="ignore")):
                for base in (ROOT / "downloads", ROOT / "data"):
                    for h in base.rglob(f"*{tok}*"):
                        new_links += link_input(h, sandbox, linklog)
            if not new_links:
                break
            print(f"      {s}: copied {new_links} missing input(s) read-only from the original tree, retrying (see copied_inputs.log)")
        dt = time.time() - t0
        print(f"[{i:3d}] {st:9s} {s:60s} {'ok' if r.returncode == 0 else 'FAILED'}  ({dt:.0f}s)")
        if r.returncode != 0:
            failed.append(s)
            if not a.keep_going:
                sys.exit(f"Stopped: {s} failed - see {logs}/{i:03d}_{Path(s).stem}.log (or rerun with --keep-going)")

    if sandbox is not None:
        touched = []
        for line in open(work / "copied_inputs.log"):
            if line.startswith("COPIED") and " -> " in line:
                orig = Path(line.split(" -> ")[0].replace("COPIED read-only:", "").strip())
                if orig.exists() and orig.stat().st_mtime >= start:
                    touched.append(str(orig))
        if touched:
            print("\nWARNING - original files modified during the run (written through a link):")
            for t in touched:
                print("   ", t)
        else:
            print("\nWrite-through check: no original file was modified.")

    collector = Path(__file__).resolve().parent / "make_supplementary_tables.py"
    cmd = [sys.executable, str(collector), "--outdir", str(out), "--fresh-since", str(start)]
    if sandbox is not None:
        cmd += ["--root", str(sandbox)]
    if a.supplement:
        cmd += ["--supplement", a.supplement]
    if not a.molecular:      # molecular tables use the verified existing outputs
        cmd += ["--existing-ok", "S19", "S20", "S21", "S22", "S23", "S24", "S25"]
    print("\nCollecting and verifying:", " ".join(cmd[1:]))
    subprocess.run(cmd)
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Failed: {failed or 'none'}. Not run (SLURM): {skipped or 'none'}")


if __name__ == "__main__":
    main()
