#!/usr/bin/env python3
"""
Supplementary Data S1: complete regional results (M3, M2, M1; 1,960 tests each).

Source : TableS1_complete_Tier2_results.xlsx written by 56_make_TableS1_complete_Tier2_results.py
         (from tier2_corrected_all.csv, script 32) in the clean rebuild workspace.
Changes: 1) "Accumbens surface area" -> "Accumbens volume" (FreeSurfer names this volume *_Accumbens_area);
         2) README article title updated to the current title;
         3) README / dictionary: "Table S8" -> "Table S6" (supplement renumbered 8 Oct 2026).
Output : /data/qiallab/Framingham/results/tables/Supplementary_Data_S1.xlsx
"""
import re, sys
from pathlib import Path
import pandas as pd

SRC = Path("/data/qiallab/Framingham/rebuild_workspace/results/paper_figures_tables/supplementary/TableS1_complete_Tier2_results.xlsx")
OUT = Path("/data/qiallab/Framingham/results/tables/Supplementary_Data_S1.xlsx")
TITLE = "Antecedent Left Atrial Remodeling, APOE ε4, and Later Brain Change in the Framingham Offspring Study"
ID = re.compile(r"^(shareid|share_id|dbgap_subject_id|subject_id|subjid|framid|pid)$", re.I)

sheets = pd.read_excel(SRC, sheet_name=None)
fixed = {}
n_acc = 0
sheets.pop("FDR_significant", None)          # same 28 rows as Table S6; not part of Data S1
for name, df in sheets.items():
    if any(ID.match(str(c)) for c in df.columns):
        sys.exit(f"STOP: participant identifier column in sheet {name}")
    df = df.copy()
    if "Brain phenotype" in df.columns:
        is_acc = df["Brain phenotype"].astype(str).str.contains("Accumbens surface area")
        n_acc += int(is_acc.sum())
        df.loc[is_acc, "Brain phenotype"] = df.loc[is_acc, "Brain phenotype"].str.replace("Accumbens surface area", "Accumbens volume")
        if "metric_type" in df.columns:
            df.loc[df["Brain phenotype"].astype(str).str.contains("Accumbens"), "metric_type"] = "volume"
    for c in df.columns:                                   # text fixes in README / dictionary cells
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(r"Article: .*", f"Article: {TITLE}", regex=True) \
                                     .str.replace("Table S8", "Table S6", regex=False).replace("nan", "")
    df.columns = [str(c).replace("Table S8", "Table S6") for c in df.columns]
    fixed[name] = df

# checks: three model sheets with 1,960 rows each, no surface-area accumbens left
for name, df in fixed.items():
    if name.startswith("M") and "Brain phenotype" in df.columns:
        assert len(df) == 1960, f"{name}: {len(df)} rows, expected 1,960"
        assert not df["Brain phenotype"].astype(str).str.contains("Accumbens surface").any()
with pd.ExcelWriter(OUT) as w:
    for name, df in fixed.items():
        df.to_excel(w, sheet_name=name, index=False)
print(f"Data S1 written: {OUT}")
print(f"sheets: {list(fixed)}; Accumbens rows relabelled: {n_acc}")
