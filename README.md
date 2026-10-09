# Antecedent left atrial remodeling, APOE ε4, and later brain change (Framingham Offspring Study)

Analysis code for the manuscript. **No data are included.** Framingham Heart Study data are available to
approved investigators through dbGaP (study phs000007).

| Folder | Contents |
|---|---|
| `analysis/` | analysis scripts, in the order listed in `analysis/RUN_ORDER.md` |
| `figures/` | one script per figure; each checks every plotted value against the manuscript (`--strict`) |
| `tables/` | Table 1, Tables 2-4, Supplementary Tables S1-S11 and Data S1; `run_supplementary_pipeline.py` reruns all analyses in an isolated workspace (inputs copied read-only) and verifies every table against the supplement |
| `package/` | assembles the submission package with checksums |

## Running
1. `conda env create -f environment.yml && conda activate framingham`
2. The scripts use the root `/data/qiallab/Framingham` with sub-folders `data/`, `downloads/`, `results/`.
   Replace it with your own root, keeping the same layout.
3. `python tables/run_supplementary_pipeline.py --list` shows the run order; without `--list` it runs everything.
4. Figure and table scripts: run with `--strict` to stop on any difference from the published values.

See `PROVENANCE.md` for how the analysis-ready datasets were produced and verified.
