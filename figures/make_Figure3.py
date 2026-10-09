#!/usr/bin/env python3
"""
Figure 3 - Regional cardiac phenotype x APOE e4 associations with cortical morphology (exploratory).

Replaces 59 / 140 / 141 / 156 / 65. Differences:
  * Panel A slopes are COMPUTED by refitting the four M3 models exactly as in
    32_tier2_corrected_scaling.py; the carrier slope is beta_heart + beta_interaction with its
    variance from the HC3 covariance (V11 + V22 + 2 V12). Script 59 hard-coded carrier CIs that did
    not match Table S9 (right lingual area was drawn as excluding zero; Table S9: -0.007 to 0.247).
  * Panel B (regional main effects with within-family q < 0.05) is read from the full-cohort
    tier2_corrected_all.csv (N = 765/781), not from a pre-made plot file.
  * Panel C renders fsaverage inflated surfaces directly from the FreeSurfer annotations
    (aparc for DK, aparc.a2009s for Destrieux); no LAVI "panel D".
  * No in-figure title ("primary" removed). Every plotted value is checked against Table 2 /
    Table S9 (Figure3_verification.csv; --strict stops on mismatch).

Outputs: Figure3.{pdf,svg,png}, Figure3A_plot_data.csv, Figure3B_plot_data.csv,
         Figure3_verification.csv, surface renders in OUTDIR/surfaces/ (summary output only).
Usage: python make_Figure3.py [--strict] [--outdir DIR] [--no-lavi]
"""
import argparse, os, re, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path("/data/qiallab/Framingham")
DATAFILE = ROOT / "results/neurocardiac_metadata_analysis_ready_metabolic.csv"
TIER2 = ROOT / "results/tier2_corrected/tier2_corrected_all.csv"        # full cohort (N=765), NOT <=2-year
SUBJECTS_DIR = Path(os.environ.get("SUBJECTS_DIR", "/usr/local/freesurfer/subjects"))
MNE_DIR = ROOT / "templates/mne_subjects"      # where MNE puts fsaverage if no FreeSurfer is installed

# The four regional interactions (Table 2), top to bottom in panel A
FINDINGS = [
    ("LVESVi", "rh_superiorfrontal_area", "surface_area", "Right superior frontal surface area — LVESVi"),
    ("LV_MASSi", "rh_lingual_vol", "volume", "Right lingual volume — LV mass index"),
    ("LV_MASSi", "rh_lingual_area", "surface_area", "Right lingual surface area — LV mass index"),
    ("LVEF", "lh_middletemporal_vol", "volume", "Left middle temporal volume — LVEF"),
]
# Table 2 (interaction) and Table S9 (stratified slopes): beta, ci_low, ci_high
VERIFY = {
    "rh_superiorfrontal_area": {"int": (-0.170, -0.260, -0.081), "nc": (0.035, -0.012, 0.081), "car": (-0.136, -0.218, -0.053)},
    "rh_lingual_vol": {"int": (0.305, 0.144, 0.467), "nc": (-0.092, -0.184, 0.001), "car": (0.213, 0.060, 0.367)},
    "rh_lingual_area": {"int": (0.226, 0.095, 0.357), "nc": (-0.106, -0.184, -0.028), "car": (0.120, -0.007, 0.247)},
    "lh_middletemporal_vol": {"int": (0.233, 0.087, 0.380), "nc": (-0.046, -0.121, 0.030), "car": (0.188, 0.060, 0.316)},
}
TOL = 0.002

# Panel C: (hemi, atlas, FreeSurfer label, legend text, colour) - Okabe-Ito
ROIS = [
    ("lh", "aparc", "middletemporal", "LVEF → left middle temporal volume", "#E69F00"),
    ("rh", "aparc", "superiorfrontal", "LVESVi → right superior frontal\nsurface area", "#56B4E9"),
    ("rh", "aparc", "lingual", "LV mass index → right lingual volume\nand surface area", "#0072B2"),
    ("lh", "aparc.a2009s", "G_occipital_middle", "LAVI → left middle-occipital thickness\nvariability (Figure 5)", "#CC79A7"),
]
# Right superior frontal is mostly dorsal/medial: a lateral view shows only a thin rim, so use dorsal + medial
VIEWS = [("lh", "lateral", "Left lateral"), ("rh", "dorsal", "Right dorsal"), ("rh", "medial", "Right medial")]

NC, CA, INK = "#3B6FB6", "#D9822B", "#1F2933"
from matplotlib import font_manager
_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in _AVAIL), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "font.size": 7, "axes.linewidth": 0.7,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


# ------------------------------------------------------------------ data preparation (= script 32)
def prepare(df):
    h = pd.to_numeric(df["height_in"], errors="coerce") * 2.54
    w = pd.to_numeric(df["weight_lb"], errors="coerce") * 0.45359237
    df["BSA_m2"] = np.sqrt(h * w / 3600.0)
    for raw, idx in (("LVEDV", "LVEDVi"), ("LVESV", "LVESVi"), ("LV_MASS", "LV_MASSi")):
        df[idx] = pd.to_numeric(df[raw], errors="coerce") / df["BSA_m2"]
    df["TotalCorticalSurfaceArea"] = (pd.to_numeric(df["lh_WhiteSurfArea_DesKil_area"], errors="coerce")
                                      + pd.to_numeric(df["rh_WhiteSurfArea_DesKil_area"], errors="coerce"))
    df["_sex"] = pd.to_numeric(df["sex_clinical"], errors="coerce")
    dm = pd.to_numeric(df["diabetes_history_nearest_exam"], errors="coerce")
    df["diabetes_history_any"] = np.where(dm.isna(), np.nan, dm.isin([1, 2]).astype(float))
    return df


def z(x):
    x = pd.to_numeric(x, errors="coerce"); sd = x.std()
    return (x - x.mean()) / sd


def fit_m3(df, heart, region, metric):
    y = pd.to_numeric(df[region], errors="coerce")
    if metric == "volume":
        y = y / pd.to_numeric(df["IntraCranialVol"], errors="coerce")
    dat = pd.DataFrame({"y": y, "heart": df[heart], "age": df["age_at_mri"], "sex": df["_sex"],
                        "interval": df["abs_delta_years"], "BMI": df["BMI_nearest_exam"],
                        "SBP": df["SBP_nearest_exam"], "smoking": df["current_smoker_nearest_exam"],
                        "diabetes": df["diabetes_history_any"], "APOE4": df["APOE4_carrier"],
                        "total_area": df["TotalCorticalSurfaceArea"]}).apply(pd.to_numeric, errors="coerce")
    need = ["y", "heart", "age", "sex", "interval", "BMI", "SBP", "smoking", "diabetes", "APOE4"]
    if metric == "surface_area":
        need.append("total_area")
    dat = dat[need].dropna()
    for c in ("y", "heart", "age", "interval", "BMI", "SBP"):
        dat[c + "_z"] = z(dat[c])
    xcols = ["heart_z", "age_z", "sex", "interval_z"]
    if metric == "surface_area":
        dat["total_area_z"] = z(dat["total_area"]); xcols.append("total_area_z")
    dat["heart_x_APOE4"] = dat["heart_z"] * dat["APOE4"]
    xcols += ["BMI_z", "SBP_z", "smoking", "diabetes", "APOE4", "heart_x_APOE4"]
    fit = sm.OLS(dat["y_z"], sm.add_constant(dat[xcols])).fit(cov_type="HC3")
    b, V = fit.params, fit.cov_params()
    bn, sn = b["heart_z"], np.sqrt(V.loc["heart_z", "heart_z"])
    bc = b["heart_z"] + b["heart_x_APOE4"]
    sc = np.sqrt(V.loc["heart_z", "heart_z"] + V.loc["heart_x_APOE4", "heart_x_APOE4"]
                 + 2 * V.loc["heart_z", "heart_x_APOE4"])
    bi, si = b["heart_x_APOE4"], np.sqrt(V.loc["heart_x_APOE4", "heart_x_APOE4"])
    return {"N": len(dat), "nc": (bn, bn - 1.96 * sn, bn + 1.96 * sn), "car": (bc, bc - 1.96 * sc, bc + 1.96 * sc),
            "int": (bi, bi - 1.96 * si, bi + 1.96 * si), "p_int": fit.pvalues["heart_x_APOE4"]}


# ------------------------------------------------------------------ labels for panel B
DK = {"transversetemporal": "transverse temporal", "superiorfrontal": "superior frontal", "middletemporal": "middle temporal",
      "temporalpole": "temporal pole", "precentral": "precentral", "postcentral": "postcentral", "lingual": "lingual",
      "superiortemporal": "superior temporal", "inferiortemporal": "inferior temporal", "precuneus": "precuneus",
      "superiorparietal": "superior parietal", "inferiorparietal": "inferior parietal", "rostralmiddlefrontal": "rostral middle frontal",
      "caudalmiddlefrontal": "caudal middle frontal", "lateraloccipital": "lateral occipital", "fusiform": "fusiform",
      "parahippocampal": "parahippocampal", "entorhinal": "entorhinal", "insula": "insula", "cuneus": "cuneus",
      "pericalcarine": "pericalcarine", "medialorbitofrontal": "medial orbitofrontal", "lateralorbitofrontal": "lateral orbitofrontal",
      "frontalpole": "frontal pole", "posteriorcingulate": "posterior cingulate", "isthmuscingulate": "isthmus cingulate",
      "caudalanteriorcingulate": "caudal anterior cingulate", "rostralanteriorcingulate": "rostral anterior cingulate"}
DESTRIEUX = {"G_precentral": "precentral gyrus", "Pole_temporal": "temporal pole",
             "G_oc-temp_med-Lingual": "lingual gyrus (medial occipitotemporal)",
             "S_oc-temp_med_and_Lingual": "medial occipitotemporal/lingual sulcus",
             "S_cingul-Marginalis": "marginal cingulate sulcus", "G_occipital_sup": "superior occipital gyrus",
             # label spellings as they appear in the tier2 variable names (hyphens -> underscores)
             "S_oc_temp_med_Lingual": "medial occipitotemporal/lingual sulcus",
             "S_cingul_Marginalis": "marginal cingulate sulcus",
             "G_occipital_middle": "middle occipital gyrus"}


def destrieux_generic(lab):
    """Readable fallback for Destrieux labels not in DESTRIEUX: G_ -> gyrus, S_ -> sulcus."""
    kind = {"G": "gyrus", "S": "sulcus", "G_and_S": "gyrus and sulcus"}
    m = re.match(r"(G_and_S|G|S)_(.+)", lab)
    if not m:
        return lab.replace("_", " ")
    return f"{m.group(2).replace('_', ' ').replace('-', ' ')} {kind[m.group(1)]}"
METRIC = {"vol": "volume", "area": "surface area", "tk": "thickness"}
CARDIAC = {"LV_MASSi": "LV mass index", "LVEF": "LVEF", "LVESVi": "LVESVi", "LVEDVi": "LVEDVi", "LV_3D_LONG": "LV longitudinal strain"}


def pretty(region, cardiac):
    m = re.match(r"(lh|rh)_(.+)_(vol|area|tk)$", region)
    if not m:
        return f"{region} — {cardiac}"
    hemi, lab, met = m.groups()
    name = DK.get(lab) or DESTRIEUX.get(lab) or destrieux_generic(lab)
    return f"{'Left' if hemi == 'lh' else 'Right'} {name} {METRIC[met]} — {CARDIAC.get(cardiac, cardiac)}"


# ------------------------------------------------------------------ panel C surface renders

def find_fsaverage():
    """fsaverage with surf/{lh,rh}.inflated and label/*.aparc(.a2009s).annot.
    Order: $SUBJECTS_DIR/fsaverage, $FREESURFER_HOME/subjects/fsaverage, a previous MNE download,
    then download with MNE-Python (pip install mne; needs internet once, e.g. on a login node)."""
    def ok(p):
        return p is not None and all((p / f).exists() for f in (
            "surf/lh.inflated", "surf/rh.inflated", "label/lh.aparc.annot", "label/rh.aparc.annot",
            "label/lh.aparc.a2009s.annot", "label/rh.aparc.a2009s.annot"))
    cands = [SUBJECTS_DIR / "fsaverage", MNE_DIR / "fsaverage"]
    if os.environ.get("FREESURFER_HOME"):
        cands.append(Path(os.environ["FREESURFER_HOME"]) / "subjects/fsaverage")
    for p in cands:
        if ok(p):
            print("Panel C: using", p); return p
    try:
        import mne
        MNE_DIR.mkdir(parents=True, exist_ok=True)
        p = Path(mne.datasets.fetch_fsaverage(subjects_dir=Path(MNE_DIR), verbose=False))  # MNE>=1.13 needs a Path
        if ok(p):
            print("Panel C: downloaded fsaverage with MNE to", p); return p
        print("Panel C: MNE fsaverage is missing files:", p)
    except Exception as e:
        print("Panel C: MNE download failed:", e)
    return None

def render_surfaces(outdir, include_lavi):
    try:
        import nibabel as nib
        from nilearn import plotting
    except ImportError:
        print("Panel C: nilearn/nibabel not available - panel left blank"); return None
    fs = find_fsaverage()
    if fs is None:
        print("Panel C: no fsaverage found or downloadable - panel left blank"); return None
    rois = [r for r in ROIS if include_lavi or r[1] != "aparc.a2009s"]
    imgs = []
    for hemi, view, title in VIEWS:
        coords, faces = nib.freesurfer.read_geometry(str(fs / f"surf/{hemi}.inflated"))
        bgf = fs / f"surf/{hemi}.sulc"
        if not bgf.exists():
            bgf = fs / f"surf/{hemi}.curv"          # MNE's fsaverage may ship curv but not sulc
        sulc = nib.freesurfer.read_morph_data(str(bgf))
        roi = np.zeros(len(coords))
        colors = ["#FFFFFF"]
        for k, (h, atlas, label, _, col) in enumerate(rois, start=1):
            colors.append(col)
            if h != hemi:
                continue
            lab, _, names = nib.freesurfer.read_annot(str(fs / f"label/{hemi}.{atlas}.annot"))
            names = [n.decode() if isinstance(n, bytes) else n for n in names]
            idx = names.index(label)
            nv = int((lab == idx).sum())
            print(f"Panel C: {hemi} {atlas:13s} {label:20s} vertices={nv}  (view {view})")
            if nv == 0:
                sys.exit(f"Panel C: label {label} has no vertices in {hemi}.{atlas}.annot")
            roi[lab == idx] = k
        f = Path(outdir) / f"surf_{hemi}_{view}.png"
        sf = plotting.plot_surf_roi((coords, faces), roi_map=np.where(roi > 0, roi, np.nan),
                                    hemi="left" if hemi == "lh" else "right", view=view, bg_map=sulc,
                                    bg_on_data=True, cmap=ListedColormap(colors), vmin=0, vmax=len(colors) - 1,
                                    colorbar=False)
        sf.savefig(f, dpi=400, bbox_inches="tight", transparent=True)
        plt.close(sf)
        imgs.append((f, title))
    return imgs, rois


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-lavi", action="store_true", help="omit the LAVI region from panel C")
    ap.add_argument("--outdir", default=str(ROOT / "results/paper_figures_tables/Figure3"))
    a = ap.parse_args()
    out = Path(a.outdir); (out / "surfaces").mkdir(parents=True, exist_ok=True)

    df = prepare(pd.read_csv(DATAFILE, low_memory=False))
    rowsA, checks = [], []
    for heart, region, metric, label in FINDINGS:
        r = fit_m3(df, heart, region, metric)
        rowsA.append({"label": label, "region": region, "cardiac": heart, "N": r["N"], "p_interaction": r["p_int"],
                      **{f"{k}_{s}": v for k in ("nc", "car", "int") for s, v in zip(("beta", "ci_low", "ci_high"), r[k])}})
        for k in ("int", "nc", "car"):
            for s, got, exp in zip(("beta", "ci_low", "ci_high"), r[k], VERIFY[region][k]):
                checks.append({"region": region, "quantity": f"{k}_{s}", "published": exp,
                               "computed": round(got, 4), "match": abs(got - exp) <= TOL})
    A = pd.DataFrame(rowsA)

    t2 = pd.read_csv(TIER2)
    B = t2[(t2["model"] == "M2_vascular") & (t2["q_cardiac"] < 0.05)].copy()
    B["label"] = [pretty(r, c) for r, c in zip(B["region"], B["cardiac"])]
    B = B.sort_values("beta_cardiac", ascending=False)

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.95], height_ratios=[1, 1.15], wspace=0.22, hspace=0.32)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[1, 0])

    # Panel A
    y = np.arange(len(A))[::-1]
    for yi, (_, r) in zip(y, A.iterrows()):
        for k, c, mk, off in (("nc", NC, "o", 0.13), ("car", CA, "s", -0.13)):
            b, lo, hi = r[f"{k}_beta"], r[f"{k}_ci_low"], r[f"{k}_ci_high"]
            axA.errorbar(b, yi + off, xerr=[[b - lo], [hi - b]], fmt=mk, ms=3.6, color=c, capsize=2, lw=1.0)
    axA.axvline(0, color="#7B8794", lw=0.7, ls="--")
    axA.set_yticks(y); axA.set_yticklabels(A["label"]); axA.tick_params(axis="y", length=0)
    axA.set_xlabel("Standardized cardiac–brain slope (β, 95% CI)")
    axA.legend(handles=[plt.Line2D([], [], color=NC, marker="o", ls="", label="APOE ε4 noncarriers"),
                        plt.Line2D([], [], color=CA, marker="s", ls="", label="APOE ε4 carriers")],
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    axA.text(-0.02, 1.04, "A", transform=axA.transAxes, fontsize=10, fontweight="bold", ha="right")

    # Panel B
    yb = np.arange(len(B))[::-1]
    for yi, (_, r) in zip(yb, B.iterrows()):
        axB.errorbar(r["beta_cardiac"], yi, xerr=[[r["beta_cardiac"] - r["CI_low"]], [r["CI_high"] - r["beta_cardiac"]]],
                     fmt="o", ms=3.2, color=INK, capsize=2, lw=0.9)
        axB.text(1.01, yi, f"q={r['q_cardiac']:.3f}", transform=axB.get_yaxis_transform(), fontsize=5.6, va="center")
    axB.axvline(0, color="#7B8794", lw=0.7, ls="--")
    axB.set_yticks(yb); axB.set_yticklabels(B["label"], fontsize=6); axB.tick_params(axis="y", length=0)
    axB.set_xlabel("Standardized cardiac main effect (β, 95% CI)")
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    axB.text(-0.02, 1.04, "B", transform=axB.transAxes, fontsize=10, fontweight="bold", ha="right")

    # Panel C
    rendered = render_surfaces(out / "surfaces", include_lavi=not a.no_lavi)
    gsC = gs[:, 1].subgridspec(4, 1, height_ratios=[1, 1, 1, 0.55], hspace=0.05)
    if rendered:
        imgs, rois = rendered
        for i, (f, title) in enumerate(imgs):
            ax = fig.add_subplot(gsC[i]); ax.imshow(plt.imread(f)); ax.axis("off")
            ax.set_title(title, fontsize=6.5, pad=1)
            if i == 0:
                ax.text(-0.02, 1.0, "C", transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right")
        axL = fig.add_subplot(gsC[3]); axL.axis("off")
        axL.legend(handles=[Patch(color=r[4], label=r[3].replace(" → ", " →\n")) for r in rois],
                   loc="upper center", frameon=False, fontsize=5.6, ncol=1, handlelength=1.2)
    else:
        ax = fig.add_subplot(gs[:, 1]); ax.axis("off")
        ax.text(0.5, 0.5, "Panel C requires fsaverage\n(set SUBJECTS_DIR)", ha="center", va="center")

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(out / f"Figure3.{ext}", bbox_inches="tight", **kw)
    A.to_csv(out / "Figure3A_plot_data.csv", index=False)
    B[["region", "cardiac", "label", "N", "beta_cardiac", "CI_low", "CI_high", "p_cardiac", "q_cardiac"]].to_csv(
        out / "Figure3B_plot_data.csv", index=False)
    chk = pd.DataFrame(checks); chk.to_csv(out / "Figure3_verification.csv", index=False)
    print(A.round(4).to_string(index=False)); print(f"\npanel B rows: {len(B)}")
    print("\nVERIFICATION (Table 2 / Table S9)\n" + chk.to_string(index=False))
    if a.strict and not chk["match"].all():
        sys.exit("STRICT: computed values differ from Table 2 / Table S9 - see Figure3_verification.csv")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
