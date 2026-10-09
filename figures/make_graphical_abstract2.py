#!/usr/bin/env python3
"""
Graphical abstract - Left atrial remodeling, APOE e4, and later brain change (Framingham Offspring Study).

Design column : drawn heart (left atrium highlighted) and FreeSurfer fsaverage brain surfaces.
Panel A       : schematic of the direction of the longitudinal LA x APOE e4 interaction (no axis values).
Panel B       : published LAVI slopes by APOE e4 status (95% CI) and the left middle occipital gyrus
                (Destrieux G_occipital_middle) on the fsaverage inflated surface.
Surfaces are rendered on first run from the MNE fsaverage download (FSAVERAGE below) and cached in the
output folder; without fsaverage the script falls back to drawn brains.
Outputs PDF, SVG and 600-dpi PNG.     Usage: python make_graphical_abstract2.py [outdir]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, PathPatch

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
FSAVERAGE = Path("/data/qiallab/Framingham/templates/mne_subjects/fsaverage")

_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in _AVAIL), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
F = 0.88 if FONT == "DejaVu Sans" else 1.0          # DejaVu is ~12% wider than Arial/Liberation

INK, MUTE = "#1F2933", "#5B6B7F"
NC, CA = "#3B6FB6", "#D9822B"            # noncarriers / carriers

fig = plt.figure(figsize=(8.0, 4.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 84); ax.axis("off")


def box(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=0.9))


def text(x, y, s, size=7, weight="normal", color=INK, ha="left", style="normal"):
    ax.text(x, y, s, fontsize=size * F, fontweight=weight, color=color, ha=ha, va="top", fontstyle=style, linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.1, color=MUTE))


# ---------------------------------------------------------------- simple drawn icons
def curve(pts, cx, cy, s, **kw):
    v = np.array(pts, float) * s + [cx, cy]
    codes = [MPath.MOVETO] + [MPath.CURVE4] * (len(pts) - 1)
    ax.add_patch(PathPatch(MPath(v, codes), fc="none", **kw))


def shape(pts, cx, cy, s, **kw):
    v = np.array(pts + [pts[0]], float) * s + [cx, cy]
    codes = [MPath.MOVETO] + [MPath.CURVE4] * (len(pts) - 1) + [MPath.CLOSEPOLY]
    return ax.add_patch(PathPatch(MPath(v, codes), **kw))


def brain(cx, cy, s):
    """Drawn lateral left hemisphere - used only if fsaverage is unavailable."""
    edge = "#8C5F68"
    ax.add_patch(Ellipse((cx + 0.78 * s, cy - 0.52 * s), 0.78 * s, 0.46 * s, angle=-12, fc="#E9C9CE", ec=edge, lw=0.8))
    shape([(-1.25, 0.05), (-1.30, 0.55), (-0.85, 0.95), (-0.25, 0.98), (0.35, 1.02), (1.05, 0.80), (1.22, 0.30),
           (1.33, -0.02), (1.20, -0.30), (0.95, -0.30), (0.70, -0.30), (0.45, -0.38), (0.20, -0.45),
           (-0.15, -0.55), (-0.55, -0.48), (-0.70, -0.25), (-0.82, -0.12), (-1.15, -0.30), (-1.25, 0.05)],
          cx, cy, s, fc="#F1D9DC", ec=edge, lw=1.1)


def heart(cx, cy, s):
    """Anterior view: aorta (red), pulmonary trunk (blue), right atrium, ventricles, left atrium in orange."""
    curve([(-0.15, 0.45), (-0.20, 1.05), (0.35, 1.35), (0.60, 0.95)], cx, cy, s, ec="#D0574D", lw=4.5)  # aortic arch
    curve([(0.25, 0.50), (0.30, 0.85), (0.55, 0.95), (0.80, 0.80)], cx, cy, s, ec="#5577AE", lw=4.0)    # pulmonary trunk
    ax.add_patch(Ellipse((cx - 0.45 * s, cy + 0.35 * s), 0.55 * s, 0.45 * s, fc="#A63B35", ec="#7A2620", lw=0.9))  # right atrium
    shape([(-0.55, 0.45), (-0.20, 0.62), (0.45, 0.65), (0.78, 0.30), (1.00, 0.05), (0.80, -0.55), (0.30, -1.05),
           (0.15, -1.15), (0.00, -1.10), (-0.20, -0.85), (-0.55, -0.40), (-0.75, 0.10), (-0.55, 0.45)],
          cx, cy, s, fc="#B8443D", ec="#7A2620", lw=1.1)
    curve([(0.20, 0.50), (0.25, 0.05), (0.32, -0.45), (0.22, -0.95)], cx, cy, s, ec="#F2C14E", lw=1.0)  # coronary artery
    # left atrium + appendage highlighted in the study colour (patient's left = viewer's right, upper border)
    shape([(0.40, 0.60), (0.72, 0.92), (1.12, 0.78), (1.10, 0.45), (1.06, 0.18), (0.70, 0.30), (0.40, 0.60)],
          cx, cy, s, fc=CA, ec="#9C5A16", lw=0.9, zorder=5)   # 7 points = one MOVETO + two Bezier segments


# ---------------------------------------------------------------- FreeSurfer fsaverage surfaces
def _mesh():
    import nibabel as nib
    coords, faces = nib.freesurfer.read_geometry(str(FSAVERAGE / "surf/lh.inflated"))
    bg = FSAVERAGE / "surf/lh.sulc"
    bg = bg if bg.exists() else FSAVERAGE / "surf/lh.curv"
    return (coords, faces), nib.freesurfer.read_morph_data(str(bg))


def render_mog_surface(dst):
    """Left middle occipital gyrus (Destrieux G_occipital_middle) highlighted, lateral view."""
    try:
        import nibabel as nib
        from nilearn import plotting
        from matplotlib.colors import ListedColormap
    except ImportError:
        return False
    if not (FSAVERAGE / "label/lh.aparc.a2009s.annot").exists():
        return False
    mesh, bg = _mesh()
    labels, _, names = nib.freesurfer.read_annot(str(FSAVERAGE / "label/lh.aparc.a2009s.annot"))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    roi = np.where(labels == names.index("G_occipital_middle"), 1.0, np.nan)
    f = plotting.plot_surf_roi(mesh, roi_map=roi, hemi="left", view="lateral", bg_map=bg, bg_on_data=True,
                               cmap=ListedColormap(["#FFFFFF", CA]), vmin=0, vmax=1, colorbar=False)
    f.savefig(dst, dpi=600, bbox_inches="tight", transparent=True); plt.close(f)
    return True


def render_plain_surface(dst):
    """Plain left hemisphere (sulcal shading only), lateral view."""
    try:
        from nilearn import plotting
    except ImportError:
        return False
    if not (FSAVERAGE / "surf/lh.inflated").exists():
        return False
    mesh, bg = _mesh()
    f = plotting.plot_surf(mesh, surf_map=None, bg_map=bg, hemi="left", view="lateral", colorbar=False)
    f.savefig(dst, dpi=600, bbox_inches="tight", transparent=True); plt.close(f)
    return True


PLAIN = OUT / "fsaverage_left_lateral_plain.png"
SURF = OUT / "fsaverage_left_lateral_G_occipital_middle.png"
if not PLAIN.exists():
    print("rendered plain fsaverage surface" if render_plain_surface(PLAIN) else "fsaverage not available - drawn brains used")
if not SURF.exists():
    print("rendered fsaverage surface (middle occipital)" if render_mog_surface(SURF) else "fsaverage not available - drawn brain used")


def brain_icon(cx, cy, w=0.085, h=0.15):
    """FreeSurfer surface centred at (cx, cy) in layout units; falls back to the drawn brain."""
    if PLAIN.exists():
        a = fig.add_axes([cx / 160 - w / 2, cy / 84 - h / 2, w, h]); a.imshow(plt.imread(PLAIN)); a.axis("off")
    else:
        brain(cx, cy, 3.3)


# ---------------------------------------------------------------- title
text(80, 81.5, "Left atrial remodeling, APOE ε4, and later brain change", 11, "bold", ha="center")
text(80, 76.3, "Framingham Heart Study Offspring Cohort", 8, color=MUTE, ha="center", style="italic")

# ---------------------------------------------------------------- column 1: design
box(3, 5, 40, 66, "#F4F6F8", "#9AA5B1")
text(23, 69, "Design", 9, "bold", ha="center")
text(6, 63.5, "Longitudinal arm (N=1,305)", 7.5, "bold")
heart(10.5, 54.0, 3.5)
ax.plot([16, 26], [55, 55], color=MUTE, lw=0.8)
for xx in (17, 21, 25):
    ax.plot(xx, 55, "o", ms=3.2, color=MUTE)
text(21, 52.8, "Echo, Exams 4–6", 6.2, ha="center")
arrow(27, 55, 31, 55)
brain_icon(37.6, 55.5)
text(37.6, 50.2, "Repeated MRI,\n~12.5 years", 6.2, ha="center")
text(6, 44.0, "LA dimension trajectory × APOE ε4\n→ rate of later lateral ventricular expansion", 6.4)
ax.plot([6, 40], [36.5, 36.5], color="#C9D1D9", lw=0.8)
text(6, 33.5, "Regional arm (N=451–765)", 7.5, "bold")
heart(12, 23.8, 3.5)
text(19.5, 27.5, "+", 10, "bold", MUTE, ha="center")
brain_icon(30, 24.5)
text(6, 16.5, "CMR × APOE ε4 → regional brain\nmorphometry (whole-brain screens)", 6.4)

# ---------------------------------------------------------------- column 2: main findings
box(46.5, 5, 76, 66, "#FFFFFF", "#9AA5B1")
text(84.5, 69, "Main findings", 9, "bold", ha="center")

# A: schematic of the longitudinal interaction (direction only)
axA = fig.add_axes([0.325, 0.335, 0.12, 0.39])
axA.plot([0, 1], [0.68, 0.84], color=NC, lw=2.0)
axA.plot([0, 1], [0.97, 0.65], color=CA, lw=2.0)
axA.set_xlim(-0.05, 1.05); axA.set_ylim(0.45, 1.1); axA.set_xticks([]); axA.set_yticks([])
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
axA.set_xlabel("Faster antecedent LA enlargement →", fontsize=6.2 * F)
axA.set_ylabel("Later ventricular expansion (mL/yr)", fontsize=6.2 * F)
axA.text(0.98, 0.875, "Noncarriers", color=NC, fontsize=6.4 * F, ha="right", va="bottom", fontweight="bold")
axA.text(0.98, 0.60, "ε4 carriers", color=CA, fontsize=6.4 * F, ha="right", va="top", fontweight="bold")
text(49.5, 63.5, "A  LA remodeling → ventricles", 7.3, "bold")
text(49.5, 21.0, "Trajectory × MRI time × APOE ε4:\nq=0.0023; random-slope q=0.034\n(schematic of direction)", 6.2, color=MUTE)

# B: published stratified LAVI slopes (Table 4 / Figure 5A)
axB = fig.add_axes([0.665, 0.335, 0.085, 0.39])
for i, (b, lo, hi, c) in enumerate(((0.045, -0.054, 0.144, NC), (-0.500, -0.725, -0.275, CA))):
    axB.plot([i, i], [lo, hi], color=c, lw=1.6)
    axB.plot(i, b, "o" if i == 0 else "s", ms=5, color=c)
axB.axhline(0, color=MUTE, lw=0.7)
axB.set_xlim(-0.7, 1.7); axB.set_ylim(-0.8, 0.3)
axB.set_xticks([0, 1]); axB.set_xticklabels(["Non-\ncarriers", "ε4\ncarriers"], fontsize=5.2 * F)
axB.tick_params(axis="y", labelsize=5.8 * F)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.set_ylabel("LAVI slope β (95% CI)", fontsize=6.0 * F, labelpad=1)
text(84, 63.5, "B  LA volume → cortex", 7.3, "bold")

if SURF.exists():
    axS = fig.add_axes([0.485, 0.46, 0.14, 0.25]); axS.imshow(plt.imread(SURF)); axS.axis("off")
else:
    brain(89.5, 48.5, 4.6)
text(88.5, 38.5, "Left middle\noccipital gyrus", 6.2, color=CA, weight="bold", ha="center")
text(81, 21.0, "Cortical-thickness variability; q=0.017\nacross 1,086 outcomes; permutation\nP=0.041; carriers n=96", 6.2, color=MUTE)
text(49.5, 10.8, "Regional ventricular (LV) findings were exploratory: their number was consistent\n"
                 "with chance in a permutation test of 1,960 tests (P=0.70).", 6.3, style="italic", color=MUTE)

# ---------------------------------------------------------------- column 3: interpretation
box(125.5, 5, 31.5, 66, "#F6F3EC", "#8A7444")
text(141.25, 69, "Interpretation", 9, "bold", ha="center")
text(128, 61.5, "APOE ε4 changed the form\nof atrial–brain relationships\nrather than uniformly\namplifying adverse\nassociations.\n\n"
                "Candidate mechanisms:\nisoform-dependent\nmyocardial remodeling and\ncerebrovascular regulation.\n\n"
                "Observational, single cohort:\nreplication with prespecified\nmodels is needed.", 6.9)
arrow(43, 38, 46.5, 38); arrow(122.5, 38, 125.5, 38)

for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
    fig.savefig(OUT / f"GraphicalAbstract.{ext}", **kw)
print("written to", OUT)
