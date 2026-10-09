#!/usr/bin/env python3
"""
Table 1 - Characteristics of the regional and longitudinal analytic cohorts.

Replaces 137/140. Differences from 140:
  * Panel A is COMPUTED from the regional analysis dataset.
  * No participant-level file is written.
  * Variables available in only part of the sample show their n.
  * One P-value format throughout.
  * Small cells (1-5) are suppressed as "≤5".
  * Every published cell is checked against the manuscript VERIFY block.

Outputs:
  Table1_PanelA.csv
  Table1_PanelB.csv
  Table1.xlsx
  Table1_verification.csv

Usage:
  python make_Table1.py [--strict] [--outdir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/data/qiallab/Framingham")

REGIONAL = (
    ROOT
    / "results"
    / "neurocardiac_metadata_analysis_ready_metabolic.csv"
)

ATRIAL = (
    ROOT
    / "results"
    / "neurocardiac_metadata_with_atria.csv"
)

LONGFILE = (
    ROOT
    / "data/longitudinal_derived"
    / "E4_E6_repeated_MRI_temporal_analysis.tsv"
)


COLS = {
    "id": [
        "shareid",
        "dbgap_subject_id",
    ],
    "genotype": [
        "APOE_genotype",
        "apoe_genotype",
        "APOE",
        "apoe",
        "APOE_GENOTYPE",
    ],
    "carrier": [
        "APOE4_carrier",
    ],
    "age": [
        "age_at_cmr",
        "age_cmr",
        "age_at_exam",
        "age_nearest_exam",
        "age_at_mri",
    ],
    "sex": [
        "sex_clinical",
        "sex",
    ],
    "bmi": [
        "BMI_nearest_exam",
        "BMI",
    ],
    "sbp": [
        "SBP_nearest_exam",
        "SBP",
    ],
    "diabetes": [
        "diabetes_history_nearest_exam",
        "diabetes_history_any",
        "diabetes",
    ],
    "smoking": [
        "current_smoker_nearest_exam",
        "current_smoking",
    ],
    "interval": ["_cmr_mri_interval_years"],
    "LVEF": ["LVEF"],
    "LVEDVi": ["LVEDVi", "_LVEDVi"],
    "LVESVi": ["LVESVi", "_LVESVi"],
    "LVMI": [
        "LV_MASSi",
        "LV_mass_index",
        "_LV_MASSi",
    ],
    "LAVI": [
        "LAVI_max",
    ],
    "LAEF": ["_LAEF_pct", "LA_total_emptying_fraction"],
}

SEX_FEMALE = 2
SMALL_CELL = 5


VERIFY = {
    "A": {
        "N": 769,
        "noncarriers": 596,
        "carriers": 173,
        "Age at CMR, years": "61.77 ± 8.22",
        "Female sex, n (%)": "448 (58.3%)",
        "LVEF, %": "60.40 ± 5.60",
        "LV mass index, g/m²": "45.91 ± 8.86",
        "Absolute CMR–brain MRI interval, years": "2.96 ± 1.25",
    },
    "B": {
        "N": 1305,
        "noncarriers": 1002,
        "carriers": 303,
        "obs": 3787,
        "Age at Examination 6, years": "56.64 ± 9.13",
        "Female sex, n (%)": "685 (52.5%)",
        "LA dimension trajectory, cm/year": "0.0355 ± 0.0604",
    },
}


def col(df, key, required=True):
    for c in COLS[key]:
        if c in df.columns:
            return c

    if required:
        sys.exit(
            f"None of {COLS[key]} found for '{key}'. "
            f"Add the correct name to COLS."
        )

    return None


def num(s):
    return pd.to_numeric(
        s,
        errors="coerce"
    )


def mean_sd(x, d=2):
    x = num(x).dropna()

    if not len(x):
        return ""

    return (
        f"{x.mean():.{d}f} ± "
        f"{x.std(ddof=1):.{d}f}"
    )


def median_iqr(x, d=2):
    x = num(x).dropna()

    if not len(x):
        return ""

    return (
        f"{x.median():.{d}f} "
        f"({x.quantile(.25):.{d}f}–"
        f"{x.quantile(.75):.{d}f})"
    )


def n_pct(x, positive=1):
    x = num(x).dropna()

    if not len(x):
        return ""

    n = int(
        (x == positive).sum()
    )

    pct = (
        100
        * n
        / len(x)
    )

    if 0 < n <= SMALL_CELL:
        return (
            f"≤{SMALL_CELL} "
            f"(≤{100 * SMALL_CELL / len(x):.1f}%)"
        )

    return (
        f"{n} "
        f"({pct:.1f}%)"
    )


def fmt_p(p):
    if p is None or np.isnan(p):
        return ""

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


def p_cont(x, g):
    z = pd.DataFrame(
        {
            "x": num(x),
            "g": num(g),
        }
    ).dropna()

    a = z.x[
        z.g == 0
    ]

    b = z.x[
        z.g == 1
    ]

    if len(a) > 1 and len(b) > 1:
        return fmt_p(
            stats.ttest_ind(
                a,
                b,
                equal_var=False
            ).pvalue
        )

    return ""


def p_bin(x, g):
    z = pd.DataFrame(
        {
            "x": num(x),
            "g": num(g),
        }
    ).dropna()

    t = pd.crosstab(
        z.x,
        z.g
    )

    if t.shape == (2, 2):
        return fmt_p(
            stats.fisher_exact(
                t.values
            )[1]
        )

    return ""


class Panel:

    def __init__(
        self,
        d,
        carrier_col,
        n_label
    ):
        self.d = d
        self.g = d[
            carrier_col
        ]

        self.rows = []

        self.non = d[
            self.g == 0
        ]

        self.car = d[
            self.g == 1
        ]

        self.cols = [
            f"{n_label} "
            f"(N = {len(d):,})",

            f"APOE ε4 noncarriers "
            f"(n = {len(self.non):,})",

            f"APOE ε4 carriers "
            f"(n = {len(self.car):,})",
        ]

    def header(
        self,
        text
    ):
        self.rows.append(
            {
                "Characteristic": text
            }
        )

    def cont(
        self,
        label,
        c,
        d=2
    ):
        n = (
            num(
                self.d[c]
            )
            .notna()
            .sum()
        )

        if n < len(self.d):
            label = (
                f"{label} "
                f"(n = {n:,})"
            )

        self.rows.append(
            {
                "Characteristic":
                    label,

                self.cols[0]:
                    mean_sd(
                        self.d[c],
                        d
                    ),

                self.cols[1]:
                    mean_sd(
                        self.non[c],
                        d
                    ),

                self.cols[2]:
                    mean_sd(
                        self.car[c],
                        d
                    ),

                "P value":
                    p_cont(
                        self.d[c],
                        self.g
                    ),
            }
        )

    def binary(
        self,
        label,
        x
    ):
        self.rows.append(
            {
                "Characteristic":
                    label,

                self.cols[0]:
                    n_pct(
                        x
                    ),

                self.cols[1]:
                    n_pct(
                        x[
                            self.g == 0
                        ]
                    ),

                self.cols[2]:
                    n_pct(
                        x[
                            self.g == 1
                        ]
                    ),

                "P value":
                    p_bin(
                        x,
                        self.g
                    ),
            }
        )

    def raw(
        self,
        label,
        a,
        b,
        c,
        p=""
    ):
        self.rows.append(
            {
                "Characteristic":
                    label,

                self.cols[0]:
                    a,

                self.cols[1]:
                    b,

                self.cols[2]:
                    c,

                "P value":
                    p,
            }
        )

    def frame(self):
        return pd.DataFrame(
            self.rows,
            columns=[
                "Characteristic",
                *self.cols,
                "P value",
            ]
        ).fillna("")


def panel_a():

    reg = pd.read_csv(
        REGIONAL,
        low_memory=False
    )

    # Body surface area and indexed LV measures:
    # exact preprocessing used in validated regional analyses.
    required = ["height_in", "weight_lb", "LVEDV", "LVESV", "LV_MASS"]
    missing = [c for c in required if c not in reg.columns]

    if missing:
        raise RuntimeError(
            "Cannot derive indexed CMR measures; missing columns: "
            + ", ".join(missing)
        )

    height_cm = num(reg["height_in"]) * 2.54
    weight_kg = num(reg["weight_lb"]) * 0.45359237

    reg["BSA_m2"] = np.sqrt(
        height_cm * weight_kg / 3600.0
    )

    reg["LVEDVi"] = num(reg["LVEDV"]) / reg["BSA_m2"]
    reg["LVESVi"] = num(reg["LVESV"]) / reg["BSA_m2"]
    reg["LV_MASSi"] = num(reg["LV_MASS"]) / reg["BSA_m2"]

    print("Derived indexed CMR measures using validated formula:")
    print("  BSA_m2   = sqrt(height_cm * weight_kg / 3600)")
    print("  LVEDVi   = LVEDV / BSA_m2")
    print("  LVESVi   = LVESV / BSA_m2")
    print("  LV_MASSi = LV_MASS / BSA_m2")
    print("  Valid BSA:", int(reg["BSA_m2"].notna().sum()), "of", len(reg))

    if "CMRDATE" not in reg.columns or "mri_date" not in reg.columns:
        raise RuntimeError(
            "CMRDATE and/or mri_date missing; cannot derive exact CMR-MRI interval."
        )

    reg["_cmr_mri_interval_years"] = (
        num(reg["CMRDATE"]) - num(reg["mri_date"])
    ).abs() / 365.25

    print(
        "Derived exact CMR-brain MRI interval:",
        f'{reg["_cmr_mri_interval_years"].mean():.4f} ± '
        f'{reg["_cmr_mri_interval_years"].std(ddof=1):.4f} years'
    )

    # Recreate the BSA-indexed CMR measures used in the regional
    # analysis and in the historical Table 1 generator.
    needed_raw = ["LVEDV", "LVESV", "LV_MASS", "BSA_m2"]
    missing_raw = [c for c in needed_raw if c not in reg.columns]

    if missing_raw:
        raise RuntimeError(
            "Cannot derive indexed LV measures. Missing columns: "
            + str(missing_raw)
        )

    bsa = num(reg["BSA_m2"])
    bad_bsa = (~np.isfinite(bsa)) | (bsa <= 0)
    bsa = bsa.mask(bad_bsa)

    reg["_LVEDVi"] = num(reg["LVEDV"]) / bsa
    reg["_LVESVi"] = num(reg["LVESV"]) / bsa
    reg["_LV_MASSi"] = num(reg["LV_MASS"]) / bsa

    print(
        "Derived indexed CMR measures from raw values / BSA_m2:",
        "_LVEDVi, _LVESVi, _LV_MASSi"
    )

    idc = col(
        reg,
        "id"
    )

    if ATRIAL.exists():

        atr = pd.read_csv(
            ATRIAL,
            low_memory=False
        )

        add = [
            c
            for c in (
                COLS["LAVI"]
                + COLS["LAEF"]
            )
            if (
                c in atr.columns
                and c not in reg.columns
            )
        ]

        if add:
            reg = reg.merge(
                atr[
                    [idc] + add
                ].drop_duplicates(
                    idc
                ),
                on=idc,
                how="left"
            )

    if "LA_total_emptying_fraction" in reg.columns:
        reg["_LAEF_pct"] = num(reg["LA_total_emptying_fraction"]) * 100.0

    cmr = [
        col(
            reg,
            k
        )
        for k in (
            "LVEF",
            "LVEDVi",
            "LVESVi",
            "LVMI",
        )
    ]

    d = (
        reg[
            reg[cmr]
            .notna()
            .any(axis=1)
        ]
        .drop_duplicates(
            idc
        )
        .copy()
    )

    print(
        "Panel A: participants with "
        f"CMR + brain MRI = {len(d)} "
        "(published 785)"
    )

    gcol = col(
        d,
        "genotype"
    )

    geno = (
        d[gcol]
        .astype(str)
        .str.upper()
        .str.replace(
            r"[^234]",
            "",
            regex=True
        )
    )

    geno = (
        geno
        .where(
            geno.str.len() == 2
        )
        .map(
            lambda s:
                "".join(
                    sorted(s)
                )
                if isinstance(
                    s,
                    str
                )
                else s
        )
    )

    d["_geno"] = geno

    d = d[
        d["_geno"].isin(
            [
                "22",
                "23",
                "24",
                "33",
                "34",
                "44",
            ]
        )
    ].copy()

    d["_carrier"] = (
        d["_geno"]
        .str.contains(
            "4"
        )
        .astype(int)
    )

    ccol = col(
        d,
        "carrier",
        required=False
    )

    if ccol is not None:

        mism = (
            num(
                d[ccol]
            )
            != d["_carrier"]
        ).sum()

        print(
            "Panel A: genotype-derived carrier "
            f"vs {ccol}: {mism} mismatches"
        )

    d["_female"] = (
        num(
            d[
                col(
                    d,
                    "sex"
                )
            ]
        )
        == SEX_FEMALE
    ).astype(int)

    dia = num(
        d[
            col(
                d,
                "diabetes"
            )
        ]
    )

    d["_diab"] = np.where(
        dia == 0,
        0,
        np.where(
            dia.isin(
                [1, 2]
            ),
            1,
            np.nan
        )
    )

    print(
        "Panel A resolved columns:",
        {
            k:
                col(
                    d,
                    k,
                    required=False
                )
            for k in COLS
        }
    )

    if col(
        d,
        "age"
    ) == "age_at_mri":

        print(
            "WARNING: no age-at-CMR column found; "
            "using age at MRI. "
            "Add the age-at-CMR name to COLS['age']."
        )

    P = Panel(
        d,
        "_carrier",
        "APOE-classified sample"
    )

    P.header(
        "Demographic and clinical characteristics"
    )

    P.cont(
        "Age at CMR, years",
        col(
            d,
            "age"
        )
    )

    P.binary(
        "Female sex, n (%)",
        d["_female"]
    )

    P.cont(
        "BMI, kg/m²",
        col(
            d,
            "bmi"
        )
    )

    P.cont(
        "Systolic blood pressure, mmHg",
        col(
            d,
            "sbp"
        )
    )

    P.binary(
        "Diabetes, n (%)",
        d["_diab"]
    )

    P.binary(
        "Current smoking, n (%)",
        num(
            d[
                col(
                    d,
                    "smoking"
                )
            ]
        )
    )

    P.cont(
        "Absolute CMR–brain MRI interval, years",
        col(
            d,
            "interval"
        )
    )

    P.header(
        "Cardiac imaging characteristics"
    )

    for lab, k in (
        (
            "LVEF, %",
            "LVEF"
        ),
        (
            "LVEDVi, mL/m²",
            "LVEDVi"
        ),
        (
            "LVESVi, mL/m²",
            "LVESVi"
        ),
        (
            "LV mass index, g/m²",
            "LVMI"
        ),
        (
            "LAVI maximum, mL/m²",
            "LAVI"
        ),
        (
            "LA total emptying fraction, %",
            "LAEF"
        ),
    ):

        c = col(
            d,
            k,
            required=False
        )

        if c:
            P.cont(
                lab,
                c
            )

    P.header(
        "APOE genotype, n (%)"
    )

    for gt, lab in (
        (
            "22",
            "ε2/ε2"
        ),
        (
            "23",
            "ε2/ε3"
        ),
        (
            "24",
            "ε2/ε4"
        ),
        (
            "33",
            "ε3/ε3"
        ),
        (
            "34",
            "ε3/ε4"
        ),
        (
            "44",
            "ε4/ε4"
        ),
    ):

        x = (
            d["_geno"]
            == gt
        ).astype(int)

        P.raw(
            f"   {lab}",
            n_pct(
                x
            ),
            n_pct(
                x[
                    d._carrier
                    == 0
                ]
            ),
            n_pct(
                x[
                    d._carrier
                    == 1
                ]
            )
        )

    return P, d


def panel_b():

    df = pd.read_csv(
        LONGFILE,
        sep="\t",
        low_memory=False
    )

    d = df[
        (
            num(
                df["la_dim_n"]
            )
            >= 3
        )
        & df[
            "la_dim_slope"
        ].notna()
        & df[
            "APOE4_carrier"
        ].notna()
        & df[
            "age6"
        ].notna()
        & df[
            "sex"
        ].notna()
        & (
            num(
                df[
                    "last_echo_to_first_mri_years"
                ]
            )
            >= 0
        )
        & df[
            "Lateralvent"
        ].notna()
        & df[
            "mri_time_years"
        ].notna()
    ].copy()

    keep = (
        d.groupby(
            "shareid"
        )[
            "Lateralvent"
        ]
        .count()
    )

    d = d[
        d[
            "shareid"
        ].isin(
            keep[
                keep >= 2
            ].index
        )
    ].copy()

    person = (
        d.sort_values(
            [
                "shareid",
                "mri_time_years",
            ]
        )
        .drop_duplicates(
            "shareid"
        )
        .set_index(
            "shareid"
        )
    )

    person["n_obs"] = (
        d.groupby(
            "shareid"
        )
        .size()
    )

    t = (
        d.groupby(
            "shareid"
        )[
            "mri_time_years"
        ]
        .agg(
            [
                "min",
                "max",
            ]
        )
    )

    person["follow"] = (
        t["max"]
        - t["min"]
    )

    person["_female"] = (
        num(
            person["sex"]
        )
        == SEX_FEMALE
    ).astype(int)

    person = (
        person
        .reset_index()
    )

    P = Panel(
        person,
        "APOE4_carrier",
        "APOE-classified sample"
    )

    obs = (
        d.groupby(
            "APOE4_carrier"
        )
        .size()
    )

    P.header(
        "Longitudinal imaging characteristics"
    )

    P.raw(
        "MRI observations, n",
        f"{len(d):,}",
        f"{obs.get(0, 0):,}",
        f"{obs.get(1, 0):,}"
    )

    for lab, c, dg in (
        (
            "MRI observations per participant, median (IQR)",
            "n_obs",
            0
        ),
        (
            "MRI follow-up, years, median (IQR)",
            "follow",
            2
        ),
        (
            "Last cardiac examination to first brain MRI, years, median (IQR)",
            "last_echo_to_first_mri_years",
            2
        ),
    ):

        P.raw(
            lab,
            median_iqr(
                person[c],
                dg
            ),
            median_iqr(
                P.non[c],
                dg
            ),
            median_iqr(
                P.car[c],
                dg
            )
        )

    P.header(
        "Participant characteristics"
    )

    P.cont(
        "Age at Examination 6, years",
        "age6"
    )

    P.binary(
        "Female sex, n (%)",
        person["_female"]
    )

    P.cont(
        "LA dimension trajectory, cm/year",
        "la_dim_slope",
        d=4
    )

    return P, person, len(d)


def verify(
    PA,
    PB,
    nobs
):

    out = []

    for key, P, extra in (
        (
            "A",
            PA,
            {}
        ),
        (
            "B",
            PB,
            {
                "obs":
                    nobs
            }
        ),
    ):

        exp = VERIFY[key]

        f = P.frame()

        got = {
            "N":
                len(P.d),

            "noncarriers":
                len(P.non),

            "carriers":
                len(P.car),

            **extra,
        }

        for k, v in exp.items():

            if k in got:
                g = got[k]

            else:

                r = f[
                    f[
                        "Characteristic"
                    ]
                    .str.startswith(
                        k
                    )
                ]

                g = (
                    r.iloc[
                        0,
                        1
                    ]
                    if len(r)
                    else "(row missing)"
                )

            out.append(
                {
                    "panel":
                        key,

                    "cell":
                        k,

                    "published":
                        v,

                    "computed":
                        g,

                    "match":
                        str(v)
                        == str(g),
                }
            )

    return pd.DataFrame(
        out
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--strict",
        action="store_true"
    )

    ap.add_argument(
        "--outdir",
        default=str(
            ROOT
            / "results/paper_figures_tables/Table1"
        )
    )

    a = ap.parse_args()

    out = Path(
        a.outdir
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    PA, _ = panel_a()

    PB, _, nobs = panel_b()

    A = PA.frame()

    B = PB.frame()

    A.to_csv(
        out
        / "Table1_PanelA.csv",
        index=False
    )

    B.to_csv(
        out
        / "Table1_PanelB.csv",
        index=False
    )

    with pd.ExcelWriter(
        out
        / "Table1.xlsx"
    ) as w:

        A.to_excel(
            w,
            sheet_name="Panel A regional",
            index=False
        )

        B.to_excel(
            w,
            sheet_name="Panel B longitudinal",
            index=False
        )

    v = verify(
        PA,
        PB,
        nobs
    )

    v.to_csv(
        out
        / "Table1_verification.csv",
        index=False
    )

    print(
        "\n"
        + A.to_string(
            index=False
        )
        + "\n\n"
        + B.to_string(
            index=False
        )
    )

    print(
        "\nVERIFICATION\n"
        + v.to_string(
            index=False
        )
    )

    if (
        a.strict
        and not v[
            "match"
        ].all()
    ):

        sys.exit(
            "STRICT: computed values differ from the published table "
            "(see Table1_verification.csv)."
        )

    print(
        f"\nwritten to {out} "
        "(summary statistics only; no participant-level output)"
    )


if __name__ == "__main__":
    main()
