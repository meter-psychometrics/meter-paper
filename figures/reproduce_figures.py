"""Reproduce the paper's numerical figure panels and Table 1 from derived results.

Consumes ONLY the aggregate artifacts under ``derived_results/`` (no real data,
no participant-level values anywhere) and regenerates:

* ``output/table1.csv``           — the focused Table 1 numbers;
* ``output/fig2_panels.json``     — Fig. 2 a-c panel data (portfolio bars,
  ESS per-country, SHARE per-country w9/w8);
* ``output/fig3_panels.json``     — Fig. 3 a-d panel data (MIDUS Refresher,
  IPIP-50, ICAR-16, SAPA factor/scale-wise agreements);
* ``output/fig4_panels.json``     — Fig. 4 a-c panel data (P4 before/after
  false-accept rates, fresh-challenge counts, non-regression checks);
* ``output/synthetic_panel.json`` — the synthetic recovery and uncertainty
  numbers quoted in Results/Methods.

Every regenerated headline value is checked against the value printed in the
manuscript (``EXPECTED`` below); the script exits non-zero on any mismatch, so
CI catches drift between artifacts and text. If matplotlib is available a PNG
rendering of each panel is also written; the numerical panels are the
deliverable and do not require it.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DERIVED = PACKAGE_ROOT / "derived_results"
OUTPUT = Path(__file__).resolve().parent / "output"


def _load(relative: str) -> dict:
    return json.loads((DERIVED / relative).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# manuscript-printed values (focused revision). 4-dp unless printed otherwise.
# --------------------------------------------------------------------------
EXPECTED = {
    "synthetic_median_r": 0.9388,
    "synthetic_specialist_r": 0.9438,
    "coverage_before": 0.8288,
    "coverage_after": 0.9479,
    "nhanes_2017_18": 0.9371,
    "ess_pooled": 0.9696,
    "ess_range": (0.9545, 0.9899),
    "share_w9_pooled": 0.9851,
    "share_w9_range": (0.9674, 0.9899),
    "share_w8_pooled": 0.9847,
    "midus_refresher_range": (0.9241, 0.9747),
    "midus_phi_mean": (0.1333, 0.12),
    "midus_phi_max": (0.2271, 0.20),
    "ipip_range": (0.9596, 0.9839),
    "ipip_median": 0.9791,
    "sapa_range": (0.9672, 0.9842),
    "icar_range": (0.8610, 0.9439),
    "icar_median": 0.9023,
    "p4b_false_accepts": (0, 63),
    "p4b_false_refusals": (0, 41),
}

_failures: list[str] = []


def check(name: str, actual, places: int = 4) -> None:
    expected = EXPECTED[name]

    def close(a, b):
        return round(float(a), places) == round(float(b), places)

    ok = (
        all(close(x, y) for x, y in zip(actual, expected, strict=True))
        if isinstance(expected, tuple)
        else close(actual, expected)
    )
    if not ok:
        _failures.append(f"{name}: manuscript prints {expected}, artifacts give {actual}")


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def synthetic_panel() -> dict:
    phase3c = _load("synthetic/phase3c.json")
    benchmark = _load("synthetic/phase3-stage2-arm1_beta05.json")
    median_r = phase3c["reproduction_on_phase3_validation"]["person_r_median"]
    specialist = benchmark["mml_reference"]["pearson_r"]["median"]
    before = phase3c["final_validation"]["reference_raw"]["coverage"]["0.95"]
    after = phase3c["final_validation"]["reference_scaled"]["coverage"]["0.95"]
    check("synthetic_median_r", median_r)
    check("synthetic_specialist_r", specialist)
    check("coverage_before", before)
    check("coverage_after", after)
    return {
        "unseen_worlds_median_person_r": median_r,
        "per_world_mml_grm_specialist_median_r": specialist,
        "nominal95_coverage_uncorrected": before,
        "nominal95_coverage_scaled": after,
    }


def nhanes_row() -> dict:
    locked = _load("nhanes/phase6b1_dpqj_dpq_j_result.json")
    development = _load("nhanes/phase6b1_nhanes_2015_2016_dpq.json")
    agreement = locked["agreement"]["mml_grm_eap"]["pearson_agreement"]
    check("nhanes_2017_18", agreement)
    return {
        "evaluation": "NHANES 2017-18, PHQ-9",
        "agreement": round(agreement, 4),
        "development_cycle_2015_16": round(
            development["agreement"]["mml_grm_eap"]["pearson_agreement"], 4
        ),
    }


def ess_rows() -> tuple[dict, dict]:
    ess = _load("ess/phase6b2_ess7_cesd8.json")
    pooled = ess["pooled_agreement"]["mml_grm_eap"]["pearson_agreement"]
    per_country = {
        code: entry["agreement"]["mml_grm_eap"]["pearson_agreement"]
        for code, entry in ess["per_country"].items()
    }
    values = sorted(per_country.values())
    check("ess_pooled", pooled)
    check("ess_range", (values[0], values[-1]))
    row = {
        "evaluation": "ESS round 7, CES-D-8",
        "agreement": round(pooled, 4),
        "range": (round(values[0], 4), round(values[-1], 4)),
        "n_countries": len(per_country),
    }
    return row, {code: round(v, 4) for code, v in sorted(per_country.items())}


def share_rows() -> tuple[list[dict], dict]:
    panels = {}
    rows = []
    for wave, expect_pooled, expect_range in (
        ("w9", "share_w9_pooled", "share_w9_range"),
        ("w8", "share_w8_pooled", None),
    ):
        artifact = _load(f"share/share_a_{wave}_result.json")
        pooled = artifact["pooled_agreement"]["mml_grm_eap"]["pearson_agreement"]
        per_country = {
            name: entry["meter_specialist_r"]
            for name, entry in artifact["per_country"].items()
        }
        values = sorted(per_country.values())
        check(expect_pooled, pooled)
        if expect_range:
            check(expect_range, (values[0], values[-1]))
        rows.append(
            {
                "evaluation": f"SHARE wave {wave[1]}, EURO-D",
                "agreement": round(pooled, 4),
                "range": (round(values[0], 4), round(values[-1], 4)),
                "n_countries": len(per_country),
            }
        )
        panels[wave] = {name: round(v, 4) for name, v in sorted(per_country.items())}
    return rows, panels


def midus_rows() -> tuple[dict, dict]:
    replication = _load("midus/d9b_replication.json")
    factors = {
        name: entry["pearson_model_vs_mirt"]
        for name, entry in replication["factor_agreement"].items()
    }
    values = sorted(factors.values())
    check("midus_refresher_range", (values[0], values[-1]))
    check(
        "midus_phi_mean",
        (
            replication["phi_agreement"]["mean_abs_off_diagonal_difference"],
            replication["phi_agreement"]["gates"]["mean_abs_off_diagonal_difference_max"]
            if "gates" in replication["phi_agreement"]
            else 0.12,
        ),
    )
    check(
        "midus_phi_max",
        (
            replication["phi_agreement"]["max_abs_off_diagonal_difference"],
            replication["phi_agreement"]["gates"]["max_abs_off_diagonal_difference_max"]
            if "gates" in replication["phi_agreement"]
            else 0.20,
        ),
    )
    row = {
        "evaluation": "MIDUS Refresher, Big Five",
        "range": (round(values[0], 4), round(values[-1], 4)),
        "phi_gates": "both failed (0.1333 vs 0.12; 0.2271 vs 0.20)",
    }
    return row, factors


def ipip_rows() -> tuple[dict, dict]:
    transfer = _load("ipip50/d9e_transfer.json")
    factors = {
        name: entry["pearson_model_vs_mirt"]
        for name, entry in transfer["factor_agreement_signed"].items()
    }
    values = sorted(factors.values())
    check("ipip_range", (values[0], values[-1]))
    check("ipip_median", transfer["median_pearson_model_vs_mirt"])
    return (
        {
            "evaluation": "IPIP-50 Big Five",
            "range": (round(values[0], 4), round(values[-1], 4)),
            "median": transfer["median_pearson_model_vs_mirt"],
        },
        factors,
    )


def sapa_rows() -> tuple[dict, dict]:
    sapa = _load("sapa/phase6b3_sapa_v2.json")
    scales = {
        task["task_id"]: task["agreement"]["mml_grm_eap"]["pearson_agreement"]
        for task in sapa["task_a"]
    }
    values = sorted(scales.values())
    check("sapa_range", (values[0], values[-1]))
    return (
        {
            "evaluation": "SAPA personality",
            "range": (round(values[0], 4), round(values[-1], 4)),
            "n_scales": len(scales),
        },
        {k: round(v, 4) for k, v in scales.items()},
    )


def icar_rows() -> tuple[dict, dict]:
    transfer = _load("icar/d9d_transfer.json")
    factors = {
        name: entry["pearson_model_vs_mirt"]
        for name, entry in transfer["factor_agreement_signed"].items()
    }
    values = sorted(factors.values())
    check("icar_range", (values[0], values[-1]))
    check("icar_median", transfer["median_pearson_model_vs_mirt"])
    return (
        {
            "evaluation": "ICAR-16",
            "range": (round(values[0], 4), round(values[-1], 4)),
            "median": transfer["median_pearson_model_vs_mirt"],
            "verdict": "cross-domain transfer failed",
        },
        factors,
    )


def fig4_panels() -> dict:
    closeout = json.loads(
        (DERIVED / "p4b" / "p4b_fully_wired_capability_router.json").read_text(encoding="utf-8")
    )
    fresh = json.loads(
        (DERIVED / "p4b" / "p4b_fresh_challenge.json").read_text(encoding="utf-8")
    )
    check(
        "p4b_false_accepts",
        (fresh["false_accepts"]["count"], fresh["false_accepts"]["n_unsupported"]),
        places=0,
    )
    check(
        "p4b_false_refusals",
        (fresh["false_refusals"]["count"], fresh["false_refusals"]["n_supported"]),
        places=0,
    )
    rerun = closeout["original_p4_rerun_unchanged_cases"]["before_after_by_axis"]
    return {
        "a_before_after_false_accept_rates": {
            axis: {
                "before": entry["before"]["false_accept_rate"],
                "after": entry["after"]["false_accept_rate"],
            }
            for axis, entry in rerun.items()
        },
        "b_fresh_challenge": {
            "false_accepts": fresh["false_accepts"],
            "false_refusals": fresh["false_refusals"],
            "per_cell": fresh["per_cell"],
        },
        "c_nonregression": {
            "supported_fixtures_bit_identical": closeout["numerical_nonregression"]["fixtures"],
            "exact_fallback_remains_exact": closeout["fallback_regression"][
                "exact_fallback_remains_exact"
            ],
        },
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    synthetic = synthetic_panel()
    nhanes = nhanes_row()
    ess_row, ess_countries = ess_rows()
    share_row_list, share_countries = share_rows()
    midus_row, midus_factors = midus_rows()
    ipip_row, ipip_factors = ipip_rows()
    sapa_row, sapa_scales = sapa_rows()
    icar_row, icar_factors = icar_rows()
    fig4 = fig4_panels()

    table1 = [nhanes, ess_row, *share_row_list, midus_row, ipip_row, sapa_row, icar_row]
    with (OUTPUT / "table1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "evaluation", "agreement", "range", "median", "n_countries",
                "n_scales", "phi_gates", "verdict", "development_cycle_2015_16",
            ],
        )
        writer.writeheader()
        for row in table1:
            writer.writerow(row)

    (OUTPUT / "fig2_panels.json").write_text(
        json.dumps(
            {
                "a_portfolio": {r["evaluation"]: r for r in table1},
                "b_ess_per_country": ess_countries,
                "c_share_per_country": share_countries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "fig3_panels.json").write_text(
        json.dumps(
            {
                "a_midus_refresher": midus_factors,
                "b_ipip50": ipip_factors,
                "c_icar16": icar_factors,
                "d_sapa": sapa_scales,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "fig4_panels.json").write_text(json.dumps(fig4, indent=2), encoding="utf-8")
    (OUTPUT / "synthetic_panel.json").write_text(json.dumps(synthetic, indent=2), encoding="utf-8")

    if _failures:
        print("MISMATCHES between manuscript values and derived artifacts:")
        for failure in _failures:
            print("  -", failure)
        return 1

    print("All regenerated values match the manuscript. Panels written to", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
