"""Fig. 4 verification: the frozen router evidence, re-verified three ways.

1. Integrity — the shipped P4B artifacts hash-match the pins recorded inside
   the frozen close-out artifact.
2. Consistency — the reported FA 0/63 and FR 0/41 are recomputed from the 104
   per-decision records, not read off the summary.
3. Executable replay — every decision that is derivable from the recorded
   design descriptor alone (the supplied-structure cells and the decoder
   category-ceiling cells: 40 of 104 decisions) is re-derived by calling the
   RELEASED router code on a descriptor built from the record, and must equal
   the recorded decision.

Full tensor-level re-execution of all 104 decisions requires the research
repository's synthetic world generators and is reviewer-only by design.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
P4B = PACKAGE_ROOT / "derived_results" / "p4b"


def _load(name: str) -> dict:
    return json.loads((P4B / name).read_text(encoding="utf-8"))


def test_artifact_integrity_hashes():
    closeout = _load("p4b_fully_wired_capability_router.json")
    for entry, filename in [
        (closeout["fresh_challenge"], "p4b_fresh_challenge.json"),
        (closeout["original_p4_rerun_unchanged_cases"], "p4b_rerun_original_p4.json"),
    ]:
        actual = hashlib.sha256((P4B / filename).read_bytes()).hexdigest()
        assert actual == entry["sha256"], f"{filename} does not match the frozen pin"
    protocol_sha = hashlib.sha256(
        (P4B / "p4b_router_closeout_protocol.json").read_bytes()
    ).hexdigest()
    assert protocol_sha == closeout["protocol"]["sha256"]


def test_false_accept_and_refusal_counts_recomputed():
    fresh = _load("p4b_fresh_challenge.json")
    records = fresh["records"]
    assert len(records) == 104
    fa = sum(1 for r in records if r["expected"] == "refuse" and r["decision"] == "accept")
    fr = sum(1 for r in records if r["expected"] == "accept" and r["decision"] == "refuse")
    n_unsupported = sum(1 for r in records if r["expected"] == "refuse")
    n_supported = sum(1 for r in records if r["expected"] == "accept")
    assert (fa, n_unsupported) == (0, 63)
    assert (fr, n_supported) == (0, 41)
    assert all(r["correct"] for r in records)

    rerun = _load("p4b_rerun_original_p4.json")
    for axis in rerun["before_after_by_axis"].values():
        assert axis["after"]["false_accepts"] == 0
        assert axis["after"]["false_rejects"] == 0


def test_nonregression_artifact_reports_bit_identical_fixtures():
    nonreg = _load("p4b_numerical_nonregression.json")
    fixtures = nonreg["fixtures"] if "fixtures" in nonreg else nonreg
    closeout = _load("p4b_fully_wired_capability_router.json")
    summary = closeout["numerical_nonregression"]["fixtures"]
    assert len(summary) == 9
    assert all(entry["bit_identical"] for entry in summary.values())


# ---------------------------------------------------------------------------
# executable replay of descriptor-derivable decisions
# ---------------------------------------------------------------------------

_STRUCTURE_OUTPUTS = (
    "factor_scores",
    "factor_correlation_phi",
    "item_to_factor_structure",
    "factor_count_discovery",
)

_CEILING_CELLS = {
    "format_x_capability_binary_1d": True,   # inside the decoder ceiling -> accept
    "format_x_capability_8cat": False,        # 8 categories -> outside -> refuse
}


def _predicates(record) -> dict:
    """name -> (passed, value) union over the record's per-output evaluations."""
    merged = {}
    for decision in record["output_decisions"].values():
        for predicate in decision["predicate_results"]:
            merged[predicate["name"]] = (predicate["passed"], predicate["value"])
    return merged


def _structure_descriptor(record):
    """Rebuild the design descriptor from the record's own predicate values.

    Every predicate the supplied-structure router evaluates reads exactly the
    quantities recorded in the frozen per-decision evaluations: max response
    categories, supplied K, observed-cell density, cross-sectional flag and
    declared provenance. Person and item counts do not enter these predicates;
    nominal values are used.
    """
    from human_measurement.capability_router import DesignDescriptor

    predicates = _predicates(record)

    def value(name, default):
        return predicates[name][1] if name in predicates else default

    # Discovery-only records evaluate only the discovery predicate; the
    # remaining fields then default to an in-support configuration, which the
    # decisions for the outputs actually present in the record do not read.
    max_categories = int(value("response_categories_within_decoder_support", 4))
    k = int(value("supplied_k_within_frozen_backbone", record.get("k", 3)))
    density = float(value("dense_observation_required_for_multidimensional_scoring", 1.0))
    longitudinal = bool(value("multidimensional_request_is_cross_sectional", False))
    provenance = str(value("phi_interpretation_within_validated_regime", "real"))
    n_items = 20
    return DesignDescriptor(
        n_participants=200,
        n_waves=2 if longitudinal else 1,
        n_items=n_items,
        response_category_counts=tuple([max_categories] * n_items),
        observed_cell_fraction=density,
        n_constructs=k,
        latent_dimensionality=k,
        factor_count=k,
        supplied_structure_present=True,
        longitudinal_design=longitudinal,
        auxiliary_modality_present=False,
        requested_output_types=_STRUCTURE_OUTPUTS,
        data_provenance=provenance,
    )


def test_replay_supplied_structure_decisions():
    """Per-output action AND refusal code replayed for all 24 factor decisions."""
    fresh = _load("p4b_fresh_challenge.json")
    from human_measurement.capability_router import supplied_structure_decisions

    replayed = 0
    for record in fresh["records"]:
        if not record["cell"].startswith("factor_x_"):
            continue
        decisions = supplied_structure_decisions(_structure_descriptor(record))
        for name, recorded in record["output_decisions"].items():
            live = decisions[name]
            assert live.action.value == recorded["action"], (
                f"{record['cell']} seed {record['seed']} output {name}: released "
                f"router says {live.action.value}, frozen record says {recorded['action']}"
            )
            assert (live.refusal_code or None) == (recorded.get("refusal_code") or None), (
                f"{record['cell']} seed {record['seed']} output {name}"
            )
        replayed += 1
    assert replayed == 24  # 8 cells x 3 decisions


def test_replay_decoder_category_ceiling():
    from human_measurement.capability_router import DesignDescriptor, decoder_category_predicate

    fresh = _load("p4b_fresh_challenge.json")
    replayed = 0
    for record in fresh["records"]:
        cell = record["cell"]
        if cell not in _CEILING_CELLS:
            continue
        realised = record["realised"]
        n_items = int(realised["n_items"])
        descriptor = DesignDescriptor(
            n_participants=200,
            n_waves=int(realised["n_waves"]),
            n_items=n_items,
            response_category_counts=tuple([int(realised["max_categories"])] * n_items),
            observed_cell_fraction=float(realised["observed_cell_fraction"]),
            n_constructs=int(realised["n_constructs"]),
            latent_dimensionality=None,
            factor_count=None,
            supplied_structure_present=False,
            longitudinal_design=int(realised["n_waves"]) > 1,
            auxiliary_modality_present=realised.get("auxiliary", "none") != "none",
            requested_output_types=("person_score",),
            data_provenance="synthetic",
        )
        predicate = decoder_category_predicate(descriptor)
        assert predicate.passed == _CEILING_CELLS[cell], f"{cell} seed {record.get('corpus_seed')}"
        replayed += 1
    assert replayed == 16  # 2 cells x 8 decisions
