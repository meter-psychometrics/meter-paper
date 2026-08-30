"""P4B: the fully wired capability router (output-specific support evaluation).

The P4 router challenge (reports/planned/p4_router_challenge.json, commit
33f11f2) measured that three declared boundary axes - latent dimensionality,
factor count and observed-cell fraction - were declaration-only: no runtime
predicate read them, so challenged out-of-record designs were accepted with
ordinary-looking numbers. Response-category enforcement was partial (binary
only), and requests routed to ``multimodal_fusion`` bypassed the longitudinal
boundary entirely because boundary application was keyed to the answering
capability rather than to the design.

This module closes those gaps by WIRING, not by inventing. Every predicate
here cites an already-frozen record:

* decoder category ceiling: the frozen feature space expresses categories
  2..CMAX = 7 (``model.features.CMAX``; release contract MAX_CATEGORIES);
* longitudinal trained category band [4, 7]: the frozen 5E family table draws
  category counts from {(4,), (5,), (5, 7)} (``model.longitudinal_worlds``);
  binary keeps its stronger 6B.2.2 measured-bias reason;
* longitudinal construct envelope: the 5E families contain one- and
  two-construct designs only (``b5_two_construct``);
* longitudinal density envelope: trained nominal response-rate floor 0.7
  (family ``e1_mcar_items``; phase3 core floor 0.75), with the 6B.2.3
  MATERIAL multiple (3.0) marking a design MATERIALLY outside - the same
  material-envelope convention the frozen module applies to wave spacing;
* supplied-structure factor scoring: 1 <= K <= K_MAX = 5 (M8B), dense
  observation required (M8 closeout), K=2 expressible but outside the
  confirmatory range;
* unrestricted structure/factor-count discovery: refused (M7 closure);
* real-data Phi: refused (D9B REPLICATION_FAIL); synthetic dense Phi is a
  diagnostic within the validated regime (M8).

Nothing here trains, recalibrates, or changes a threshold, weight, or frozen
artifact. Authorization is evaluated separately for each requested output:
the same input may receive different decisions for different outputs, and a
refused output is withheld (the existing suppression rule), never serialized
with a warning attached.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from human_measurement.longitudinal_design import (
    BINARY_CATEGORY_COUNT,
    CHANGE_ATTENUATION_WARNING,
    LONGITUDINAL_CONTRACTS,
    MATERIAL_SPACING_MULTIPLE,
    OUT_OF_SUPPORT_WARNING,
    TRAIT_ABSORBS_STATE_WARNING,
    TRAIT_SHARE_BIAS_WARNING,
    DesignRouting,
)
from human_measurement.model.features import CMAX

ROUTER_VERSION = "p4b-capability-router-v1"

#: The contract this router enforces. Asserted against the frozen protocol
#: block at import sites that carry the contract, recorded on every decision.
#: 3.6 is the P5 EXIT_2 relabel revision: text-only in the uncertainty
#: entries; no router predicate, boundary or status changed.
CONTRACT_VERSION = "3.6"

#: Frozen envelope constants. Each cites the record it is read from; none is
#: chosen here.
DECODER_CATEGORY_CEILING = CMAX  # frozen feature space: categories 2..7
LONGITUDINAL_CATEGORY_BAND = (4, 7)  # 5E family table category choices
LONGITUDINAL_CONSTRUCT_MAX = 2  # 5E families: 1- and 2-construct designs only
LONGITUDINAL_TRAINED_DENSITY_FLOOR = 0.7  # 5E e1_mcar_items nominal rate
#: A design is MATERIALLY outside the density envelope below floor / multiple,
#: reusing the frozen 6B.2.3 material-envelope convention (spacing uses the
#: same multiple above its band).
LONGITUDINAL_MATERIAL_DENSITY = LONGITUDINAL_TRAINED_DENSITY_FLOOR / MATERIAL_SPACING_MULTIPLE

#: Largest number of supplied factors the frozen backbone expresses (M8B).
SUPPLIED_STRUCTURE_K_MAX = 5

#: New warning identifiers for the wired envelope predicates. Strings are the
#: contract, following the 6B.2.3 convention.
CATEGORY_ENVELOPE_WARNING = "longitudinal_response_categories_outside_trained_band"
CONSTRUCT_ENVELOPE_WARNING = "longitudinal_construct_count_outside_trained_envelope"
DENSITY_ENVELOPE_WARNING = "longitudinal_observed_fraction_materially_below_trained_envelope"


class SupportAction(StrEnum):
    """Output-specific actions. Deliberately NOT members of ``interface.Status``
    (that vocabulary is frozen at five values); these say what happened to ONE
    requested output, in the same sense the 6B.2.3 capability statuses say what
    an output means for one design."""

    PERMIT = "permit"
    LIMIT = "limit"
    FALLBACK = "fallback"
    REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class PredicateResult:
    """One executed predicate: what was compared with what, and the outcome."""

    name: str
    passed: bool
    value: Any
    boundary: Any
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "boundary": self.boundary,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class OutputDecision:
    """The audit record for one requested output.

    A single run can be inspected to determine what was requested, which
    descriptor values were derived, which predicates were evaluated, why each
    output was permitted/refused/limited, and which contract governed it.
    """

    output_type: str
    action: SupportAction
    predicates_evaluated: tuple[str, ...]
    predicate_results: tuple[PredicateResult, ...]
    warnings: tuple[str, ...] = ()
    fallback_used: bool = False
    refusal_code: str | None = None
    evidence_refs: tuple[str, ...] = ()
    contract_version: str = CONTRACT_VERSION
    enforcement_state: str = "enforced_runtime"
    router_version: str = ROUTER_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_type": self.output_type,
            "action": self.action.value,
            "predicates_evaluated": list(self.predicates_evaluated),
            "predicate_results": [p.as_dict() for p in self.predicate_results],
            "warnings": list(self.warnings),
            "fallback_used": self.fallback_used,
            "refusal_code": self.refusal_code,
            "evidence_refs": list(self.evidence_refs),
            "contract_version": self.contract_version,
            "enforcement_state": self.enforcement_state,
            "router_version": self.router_version,
        }


@dataclass(frozen=True, slots=True)
class DesignDescriptor:
    """The canonical runtime target-design descriptor.

    Every field is derived from the delivered tensors and the caller's
    explicit declarations - never from exploratory structure discovery.
    ``latent_dimensionality`` is the dimensionality of the REQUESTED scientific
    operation (1D scoring -> 1; supplied K-factor scoring -> K; a discovery
    request -> None, governed by the discovery contract), not an assertion that
    METER discovered the true dimensionality. ``factor_count`` is the supplied
    item-to-factor map's K where a structure was supplied, else None.
    ``n_constructs`` is the observable construct structure of the delivered
    data: the number of distinct declared instruments.
    """

    n_participants: int
    n_waves: int
    n_items: int
    response_category_counts: tuple[int, ...]
    observed_cell_fraction: float
    n_constructs: int
    latent_dimensionality: int | None
    factor_count: int | None
    supplied_structure_present: bool
    longitudinal_design: bool
    auxiliary_modality_present: bool
    requested_output_types: tuple[str, ...]
    data_provenance: str = "undeclared"
    descriptor_version: str = ROUTER_VERSION

    def max_categories(self) -> int:
        return max(self.response_category_counts) if self.response_category_counts else 0

    def min_categories(self) -> int:
        return min(self.response_category_counts) if self.response_category_counts else 0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def derive_descriptor(request: Any, requested_outputs: tuple[str, ...]) -> DesignDescriptor:
    """Derive the canonical descriptor from an ``InferenceRequest`` at runtime.

    Category counts come from the explicit per-item metadata the container
    validates (``n_categories``); observedness comes exclusively from the mask,
    so missing sentinels (-1), structural missingness and masked cells are
    excluded by construction; construct structure comes from the declared
    instrument partition (``item_instrument``). Nothing is inferred by fitting.
    """
    import torch

    tensors = request.tensors
    p, t, j = tensors.responses.shape
    categories = tuple(int(c) for c in tensors.n_categories.tolist())
    observed_fraction = float(tensors.mask.to(dtype=torch.float64).mean())
    n_constructs = len(set(int(i) for i in tensors.item_instrument.tolist()))
    longitudinal = int(t) >= 2
    multidim_requested = bool(getattr(request, "request_multidimensional", False))
    return DesignDescriptor(
        n_participants=int(p),
        n_waves=int(t),
        n_items=int(j),
        response_category_counts=categories,
        observed_cell_fraction=observed_fraction,
        n_constructs=n_constructs,
        # The core interface offers no supplied-structure input; a
        # multidimensional request is the M4 joint diagnostic operation, whose
        # target dimensionality is not declared by the caller - recorded as
        # None (a diagnostic/discovery-flavoured request), never guessed.
        latent_dimensionality=None if multidim_requested else 1,
        factor_count=None,
        supplied_structure_present=False,
        longitudinal_design=longitudinal,
        auxiliary_modality_present=request.continuous is not None,
        requested_output_types=tuple(requested_outputs),
        data_provenance=str(getattr(request, "data_provenance", "undeclared")),
    )


def validate_supplied_structure(
    n_items: int,
    item_factor: tuple[int, ...],
    requested_k: int | None,
) -> int:
    """Derive and consistency-check K from a supplied item-to-factor map.

    Refuses (raises) rather than guessing when the supplied pieces disagree:
    item count vs map length, active factor indices vs implied K, and a
    separately requested K vs the map's K.
    """
    if len(item_factor) != n_items:
        raise ValueError(
            f"supplied structure maps {len(item_factor)} items; the data carry {n_items}"
        )
    if min(item_factor) < 0:
        raise ValueError("supplied structure factor indices must be 0-based and non-negative")
    k = max(item_factor) + 1
    active = len(set(item_factor))
    if active != k:
        raise ValueError(
            f"supplied structure implies K={k} but only {active} factor columns are active; "
            "every factor must claim at least one item"
        )
    if requested_k is not None and int(requested_k) != k:
        raise ValueError(
            f"requested K={int(requested_k)} disagrees with the supplied map's K={k}; "
            "refusing to guess which was intended"
        )
    return int(k)


# ---------------------------------------------------------------------------
# executable predicates
# ---------------------------------------------------------------------------


def decoder_category_predicate(descriptor: DesignDescriptor) -> PredicateResult:
    """Categories beyond CMAX are architecturally inexpressible - every path."""
    value = descriptor.max_categories()
    return PredicateResult(
        name="response_categories_within_decoder_support",
        passed=value <= DECODER_CATEGORY_CEILING,
        value=value,
        boundary=DECODER_CATEGORY_CEILING,
        evidence="frozen feature space CMAX=7 (model.features); release MAX_CATEGORIES=7",
    )


def longitudinal_category_band_predicate(descriptor: DesignDescriptor) -> PredicateResult:
    low, high = LONGITUDINAL_CATEGORY_BAND
    binary = descriptor.max_categories() <= BINARY_CATEGORY_COUNT
    inside = (
        not binary and descriptor.min_categories() >= low and descriptor.max_categories() <= high
    )
    return PredicateResult(
        name="longitudinal_response_categories_within_trained_band",
        passed=inside,
        value=[descriptor.min_categories(), descriptor.max_categories()],
        boundary=list(LONGITUDINAL_CATEGORY_BAND),
        evidence=(
            "5E family table draws categories from {(4,), (5,), (5,7)} "
            "(model.longitudinal_worlds); binary carries the 6B.2.2 measured bias (+0.378)"
        ),
    )


def longitudinal_construct_predicate(descriptor: DesignDescriptor) -> PredicateResult:
    return PredicateResult(
        name="longitudinal_construct_count_within_trained_envelope",
        passed=descriptor.n_constructs <= LONGITUDINAL_CONSTRUCT_MAX,
        value=descriptor.n_constructs,
        boundary=LONGITUDINAL_CONSTRUCT_MAX,
        evidence=(
            "5E trained families contain 1- and 2-construct designs only "
            "(b5_two_construct); multidimensional longitudinal inference is not validated (L36)"
        ),
    )


def longitudinal_density_predicate(descriptor: DesignDescriptor) -> PredicateResult:
    return PredicateResult(
        name="longitudinal_observed_fraction_not_materially_below_trained_envelope",
        passed=descriptor.observed_cell_fraction >= LONGITUDINAL_MATERIAL_DENSITY,
        value=round(descriptor.observed_cell_fraction, 8),
        boundary=round(LONGITUDINAL_MATERIAL_DENSITY, 8),
        evidence=(
            "trained nominal response-rate floor 0.7 (5E e1_mcar_items; phase3 floor 0.75) "
            "with the frozen 6B.2.3 MATERIAL multiple 3.0 applied on the density axis"
        ),
    )


@dataclass(frozen=True, slots=True)
class EnvelopeEvaluation:
    """The wired longitudinal envelope predicates for one descriptor."""

    predicates: tuple[PredicateResult, ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def out_of_envelope(self) -> bool:
        return bool(self.reasons)


def evaluate_longitudinal_envelope(descriptor: DesignDescriptor) -> EnvelopeEvaluation:
    """Run the three newly wired envelope predicates for a longitudinal design.

    These COMPOSE with the frozen 6B.2.3 routing (binary items, spacing,
    metadata); they do not replace it. Binary designs are already routed out of
    support by the frozen predicate, so the category-band predicate adds a
    reason only for non-binary out-of-band designs.
    """
    category = longitudinal_category_band_predicate(descriptor)
    construct = longitudinal_construct_predicate(descriptor)
    density = longitudinal_density_predicate(descriptor)
    reasons: list[str] = []
    warnings: list[str] = []
    binary = descriptor.max_categories() <= BINARY_CATEGORY_COUNT
    if not category.passed and not binary:
        reasons.append(
            f"response categories {category.value} outside the 5E trained band "
            f"[{LONGITUDINAL_CATEGORY_BAND[0]}, {LONGITUDINAL_CATEGORY_BAND[1]}]"
        )
        warnings.append(CATEGORY_ENVELOPE_WARNING)
    if not construct.passed:
        reasons.append(
            f"{descriptor.n_constructs} declared constructs exceed the 1-2-construct "
            "trained longitudinal envelope (L36-adjacent; not validated)"
        )
        warnings.append(CONSTRUCT_ENVELOPE_WARNING)
    if not density.passed:
        reasons.append(
            f"observed cell fraction {density.value} is materially below the trained "
            f"envelope (floor {LONGITUDINAL_TRAINED_DENSITY_FLOOR} / material multiple "
            f"{MATERIAL_SPACING_MULTIPLE})"
        )
        warnings.append(DENSITY_ENVELOPE_WARNING)
    return EnvelopeEvaluation(
        predicates=(category, construct, density),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def merge_envelope_into_routing(
    routing: DesignRouting, envelope: EnvelopeEvaluation
) -> DesignRouting:
    """Compose the frozen 6B.2.3 routing with the wired envelope predicates.

    When the envelope adds nothing, the frozen routing is returned UNCHANGED
    (object identity), which is what keeps in-envelope behaviour bit-identical.
    When it does, the design is out of validated support: the same
    status_out_of_support values the frozen contracts declare are applied -
    no new status is invented.
    """
    if not envelope.out_of_envelope:
        return routing
    statuses = {
        capability: str(contract["status_out_of_support"])
        for capability, contract in LONGITUDINAL_CONTRACTS.items()
    }
    warnings = [*routing.warnings, *envelope.warnings]
    if OUT_OF_SUPPORT_WARNING not in warnings:
        warnings.append(OUT_OF_SUPPORT_WARNING)
    for capability_warning in (
        TRAIT_SHARE_BIAS_WARNING,
        CHANGE_ATTENUATION_WARNING,
        TRAIT_ABSORBS_STATE_WARNING,
    ):
        if capability_warning not in warnings:
            warnings.append(capability_warning)
    return DesignRouting(
        design=routing.design,
        out_of_support=True,
        reasons=(*routing.reasons, *envelope.reasons),
        warnings=tuple(dict.fromkeys(warnings)),
        capability_statuses=statuses,
        metadata_incomplete=routing.metadata_incomplete,
    )


# ---------------------------------------------------------------------------
# output-specific decisions
# ---------------------------------------------------------------------------

#: Outputs whose numeric value is withheld outright when the design is out of
#: validated support. ``trait_share`` was already suppressed by the frozen
#: boundary layer; the binary change decision carries the contract status
#: ``not_supported_for_primary_inference``, and serializing the flag anyway
#: while attaching that status is the warn-but-serialize pattern this router
#: exists to close.
OUT_OF_SUPPORT_REFUSED_OUTPUTS = {
    "trait_share": "longitudinal_trait_share",
    "trait_variance_share": "longitudinal_trait_share",
    "variance_decomposition": "longitudinal_trait_share",
    "change_flag": "binary_change_detection",
    "threshold": "binary_change_detection",
}

#: Outputs delivered under a downgraded (limited) contract when out of support.
OUT_OF_SUPPORT_LIMITED_OUTPUTS = {
    "trait": "longitudinal_trait",
    "state": "longitudinal_state",
    "level": "longitudinal_trait",
    "level_sd": "longitudinal_trait",
    "change_score": "continuous_change",
    "continuous_change_rank_score": "continuous_change",
    "detection_wave": "continuous_change",
    "direction": "continuous_change",
    "ambiguity": "continuous_change",
}


def decide_longitudinal_outputs(
    descriptor: DesignDescriptor,
    routing: DesignRouting,
    envelope: EnvelopeEvaluation,
    delivered: tuple[str, ...],
    suppressed: tuple[str, ...],
) -> tuple[OutputDecision, ...]:
    """One decision per delivered or withheld output on the longitudinal path."""
    base_predicates: tuple[PredicateResult, ...] = (
        decoder_category_predicate(descriptor),
        *envelope.predicates,
        PredicateResult(
            name="frozen_6b23_routing",
            passed=not routing.out_of_support,
            value=list(routing.reasons),
            boundary="binary items / spacing >= 3x training band / metadata complete",
            evidence="route_longitudinal_design (Milestone 6B.2.3, unchanged)",
        ),
    )
    names = tuple(p.name for p in base_predicates)
    out = routing.out_of_support
    decisions: list[OutputDecision] = []
    for name in delivered:
        contract_key = OUT_OF_SUPPORT_LIMITED_OUTPUTS.get(name)
        limited = out and contract_key is not None
        decisions.append(
            OutputDecision(
                output_type=name,
                action=SupportAction.LIMIT if limited else SupportAction.PERMIT,
                predicates_evaluated=names,
                predicate_results=base_predicates,
                warnings=routing.warnings if limited else (),
                evidence_refs=(
                    (f"LONGITUDINAL_CONTRACTS[{contract_key!r}]",) if contract_key else ()
                ),
            )
        )
    for name in suppressed:
        contract_key = OUT_OF_SUPPORT_REFUSED_OUTPUTS.get(name)
        decisions.append(
            OutputDecision(
                output_type=name,
                action=SupportAction.REFUSE,
                predicates_evaluated=names,
                predicate_results=base_predicates,
                warnings=routing.warnings,
                refusal_code=(
                    f"out_of_support:{contract_key}" if contract_key else "out_of_support"
                ),
                evidence_refs=(
                    (f"LONGITUDINAL_CONTRACTS[{contract_key!r}]",) if contract_key else ()
                ),
            )
        )
    return tuple(decisions)


def decide_fusion_outputs(
    descriptor: DesignDescriptor,
    routing: DesignRouting | None,
    envelope: EnvelopeEvaluation | None,
    fallback: bool,
    delivered: tuple[str, ...],
) -> tuple[OutputDecision, ...]:
    """Decisions for the multimodal path: the admission gate composes with the
    longitudinal envelope instead of bypassing it."""
    predicates: list[PredicateResult] = [
        decoder_category_predicate(descriptor),
        PredicateResult(
            name="auxiliary_admission_gate",
            passed=not fallback,
            value="rejected" if fallback else "admitted",
            boundary="reliability gate > GATE_HARD_REJECT (0.25), frozen",
            evidence="multimodal_arms.GATE_HARD_REJECT; exact psychometric fallback (M10A/M10B)",
        ),
    ]
    warnings: tuple[str, ...] = ()
    out_of_support = False
    if envelope is not None and routing is not None:
        predicates.extend(envelope.predicates)
        predicates.append(
            PredicateResult(
                name="frozen_6b23_routing",
                passed=not routing.out_of_support,
                value=list(routing.reasons),
                boundary="binary items / spacing >= 3x training band / metadata complete",
                evidence="route_longitudinal_design applied compositionally (P4 bypass closed)",
            )
        )
        out_of_support = routing.out_of_support
        warnings = routing.warnings
    names = tuple(p.name for p in predicates)
    fused = {"fused_mean", "fused_precision", "effective_continuous_precision"}
    decisions: list[OutputDecision] = []
    for name in delivered:
        if fallback and name in fused:
            action = SupportAction.FALLBACK
        elif out_of_support:
            action = SupportAction.LIMIT
        else:
            action = SupportAction.PERMIT
        decisions.append(
            OutputDecision(
                output_type=name,
                action=action,
                predicates_evaluated=names,
                predicate_results=tuple(predicates),
                warnings=warnings if action is not SupportAction.PERMIT else (),
                fallback_used=fallback and name in fused,
                refusal_code=None,
                evidence_refs=("exact_psychometric_fallback (contract v3.5)",)
                if fallback and name in fused
                else (),
            )
        )
    return tuple(decisions)


def decide_refused_request(
    descriptor: DesignDescriptor,
    withheld: tuple[str, ...],
    refusal_code: str,
    predicate: PredicateResult,
) -> tuple[OutputDecision, ...]:
    """Every withheld output of a request refused before or at routing."""
    return tuple(
        OutputDecision(
            output_type=name,
            action=SupportAction.REFUSE,
            predicates_evaluated=(predicate.name,),
            predicate_results=(predicate,),
            refusal_code=refusal_code,
        )
        for name in withheld
    )


# ---------------------------------------------------------------------------
# supplied-structure decisions (the meter.score layer)
# ---------------------------------------------------------------------------


def supplied_structure_decisions(
    descriptor: DesignDescriptor,
) -> dict[str, OutputDecision]:
    """Output-specific decisions for a supplied-structure (K >= 2) request.

    The same input legitimately receives different decisions for different
    requested outputs: factor scores may be permitted while Phi, unrestricted
    structure discovery and factor-count discovery are refused.
    """
    k = descriptor.factor_count
    dense = descriptor.observed_cell_fraction >= 1.0
    k_pred = PredicateResult(
        name="supplied_k_within_frozen_backbone",
        passed=k is not None and 1 <= k <= SUPPLIED_STRUCTURE_K_MAX,
        value=k,
        boundary=SUPPLIED_STRUCTURE_K_MAX,
        evidence="K_MAX=5 (M8B confirmatory benchmark; release contract)",
    )
    dense_pred = PredicateResult(
        name="dense_observation_required_for_multidimensional_scoring",
        passed=dense,
        value=round(descriptor.observed_cell_fraction, 8),
        boundary=1.0,
        evidence="M8 closeout: sparse multidimensional administration UNSUPPORTED",
    )
    ceiling = decoder_category_predicate(descriptor)
    structure_pred = PredicateResult(
        name="structure_is_supplied_not_discovered",
        passed=descriptor.supplied_structure_present,
        value=descriptor.supplied_structure_present,
        boundary=True,
        evidence="M7 closure: unrestricted 1-5-factor discovery experimental_out_of_support",
    )
    longitudinal_pred = PredicateResult(
        name="multidimensional_request_is_cross_sectional",
        passed=not descriptor.longitudinal_design,
        value=descriptor.longitudinal_design,
        boundary=False,
        evidence="L36: multidimensional longitudinal inference is not validated",
    )
    # Order matters: it is the precedence of the release contract's gates
    # (category ceiling, structure supplied, K bound, longitudinal, density).
    score_predicates = (ceiling, structure_pred, k_pred, longitudinal_pred, dense_pred)
    names = tuple(p.name for p in score_predicates)
    all_pass = all(p.passed for p in score_predicates)
    k2_limited = all_pass and k == 2

    if not all_pass:
        failed = next(p for p in score_predicates if not p.passed)
        scores = OutputDecision(
            output_type="factor_scores",
            action=SupportAction.REFUSE,
            predicates_evaluated=names,
            predicate_results=score_predicates,
            refusal_code=f"predicate_failed:{failed.name}",
        )
    else:
        scores = OutputDecision(
            output_type="factor_scores",
            action=SupportAction.LIMIT if k2_limited else SupportAction.PERMIT,
            predicates_evaluated=names,
            predicate_results=score_predicates,
            warnings=("meter:k2_outside_confirmatory_range",) if k2_limited else (),
            evidence_refs=("M8B_PASS", "D9E_PASS", "D9B score gates"),
        )

    synthetic = descriptor.data_provenance == "synthetic"
    phi_pred = PredicateResult(
        name="phi_interpretation_within_validated_regime",
        passed=all_pass and synthetic,
        value=descriptor.data_provenance,
        boundary="synthetic (dense, K in range)",
        evidence="D9B locked replication failed both Phi gates (REPLICATION_FAIL); "
        "M8: Phi within the validated dense synthetic regime only",
    )
    phi = OutputDecision(
        output_type="factor_correlation_phi",
        action=SupportAction.LIMIT if phi_pred.passed else SupportAction.REFUSE,
        predicates_evaluated=(*names, phi_pred.name),
        predicate_results=(*score_predicates, phi_pred),
        warnings=("diagnostic_only_never_a_finding",) if phi_pred.passed else (),
        refusal_code=None if phi_pred.passed else "real_data_phi_not_supported:D9B",
    )

    discovery_pred = PredicateResult(
        name="unrestricted_structure_discovery_supported",
        passed=False,
        value="requested" if not descriptor.supplied_structure_present else "not_requested",
        boundary="never (M7 closure)",
        evidence="battery_structure_discovery_1_to_5 = experimental_out_of_support",
    )
    discovery = OutputDecision(
        output_type="item_to_factor_structure",
        action=SupportAction.REFUSE,
        predicates_evaluated=(discovery_pred.name,),
        predicate_results=(discovery_pred,),
        refusal_code="structure_discovery_out_of_support:M7",
    )
    count_discovery = OutputDecision(
        output_type="factor_count_discovery",
        action=SupportAction.REFUSE,
        predicates_evaluated=(discovery_pred.name,),
        predicate_results=(discovery_pred,),
        refusal_code="factor_count_discovery_out_of_support:M7",
    )
    return {
        "factor_scores": scores,
        "factor_correlation_phi": phi,
        "item_to_factor_structure": discovery,
        "factor_count_discovery": count_discovery,
    }
