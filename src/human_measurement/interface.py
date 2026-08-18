"""Milestone 6A: the integrated inference interface (ADR 2k).

One versioned entry point over every accepted and experimental capability. No
model is trained here, no threshold recalibrated, no frozen result changed:
this module routes to the frozen capabilities and reports what answered, under
what status, with what provenance.

The load-bearing rule is **suppression**. An ``unsupported`` result carries no
numeric estimate at all - the field is absent from ``outputs`` and a
``suppressed`` entry names what was withheld and why - because a number that
looks ordinary is exactly what gets mistaken for a validated one. Integration
never promotes: each capability's status here equals its status in its own
frozen provenance block, asserted by test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

INPUT_SCHEMA_VERSION = "6a-input-v1"
OUTPUT_SCHEMA_VERSION = "6a-output-v1"
INTERFACE_VERSION = "6a-v1"


class Status(StrEnum):
    """The five permitted output statuses (ADR 2k). No sixth value exists."""

    SUPPORTED = "supported"
    SUPPORTED_EXPERIMENTAL = "supported_experimental"
    CONDITIONALLY_SUPPORTED_EXPERIMENTAL = "conditionally_supported_experimental"
    INDETERMINATE = "indeterminate"
    UNSUPPORTED = "unsupported"


#: The ten integrated capabilities and the frozen block each one's status is
#: pinned to. Statuses are copied from those blocks, never chosen here.
CAPABILITIES: dict[str, dict[str, Any]] = {
    "psychometric_1d": {
        "status": Status.SUPPORTED,
        "milestone": "M3",
        "outputs": ("person_score", "person_sd", "item_parameters"),
    },
    "multidimensional": {
        "status": Status.SUPPORTED,
        "milestone": "M4",
        "outputs": ("person_score", "person_sd", "dimensionality"),
    },
    "item_assignment": {
        "status": Status.SUPPORTED,
        "milestone": "M4",
        "outputs": ("item_assignment", "dimensionality"),
    },
    "uncertainty_calibration": {
        "status": Status.SUPPORTED,
        "milestone": "M3C",
        "outputs": ("person_sd", "coverage"),
        "note": (
            "the frozen scalar calibration is validated only for its own 1-D family and "
            "is never applied outside it"
        ),
    },
    "dif_comparability": {
        "status": Status.SUPPORTED,
        "milestone": "M5A",
        "frozen_block": "SELECTED_MODEL_5A",
        "outputs": ("dif_flags", "comparability_status", "adjusted_scores"),
    },
    "instrument_linking": {
        "status": Status.SUPPORTED,
        "milestone": "M5D",
        "frozen_block": "SELECTED_MODEL_5D",
        "outputs": ("linking_status", "transform", "bridge_diagnostics"),
    },
    "longitudinal_trait_state": {
        "status": Status.SUPPORTED_EXPERIMENTAL,
        "milestone": "M5E",
        "frozen_block": "EXPERIMENTAL_MODEL_5E",
        "outputs": ("trait", "state", "level", "level_sd"),
    },
    "continuous_change_score": {
        "status": Status.SUPPORTED_EXPERIMENTAL,
        "milestone": "M5E.1",
        "frozen_block": "SELECTED_DETECTOR_5E1",
        "outputs": ("change_score", "detection_wave", "direction"),
        "note": "a ranking score, never a calibrated probability",
    },
    "binary_change_detection": {
        "status": Status.CONDITIONALLY_SUPPORTED_EXPERIMENTAL,
        "milestone": "M5E.1",
        "frozen_block": "SELECTED_DETECTOR_5E1",
        "outputs": ("change_flag", "threshold"),
    },
    "multimodal_fusion": {
        "status": Status.SUPPORTED_EXPERIMENTAL,
        "milestone": "M5F",
        "frozen_block": "CAPABILITY_5F",
        "outputs": ("fused_mean", "fused_precision", "reliability_gate", "modality_contribution"),
        "note": (
            "experimental_not_adopted; scoped to one continuous modality measuring the "
            "same one-dimensional construct"
        ),
    },
}

#: Input conditions that have no supported answer, with the limitation each
#: one comes from. Routing consults this before choosing a capability.
UNSUPPORTED_COMBINATIONS: dict[str, dict[str, str]] = {
    "multidimensional_longitudinal": {
        "reason": "multidimensional longitudinal inference degrades and is not validated",
        "limitation": "L36",
    },
    "related_but_distinct_multimodal": {
        "reason": (
            "fusion of a related but distinct continuous construct is outside the "
            "validated same-construct scope"
        ),
        "limitation": "L42",
    },
    "multiple_continuous_modalities": {
        "reason": "only one continuous modality is within the validated scope",
        "limitation": "CAPABILITY_5F.unsupported",
    },
    "single_wave_change_detection": {
        "reason": "change detection requires at least two observed waves",
        "limitation": "L22",
    },
    # --- P4B capability-router close-out ----------------------------------
    "response_categories_exceed_decoder_support": {
        "reason": (
            "the frozen decoder's feature space expresses items with 2..7 response categories "
            "(CMAX = 7, the largest response scale in the corpus). An item beyond that is "
            "INEXPRESSIBLE, not merely unvalidated: the category histogram and threshold heads "
            "are sized to CMAX, so a request carrying more categories would be silently "
            "truncated into a different instrument than the one supplied"
        ),
        "limitation": "frozen decoder feature space (model.features.CMAX); release MAX_CATEGORIES",
    },
    # --- Milestone 6B.2.1 -------------------------------------------------
    "multigroup_dif": {
        "reason": (
            "the frozen DIF head is architecturally two-group: group_code_from_settings "
            "reduces every setting to (setting > 0), so a k-group design silently becomes one "
            "group against the other k-1 pooled. Decompose into prespecified two-group "
            "contrasts with an explicit DifContrast"
        ),
        "limitation": "L46",
    },
    "dif_without_declared_contrast": {
        "reason": (
            "a two-group DIF request must declare which group is the reference and which the "
            "focal. Without it the assignment falls out of positional setting order, so the "
            "sign of every effect - and therefore its direction - is an accident of how the "
            "data happened to be sorted"
        ),
        "limitation": "L46",
    },
}


@dataclass(frozen=True, slots=True)
class DifContrast:
    """An explicitly declared two-group DIF contrast (Milestone 6B.2.1).

    Required for every DIF request. The frozen head reads group membership as
    ``setting > 0``, which means the reference/focal assignment is decided by
    positional setting order unless the caller says otherwise. That is fine
    when there are exactly two settings and the caller knows which is which,
    and silently wrong the moment either assumption slips - the effect sign
    flips and the direction of every finding reverses.

    Naming both groups makes the contrast auditable after the fact, which the
    ESS Round 7 run needed: 20 contrasts against one reference are only
    interpretable if each states its reference.
    """

    reference: str
    focal: str

    def __post_init__(self) -> None:
        if not self.reference or not self.focal:
            raise ValueError("a DIF contrast must name both a reference and a focal group")
        if self.reference == self.focal:
            raise ValueError(
                f"reference and focal group are both {self.reference!r}; a contrast between a "
                "group and itself is not a contrast"
            )

    def label(self) -> str:
        return f"{self.reference}_vs_{self.focal}"


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """What the caller declares. Routing reads shape and declarations only.

    ``continuous_construct`` is the caller's *declaration* about what the
    continuous block measures, not something inferred from the data: the
    interface will not guess, and an undeclared or non-same construct routes to
    an unsupported result rather than a quietly fused one.
    """

    tensors: Any
    schema_version: str = INPUT_SCHEMA_VERSION
    continuous: Any | None = None
    continuous_construct: str | None = None  # "same" | "related_distinct" | None
    n_continuous_modalities: int = 0
    second_instrument: Any | None = None
    request_multidimensional: bool = False
    request_change_detection: bool = False
    request_dif: bool = False
    request_item_assignment: bool = False
    request_continuous_change_score_only: bool = False

    #: Milestone 6B.2.1. Required whenever ``request_dif`` is set.
    dif_contrast: DifContrast | None = None
    #: Milestone 6B.2.3. The observable longitudinal design descriptor. When a
    #: longitudinal request supplies one, routing enforces the operating
    #: boundaries measured in 6B.2.2; when it does not, the conservative route
    #: is taken and a metadata-incomplete warning is emitted.
    longitudinal_design: Any | None = None
    #: Milestone 6B.2.1. Governs whether the real-world interval-scale warning
    #: is attached to outputs carrying an uncertainty quantity.
    #:
    #: Fail-safe by design: the warning is emitted for ``"real"`` AND for
    #: ``"undeclared"``. Suppressing it requires positively declaring the data
    #: synthetic. A caller who forgets gets the warning; only a caller who
    #: asserts the calibrated case goes without it, and on synthetic data that
    #: assertion is true - the Phase 3C scalar calibration is validated there.
    data_provenance: str = "undeclared"  # "synthetic" | "real" | "undeclared"

    def n_waves(self) -> int:
        return int(self.tensors.responses.shape[1])

    def n_settings(self) -> int:
        """Distinct declared settings. The group count a DIF request implies."""
        settings = getattr(self.tensors, "setting_of_person", None)
        if settings is None:
            return 1
        try:
            return len(set(int(s) for s in settings.reshape(-1).tolist()))
        except (AttributeError, TypeError, ValueError):
            return 1


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """A versioned, self-describing result.

    ``outputs`` contains only values the status actually licenses. Anything
    withheld appears in ``suppressed`` with its reason - never as a number.
    """

    capability: str
    status: Status
    route: tuple[str, ...]
    outputs: dict[str, Any] = field(default_factory=dict)
    suppressed: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Milestone 6B.2.3. What the output MEANS for this particular design, from
    #: `longitudinal_design.CAPABILITY_STATUSES`. Distinct from ``status``,
    #: which says how the request was ROUTED and remains one of the five frozen
    #: values. A design can route to a supported capability and still carry
    #: `not_interpretable_for_design`, which is exactly the HRS trait-share
    #: case, and one field cannot express both without losing one of them.
    capability_status: str | None = None
    #: The observable design descriptor and its routing decision, when the
    #: request was longitudinal. Carried on the result so an artifact can be
    #: audited later without re-deriving what the design was.
    design: dict[str, Any] | None = None
    #: P4B: one audit record per requested output (capability_router.
    #: OutputDecision). Records which predicates were actually evaluated and
    #: why each output was permitted, limited, fallback-routed or refused.
    #: Additive: absent entries mean the request predates the wired router.
    output_decisions: tuple[Any, ...] = ()
    input_schema_version: str = INPUT_SCHEMA_VERSION
    output_schema_version: str = OUTPUT_SCHEMA_VERSION
    interface_version: str = INTERFACE_VERSION

    def is_usable(self) -> bool:
        return self.status not in (Status.UNSUPPORTED, Status.INDETERMINATE)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable form; every flag survives serialization."""
        return {
            "capability": self.capability,
            "status": self.status.value,
            "route": list(self.route),
            "outputs": {k: _plain(v) for k, v in self.outputs.items()},
            "suppressed": dict(self.suppressed),
            "warnings": list(self.warnings),
            "output_decisions": [
                decision.as_dict() if hasattr(decision, "as_dict") else decision
                for decision in self.output_decisions
            ],
            "provenance": self.provenance,
            "input_schema_version": self.input_schema_version,
            "output_schema_version": self.output_schema_version,
            "interface_version": self.interface_version,
        }


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().numpy().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def route(request: InferenceRequest) -> tuple[str, tuple[str, ...], list[str]]:
    """Deterministic, total routing from declared shape and content.

    Returns (capability, route trace, warnings). Consults the unsupported
    table first: an input with no supported answer must never be routed to a
    capability that would produce an ordinary-looking number for it.
    """
    trace: list[str] = [f"waves={request.n_waves()}"]
    # P4B: the decoder ceiling is checked first because a request beyond it is
    # inexpressible on EVERY path - no capability can answer it faithfully.
    # Read from the declared per-item metadata, never inferred by fitting.
    categories = getattr(request.tensors, "n_categories", None)
    if categories is not None:
        try:
            max_categories = int(max(int(c) for c in categories.tolist()))
        except (AttributeError, TypeError, ValueError):
            max_categories = None
        if max_categories is not None:
            from human_measurement.model.features import CMAX

            if max_categories > CMAX:
                trace.append(f"max_response_categories={max_categories}>CMAX={CMAX}")
                return (
                    "unsupported:response_categories_exceed_decoder_support",
                    tuple(trace),
                    [],
                )
    if request.longitudinal_design is not None:
        # Part A: the descriptor is written into the route trace, so a result's
        # provenance shows what design it was routed for, not merely which
        # capability answered.
        trace.extend(request.longitudinal_design.trace_entries())
    warnings: list[str] = []
    longitudinal = request.n_waves() >= 2

    if request.n_continuous_modalities > 1:
        trace.append("multiple_continuous_modalities")
        return "unsupported:multiple_continuous_modalities", tuple(trace), warnings
    if request.request_multidimensional and longitudinal:
        trace.append("multidimensional+longitudinal")
        return "unsupported:multidimensional_longitudinal", tuple(trace), warnings
    if request.request_change_detection and request.n_waves() < 2:
        trace.append("change_detection_without_repeated_waves")
        return "unsupported:single_wave_change_detection", tuple(trace), warnings

    if request.continuous is not None:
        trace.append("continuous_block_present")
        if request.continuous_construct != "same":
            trace.append(f"declared_construct={request.continuous_construct}")
            return "unsupported:related_but_distinct_multimodal", tuple(trace), warnings
        if not bool(request.continuous.mask.any()):
            # Absent modality is not an error: it is the psychometric-only path,
            # and the fallback is named rather than silent.
            trace.append("continuous_block_empty")
            warnings.append("continuous_modality_absent_psychometric_only_path")
            return (
                ("longitudinal_trait_state" if longitudinal else "psychometric_1d"),
                tuple(trace),
                warnings,
            )
        return "multimodal_fusion", tuple(trace), warnings

    if request.second_instrument is not None:
        trace.append("second_instrument_declared")
        return "instrument_linking", tuple(trace), warnings
    if request.request_dif:
        trace.append("dif_requested")
        # Milestone 6B.2.1, gates 1 and 2. Both checks precede the capability
        # so a design the frozen head cannot express never reaches it. Before
        # this, a 21-country request routed here at status `supported` and
        # returned a confident contrast between setting 0 and everything else
        # (L46) - it did not error, which is what made it dangerous.
        n_settings = request.n_settings()
        if n_settings > 2:
            trace.append(f"settings={n_settings}")
            return "unsupported:multigroup_dif", tuple(trace), warnings
        if request.dif_contrast is None:
            trace.append("no_declared_contrast")
            return "unsupported:dif_without_declared_contrast", tuple(trace), warnings
        trace.append(f"contrast={request.dif_contrast.label()}")
        return "dif_comparability", tuple(trace), warnings
    if request.request_item_assignment:
        trace.append("item_assignment_requested")
        return "item_assignment", tuple(trace), warnings
    if request.request_continuous_change_score_only:
        trace.append("continuous_change_score_requested")
        return "continuous_change_score", tuple(trace), warnings
    if request.request_change_detection:
        trace.append("change_detection_requested")
        return "binary_change_detection", tuple(trace), warnings
    if longitudinal:
        return "longitudinal_trait_state", tuple(trace), warnings
    if request.request_multidimensional:
        trace.append("multidimensional_requested")
        return "multidimensional", tuple(trace), warnings
    return "psychometric_1d", tuple(trace), warnings


def unsupported_result(
    key: str, trace: tuple[str, ...], withheld: tuple[str, ...]
) -> InferenceResult:
    """An unsupported result: no estimate, only the reason it is absent."""
    record = UNSUPPORTED_COMBINATIONS[key]
    return InferenceResult(
        capability=f"unsupported:{key}",
        status=Status.UNSUPPORTED,
        route=trace,
        outputs={},
        suppressed={name: record["reason"] for name in withheld},
        warnings=(f"unsupported_input:{key}", f"see:{record['limitation']}"),
        provenance=frozen_provenance(),
    )


def frozen_provenance() -> dict[str, Any]:
    """Every hash and calibration record the integrated stack depends on."""
    # Paper-reproducibility trim: in the research repository these records
    # live inside the (unreleased) capability modules; the values are carried
    # verbatim in frozen_pins and asserted identical by the fixture tests.
    from human_measurement.frozen_pins import (
        CAPABILITY_5F,
        SELECTED_DETECTOR_5E1,
        SELECTED_MODEL_5A,
        SELECTED_MODEL_5D,
    )
    from human_measurement.model.longitudinal import EXPERIMENTAL_MODEL_5E

    root = Path(__file__).resolve().parents[2]

    def config_hash(name: str) -> str | None:
        # content_sha256, not sha256(read_bytes()): these hashes are reported
        # as provenance and compared against values recorded in the frozen
        # governance packages, so they must identify the configuration rather
        # than the line-ending convention of whichever checkout produced them.
        # This previously reached the same answer by accident, via read_text()
        # applying universal newlines.
        from human_measurement.frozen_artifacts import content_sha256

        path = root / "configs" / "model" / name
        if not path.exists():
            return None
        return content_sha256(path)

    return {
        "interface_version": INTERFACE_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "dif_model_sha256": SELECTED_MODEL_5A["model_state_sha256"],
        "dif_operating_threshold": SELECTED_MODEL_5A["operating_threshold"],
        "linking_model_sha256": SELECTED_MODEL_5D["model_state_sha256"],
        "linking_decision_config_sha256": SELECTED_MODEL_5D["decision_config_sha256"],
        "longitudinal_model_sha256": EXPERIMENTAL_MODEL_5E["model_state_sha256"],
        "longitudinal_status": EXPERIMENTAL_MODEL_5E["status"],
        "change_detector": SELECTED_DETECTOR_5E1["detector"],
        "change_detector_status": SELECTED_DETECTOR_5E1["status"],
        "change_score_is_calibrated_probability": SELECTED_DETECTOR_5E1[
            "score_is_calibrated_probability"
        ],
        "multimodal_status": CAPABILITY_5F["status"],
        "multimodal_wearable_validation_claim": CAPABILITY_5F["wearable_validation_claim"],
        "calibration_configs": {
            name: config_hash(name)
            for name in (
                "phase5e1_reference.json",
                "phase5d_decision.json",
                "phase5f_fusion.json",
                "phase5f1_selected.json",
            )
        },
    }


def capability_status(name: str) -> Status:
    return CAPABILITIES[name]["status"]


def describe() -> str:
    """Human-readable capability listing; statuses exactly as frozen."""
    lines = [f"integrated interface {INTERFACE_VERSION}"]
    for name, record in CAPABILITIES.items():
        note = f"  ({record['note']})" if "note" in record else ""
        lines.append(f"  {name:28s} {record['status'].value}{note}")
    return "\n".join(lines)


def contract_json() -> str:
    """The frozen output contract, for the regression matrix to hash."""
    return json.dumps(
        {
            "interface_version": INTERFACE_VERSION,
            "input_schema_version": INPUT_SCHEMA_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "statuses": [s.value for s in Status],
            "capabilities": {
                name: {
                    "status": record["status"].value,
                    "milestone": record["milestone"],
                    "outputs": list(record["outputs"]),
                }
                for name, record in CAPABILITIES.items()
            },
            "unsupported_combinations": UNSUPPORTED_COMBINATIONS,
        },
        sort_keys=True,
        indent=2,
    )
