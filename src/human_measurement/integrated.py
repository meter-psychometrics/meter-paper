"""Milestone 6A: execution behind the integrated interface (ADR 2k).

``infer`` routes a request and calls the *same frozen functions* the
capability-specific runners call, on the same inputs. There is therefore no
legitimate source of numerical drift between the two paths, which is what lets
the regression matrix demand exact equality rather than a tolerance.

Nothing here trains, recalibrates, or fits.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from human_measurement.interface import (
    CAPABILITIES,
    UNSUPPORTED_COMBINATIONS,
    InferenceRequest,
    InferenceResult,
    Status,
    frozen_provenance,
    route,
    unsupported_result,
)


def _psychometric(request: InferenceRequest, models: dict[str, Any]) -> dict[str, Any]:
    from human_measurement.model.multimodal import psychometric_message

    with torch.no_grad():
        forward = models["longitudinal"](request.tensors)
    mean, precision = psychometric_message(forward)
    return {
        "forward": forward,
        "person_score": mean,
        "person_sd": precision.clamp(min=1e-12).rsqrt(),
    }


#: Outputs whose VALUE is an uncertainty width. Milestone 6B.2.1 attaches the
#: interval-scale warning to any result carrying one of these.
UNCERTAINTY_OUTPUTS = ("person_sd", "level_sd", "fused_precision")

#: The warning string, taken from the capability contract rather than retyped,
#: so the two cannot drift apart.
INTERVAL_SCALE_WARNING = "real_world_interval_scale_not_calibrated"


def infer(request: InferenceRequest, models: dict[str, Any]) -> InferenceResult:
    """The single integrated entry point.

    ``models`` carries the already-hash-verified frozen artifacts, so this
    function never loads or mutates a checkpoint.

    Milestone 6B.2.1: any result carrying an uncertainty width leaves here with
    the interval-scale warning attached, unless the caller has positively
    declared the data synthetic. Before this the warning existed only in the
    contract and in prose - a downstream caller reading `person_sd` off a real
    dataset met nothing that said its scale was uncalibrated.
    """
    descriptor = _derive_descriptor(request)
    result = _execute(request, models)
    result = _apply_longitudinal_boundaries(request, result, descriptor)
    result = _apply_multimodal_composition(request, result, descriptor)
    result = _attach_output_decisions(request, result, descriptor)
    if request.data_provenance == "synthetic":
        return result
    if not any(name in result.outputs for name in UNCERTAINTY_OUTPUTS):
        return result
    if INTERVAL_SCALE_WARNING in result.warnings:
        return result
    return dataclasses.replace(result, warnings=(*result.warnings, INTERVAL_SCALE_WARNING))


def _derive_descriptor(request: InferenceRequest) -> Any:
    """The canonical runtime design descriptor (P4B). Read, never fitted."""
    from human_measurement.capability_router import derive_descriptor

    capability, _, _ = route(request)
    if capability.startswith("unsupported:"):
        requested: tuple[str, ...] = ()
    else:
        requested = tuple(CAPABILITIES[capability]["outputs"])
    return derive_descriptor(request, requested)


#: Capabilities whose outputs the longitudinal operating boundaries govern.
LONGITUDINAL_CAPABILITIES = (
    "longitudinal_trait_state",
    "binary_change_detection",
    "continuous_change_score",
)


def _apply_longitudinal_boundaries(
    request: InferenceRequest, result: InferenceResult, descriptor: Any | None = None
) -> InferenceResult:
    """Enforce the Milestone 6B.2.2 operating boundaries at the output boundary.

    Applies only to longitudinal capabilities. Suppresses what the design makes
    uninterpretable, renames what would otherwise read as a metric quantity,
    attaches the capability status and the warnings, and carries the design
    descriptor onto the result so an artifact stays auditable.

    Numbers that survive are unchanged: this reroutes and relabels, it does not
    recompute. That is what lets the regression pins demand exact equality.
    """
    from human_measurement.longitudinal_design import (
        LONGITUDINAL_CONTRACTS,
        route_longitudinal_design,
    )

    if result.capability not in LONGITUDINAL_CAPABILITIES:
        return result

    design = request.longitudinal_design
    if design is None:
        # Fail-safe: an undescribed longitudinal design is not a supported one.
        from human_measurement.longitudinal_design import LongitudinalDesign

        design = LongitudinalDesign(
            n_participants=int(request.tensors.responses.shape[0]),
            n_items=int(request.tensors.responses.shape[2]),
            item_response_type="unknown",
            n_waves=request.n_waves(),
        )
    routing = route_longitudinal_design(design)
    # P4B: the wired envelope predicates (trained category band, construct
    # count, material density) COMPOSE with the frozen 6B.2.3 routing. When
    # they add nothing, `routing` is returned unchanged - identical behaviour.
    if descriptor is not None:
        from human_measurement.capability_router import (
            evaluate_longitudinal_envelope,
            merge_envelope_into_routing,
        )

        routing = merge_envelope_into_routing(routing, evaluate_longitudinal_envelope(descriptor))

    outputs = dict(result.outputs)
    suppressed = dict(result.suppressed)
    contracts = LONGITUDINAL_CONTRACTS

    # Trait share is never emitted as a user-facing number on an
    # out-of-support design. It stays available to an internal diagnostic
    # caller, but not from here.
    share_contract = contracts["longitudinal_trait_share"]
    if routing.capability_statuses["longitudinal_trait_share"] == "not_interpretable_for_design":
        for name in ("trait_share", "trait_variance_share", "variance_decomposition"):
            if name in outputs:
                outputs.pop(name)
            suppressed.setdefault(name, share_contract["disclosure"])
        suppressed.setdefault("trait_share", share_contract["disclosure"])

    # A continuous change score that is ordinal-only must not carry a name that
    # reads as a magnitude. Renaming is the cheapest honest intervention: the
    # value is unchanged, the invitation to misread it is not.
    change_contract = contracts["continuous_change"]
    if routing.capability_statuses["continuous_change"] == "ordinal_only":
        for name in ("continuous_change_score", "change_score", "continuous_change"):
            if name in outputs:
                outputs[change_contract["output_rename"]] = outputs.pop(name)

    # P4B: on an out-of-support design the binary change decision carries the
    # contract status `not_supported_for_primary_inference`. The flag is
    # withheld under the suppression rule rather than serialized with that
    # status attached - a number that looks ordinary is what gets misread.
    if (
        routing.capability_statuses["binary_change_detection"]
        == "not_supported_for_primary_inference"
        and result.capability == "binary_change_detection"
    ):
        for name in ("change_flag", "threshold"):
            if name in outputs:
                outputs.pop(name)
                suppressed.setdefault(
                    name,
                    "not_supported_for_primary_inference on an out-of-support design "
                    "(L48, L50): sensitivity 0.041 at the frozen operating point",
                )

    capability_status = routing.capability_statuses.get(
        {
            "longitudinal_trait_state": "longitudinal_trait",
            "binary_change_detection": "binary_change_detection",
            "continuous_change_score": "continuous_change",
        }[result.capability]
    )

    return dataclasses.replace(
        result,
        outputs=outputs,
        suppressed=suppressed,
        capability_status=capability_status,
        design=routing.as_dict(),
        warnings=tuple(dict.fromkeys((*result.warnings, *routing.warnings))),
    )


def _apply_multimodal_composition(
    request: InferenceRequest, result: InferenceResult, descriptor: Any
) -> InferenceResult:
    """P4B: close the longitudinal x multimodal routing bypass.

    A requested output must satisfy ALL applicable support predicates. A
    fusion request over longitudinal tensors receives the same longitudinal
    design evaluation a longitudinal request receives; before this, boundary
    application was keyed to the answering capability, so the fused result
    escaped the longitudinal boundary entirely (measured in P4).

    Numbers are unchanged: like the 6B.2.3 layer, this reroutes and relabels.
    """
    from human_measurement.capability_router import (
        evaluate_longitudinal_envelope,
        merge_envelope_into_routing,
    )
    from human_measurement.longitudinal_design import (
        LongitudinalDesign,
        route_longitudinal_design,
    )

    if result.capability != "multimodal_fusion" or request.n_waves() < 2:
        return result

    design = request.longitudinal_design
    if design is None:
        design = LongitudinalDesign(
            n_participants=int(request.tensors.responses.shape[0]),
            n_items=int(request.tensors.responses.shape[2]),
            item_response_type="unknown",
            n_waves=request.n_waves(),
        )
    routing = merge_envelope_into_routing(
        route_longitudinal_design(design), evaluate_longitudinal_envelope(descriptor)
    )
    return dataclasses.replace(
        result,
        design=routing.as_dict(),
        capability_status="experimental_out_of_support" if routing.out_of_support else None,
        warnings=tuple(dict.fromkeys((*result.warnings, *routing.warnings))),
    )


def _attach_output_decisions(
    request: InferenceRequest, result: InferenceResult, descriptor: Any
) -> InferenceResult:
    """P4B: one audit record per output - what was evaluated, and why."""
    from human_measurement.capability_router import (
        OutputDecision,
        PredicateResult,
        SupportAction,
        decide_fusion_outputs,
        decide_longitudinal_outputs,
        decide_refused_request,
        decoder_category_predicate,
        evaluate_longitudinal_envelope,
    )
    from human_measurement.longitudinal_design import DesignRouting

    if result.capability.startswith("unsupported:"):
        key = result.capability.split(":", 1)[1]
        record = UNSUPPORTED_COMBINATIONS[key]
        predicate = PredicateResult(
            name=f"unsupported_combination:{key}",
            passed=False,
            value=key,
            boundary="declared unsupported combination",
            evidence=str(record["limitation"]),
        )
        decisions = decide_refused_request(
            descriptor, tuple(sorted(result.suppressed)), key, predicate
        )
    elif result.capability in LONGITUDINAL_CAPABILITIES:
        envelope = evaluate_longitudinal_envelope(descriptor)
        design_dict = result.design or {}
        routing = DesignRouting(
            design=None,  # decisions read only the fields below
            out_of_support=bool(design_dict.get("out_of_support")),
            reasons=tuple(design_dict.get("reasons", ())),
            warnings=tuple(design_dict.get("warnings", ())),
            capability_statuses=dict(design_dict.get("capability_statuses", {})),
            metadata_incomplete=bool(design_dict.get("metadata_incomplete")),
        )
        decisions = decide_longitudinal_outputs(
            descriptor,
            routing,
            envelope,
            tuple(sorted(result.outputs)),
            tuple(sorted(result.suppressed)),
        )
    elif result.capability == "multimodal_fusion":
        envelope = evaluate_longitudinal_envelope(descriptor) if request.n_waves() >= 2 else None
        design_dict = result.design
        routing = None
        if design_dict is not None:
            routing = DesignRouting(
                design=None,
                out_of_support=bool(design_dict.get("out_of_support")),
                reasons=tuple(design_dict.get("reasons", ())),
                warnings=tuple(design_dict.get("warnings", ())),
                capability_statuses=dict(design_dict.get("capability_statuses", {})),
                metadata_incomplete=bool(design_dict.get("metadata_incomplete")),
            )
        decisions = decide_fusion_outputs(
            descriptor,
            routing,
            envelope,
            bool(result.outputs.get("fallback", False)),
            tuple(sorted(result.outputs)),
        )
    else:
        ceiling = decoder_category_predicate(descriptor)
        decisions = tuple(
            OutputDecision(
                output_type=name,
                action=SupportAction.PERMIT,
                predicates_evaluated=(ceiling.name,),
                predicate_results=(ceiling,),
            )
            for name in sorted(result.outputs)
        )
    return dataclasses.replace(result, output_decisions=decisions)


def _execute(request: InferenceRequest, models: dict[str, Any]) -> InferenceResult:
    """Routing and execution. Wrapped by `infer`, never called directly."""
    capability, trace, warnings = route(request)
    provenance = frozen_provenance()

    if capability.startswith("unsupported:"):
        key = capability.split(":", 1)[1]
        withheld = {
            "response_categories_exceed_decoder_support": (
                "person_score",
                "person_sd",
                "trait",
                "state",
                "level",
            ),
            "multidimensional_longitudinal": ("trait", "state", "level", "person_score"),
            "related_but_distinct_multimodal": ("fused_mean", "fused_precision"),
            "multiple_continuous_modalities": ("fused_mean", "fused_precision"),
            "single_wave_change_detection": ("change_flag", "change_score"),
            "multigroup_dif": ("dif_flags", "comparability_status", "adjusted_scores"),
            "dif_without_declared_contrast": (
                "dif_flags",
                "comparability_status",
                "adjusted_scores",
            ),
        }[key]
        return unsupported_result(key, trace, withheld)

    if capability in ("multidimensional", "item_assignment", "uncertainty_calibration"):
        from human_measurement.model.multidim import dimensionality_diagnostics

        with torch.no_grad():
            joint = models["multidim"](request.tensors)
        diagnostics = dimensionality_diagnostics(request.tensors, joint)
        outputs: dict[str, Any] = {
            "person_score": joint.mu,
            "person_sd": (0.5 * joint.logvar).exp(),
            "dimensionality": diagnostics,
        }
        if capability == "item_assignment":
            outputs["item_assignment"] = joint.assign
            outputs["item_discrimination"] = joint.a
        return InferenceResult(
            capability=capability,
            status=CAPABILITIES[capability]["status"],
            route=trace,
            outputs=outputs,
            warnings=tuple(warnings),
            provenance=provenance,
        )

    if capability == "dif_comparability":
        from human_measurement.model.dif import comparability_status, group_summaries

        with torch.no_grad():
            forward = models["dif"].forward_with_groups(request.tensors)
        unadjusted = group_summaries(forward, dif_adjusted=False)
        adjusted = group_summaries(forward, dif_adjusted=True)
        status_record = comparability_status(
            forward, unadjusted["impact_estimate"], adjusted["impact_estimate"]
        )
        return InferenceResult(
            capability=capability,
            status=CAPABILITIES[capability]["status"],
            route=trace,
            outputs={
                "comparability_status": status_record,
                "impact_unadjusted": unadjusted["impact_estimate"],
                "impact_adjusted": adjusted["impact_estimate"],
                "operating_threshold": provenance["dif_operating_threshold"],
            },
            warnings=tuple(warnings),
            provenance=provenance,
        )

    if capability == "instrument_linking":
        from human_measurement.model.linking import link

        linked = link(
            models["linking"],
            request.tensors,
            request.second_instrument,
            model_state_sha256=provenance["linking_model_sha256"],
        )
        return InferenceResult(
            capability=capability,
            status=CAPABILITIES[capability]["status"],
            route=trace,
            outputs={
                "linking_status": linked.status,
                "transform_alpha": linked.alpha,
                "transform_beta": linked.beta,
                "transform_se": linked.transform_se,
                "p_same_construct": linked.p_same,
                "bridge_correlation": linked.bridge_correlation,
                "n_anchors_retained": linked.n_anchors_retained,
                # Native-scale outputs are preserved even when linking is
                # supported (SELECTED_MODEL_5D convention).
                "native_person_mean": linked.native_person_mean,
                "linked_person_mean": linked.linked_person_mean,
            },
            warnings=tuple([*warnings, *linked.warnings]),
            provenance=provenance,
        )

    if capability == "psychometric_1d":
        block = _psychometric(request, models)
        return InferenceResult(
            capability=capability,
            status=CAPABILITIES[capability]["status"],
            route=trace,
            outputs={
                "person_score": block["person_score"],
                "person_sd": block["person_sd"],
            },
            warnings=tuple(warnings),
            provenance=provenance,
        )

    if capability == "longitudinal_trait_state":
        block = _psychometric(request, models)
        forward = block["forward"]
        return InferenceResult(
            capability=capability,
            status=CAPABILITIES[capability]["status"],
            route=trace,
            outputs={
                "trait": forward.trait_mu[:, 0],
                "state": forward.state_mu[:, :, 0],
                "level": forward.level_mu[:, :, 0],
                "level_sd": block["person_sd"],
            },
            suppressed={
                "wave_constant_drift": "unidentified without anchors (L35)",
            },
            warnings=tuple(warnings),
            provenance=provenance,
        )

    if capability in ("binary_change_detection", "continuous_change_score"):
        from human_measurement.model.change_detection import (
            NullReference,
            decide,
            person_scores,
            reference_path,
        )

        scores = person_scores(models["longitudinal"], request.tensors)
        reference = NullReference.from_json(reference_path().read_text(encoding="utf-8"))
        decision = decide(
            scores,
            reference,
            detector="likelihood_ratio",
            model_sha256=provenance["longitudinal_model_sha256"],
        )
        score_only = capability == "continuous_change_score"
        indeterminate = bool(decision.indeterminate.all()) and not score_only
        outputs: dict[str, Any] = {
            "change_score": decision.continuous_change_score,
            "detection_wave": decision.detection_wave,
            "direction": decision.change_direction,
            "ambiguity": decision.ambiguity,
        }
        suppressed: dict[str, str] = {}
        if score_only:
            # This capability is the ranking score alone; the binary decision
            # belongs to binary_change_detection and is not offered here.
            suppressed["change_flag"] = "not offered by continuous_change_score"
        elif indeterminate:
            # The 5E.1 safeguard: the binary decision is withheld entirely,
            # while the continuous score - a separate capability - remains.
            suppressed["change_flag"] = "high_state_variance_outside_validated_null_reference (L38)"
        else:
            outputs["change_flag"] = decision.change_flag
            outputs["threshold"] = decision.threshold
        return InferenceResult(
            capability=capability,
            status=Status.INDETERMINATE if indeterminate else CAPABILITIES[capability]["status"],
            route=trace,
            outputs=outputs,
            suppressed=suppressed,
            warnings=tuple([*warnings, *getattr(decision, "warnings", ())]),
            provenance=provenance,
        )

    if capability == "multimodal_fusion":
        from human_measurement.model.multimodal import (
            drift_diagnostic,
            fuse,
            ordinal_summary,
            psychometric_message,
        )
        from human_measurement.model.multimodal_arms import GATE_HARD_REJECT

        with torch.no_grad():
            forward = models["longitudinal"](request.tensors)
            mean, precision = psychometric_message(forward)
            bundle = models["fusion"]
            mu, raw = bundle.encoder(request.continuous)
            gate = bundle.encoder.reliability_gate(
                request.continuous, mean, mu, ordinal_summary(request.tensors)
            )
            if float(gate) <= GATE_HARD_REJECT:
                gate = torch.tensor(0.0)
            drift, _ = drift_diagnostic(request.continuous)
        fused = fuse(mean, precision, mu, raw, gate, drift_warning=drift)
        extra = list(warnings)
        if fused.fallback:
            extra.append("continuous_modality_rejected_psychometric_only_result")
        if fused.drift_warning:
            extra.append("device_drift_detected")
        return InferenceResult(
            capability=capability,
            status=CAPABILITIES[capability]["status"],
            route=trace,
            outputs={
                "fused_mean": fused.fused_mean,
                "fused_precision": fused.fused_precision,
                "reliability_gate": fused.reliability_gate,
                "effective_continuous_precision": fused.effective_continuous_precision,
                "modality_displacement": fused.modality_displacement,
                "conflict_score": fused.conflict_score,
                "fallback": fused.fallback,
            },
            warnings=tuple(extra),
            provenance=provenance,
        )

    raise KeyError(f"no execution path for capability {capability!r}")
