"""Frozen provenance pins, extracted verbatim from the research modules.

In the research repository these constant records live inside the module that
implements the corresponding capability (DIF, linking, change detection,
multimodal fusion). Those capabilities are outside the METER foundational
paper, so their implementations are not released; ``interface.frozen_provenance``
reads the records from here instead. Every value is copied verbatim — the
fixture test ``test_frozen_provenance_identical`` asserts that the provenance
dictionary produced by this package is byte-identical to the one produced by
the frozen research stack.
"""

from __future__ import annotations

#: verbatim from human_measurement/model/dif.py (value of DIF_EFFECT_THRESHOLD inlined)
SELECTED_MODEL_5A: dict[str, object] = {
    "arm": "arm3_label",
    "operating_threshold": 0.36435567004776,
    "effect_threshold_logits": 0.2,
    "model_state_sha256": "cdd81044ba37b2106c678b949cc00999fa77818e1d0b03cf43facb71e1714d92",
    "config_sha256": "0f219a4d190a610ae7f69ec16949e12ab2679640c6d71b733f78a117d32b0f56",
    "comparator_convention": (
        "flag = P(|effect| > 0.2 logits) > operating_threshold; adjustment = drop flagged "
        "items and re-infer (>=4 clean items per construct required); statuses per "
        "comparability_status: supported / supported_after_adjustment / weak / "
        "not_defensible from flagged-information share, clean-item floor, and impact "
        "sensitivity to adjustment"
    ),
}

#: verbatim from human_measurement/model/linking.py (LINKING_VERSION inlined)
SELECTED_MODEL_5D: dict[str, object] = {
    "arm": "arm3_correspond",
    "model_state_sha256": "cb43dd8a705609e1ee957100569995651d463dd8aaf33f7c1ff308807d3ebf93",
    "config_sha256": "d060f3511b633ec97e75e190b3c896654533455b640655a9fcaa46a94cca4753",
    "decision_config_sha256": "2ce5d43a2404af1c7ed8c6d754e1578029920457d59267ae5f11764d8aa255ab",
    "benchmark_manifest_sha256": "9cbf6e6fff6c204e455b1eda62cb9d8eb9ec1794d86ac64857dc9e572df3952f",
    "linking_version": "5d-v1",
    "decision_rule": (
        "graded four-status rule (supported / supported_with_warning / ambiguous / "
        "not_supported) over structural same-construct probability, bridge floors, "
        "disattenuated person-bridge correlation, transform uncertainty, and anchor "
        "stability; raw-score correlation appears nowhere"
    ),
    "statuses": ("supported", "supported_with_warning", "ambiguous", "not_supported"),
}

#: verbatim from human_measurement/model/change_detection.py (versions inlined)
SELECTED_DETECTOR_5E1: dict[str, object] = {
    "detector": "likelihood_ratio",
    "status": "conditionally_supported_experimental",
    "detector_version": "5e1-v1",
    "default_profile": "conservative",
    "reference_sha256": "2709fa550586d7342dc9637f02110111d10655fe98619afee9b67eb7084d6f12",
    "config_sha256": "07121c859c16231d33951c1737118b4fb5ef34d16b92c6b86f421380ad008c39",
    "benchmark_manifest_sha256": (
        "ac76a1ebc1e23bf4494b3f48a57eb151318e7ed8513da9c5eba62ddf2bfd462b"
    ),
    "longitudinal_model_sha256": (
        "224a701b8f6981e1e3eeeb64426a9bc8ab0122d7780bb74bb606a0a630397fc8"
    ),
    "score_is_calibrated_probability": False,  # L37: a ranking, not a posterior
    "out_of_support_condition": "high_state_variance_outside_validated_null_reference",
}

#: verbatim from human_measurement/model/multimodal.py
CAPABILITY_5F = {
    "status": "experimental_not_adopted",
    "selected_arm": "arm6_gated_fusion",
    "failed_criterion": "5_related_but_distinct_constructs",
    "distinct_construct_import": 0.068,
    "distinct_construct_import_limit": 0.05,
    "supported": (
        "one_ordinal_psychometric_modality",
        "one_continuous_longitudinal_modality",
        "one_dimensional_compatible_latent_construct",
        "two_to_six_waves",
        "informative_same_construct_fusion",
        "modality_absence",
        "modality_noise",
        "explicit_conflict",
        "tested_correlated_evidence",
        "tested_synthetic_mechanism_shifts",
    ),
    "unsupported": (
        "unrestricted_fusion_of_related_but_distinct_constructs",
        "multiple_continuous_modalities",
        "real_wearable_or_biomarker_transfer",
        "raw_high_frequency_sensor_data",
    ),
    "wearable_validation_claim": False,
}
