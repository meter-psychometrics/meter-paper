"""Frozen evaluation protocols for real-data cycles (Milestone 6B.1).

Two procedures are frozen here **before** the locked validation dataset is
opened, so neither can be shaped by what that dataset shows:

* `STABILITY_PROTOCOL` — the balanced paired-split stability diagnostic,
  promoted from the L44 follow-up after it separated instrument from model.
* `UNCERTAINTY_PROTOCOL` — the uncertainty-behaviour protocol, which exists
  because L45 showed that predicted width can stay flat while error grows, and
  no single contrast can detect that.

Freezing means the parameters are constants with a recorded hash, asserted by
test. Changing one is a visible, deliberate act with a new version, not an
adjustment made while looking at results.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL_SCHEMA_VERSION = "6b1-protocols-v1"

#: Balanced paired-split stability (frozen 2026-08-04, after the L44 follow-up).
#:
#: The three properties that made it answer the question it was asked:
#: PAIRED, so split-to-split variance cancels and a +0.004 difference is
#: resolvable; BALANCED on endorsement, so no estimator is penalised for
#: difficulty sensitivity; and REPEATED, because one split is one draw - the
#: original 0.691 sat below the tenth percentile of the true distribution.
STABILITY_PROTOCOL: dict[str, Any] = {
    "protocol_id": "balanced_paired_split_stability",
    "version": "1.0",
    "frozen_on": "2026-08-04",
    "n_splits": 60,
    "n_splits_expensive_comparator": 30,
    "seed": 6_400,
    "balancing": "items ordered by endorsement rate; assignment randomised within adjacent pairs",
    "paired": True,
    "comparators": ["sum_score", "principal_component", "mml_grm_eap"],
    "reported": [
        "per-method median, mean, sd, q10, q90 of the split-half correlation",
        "paired difference (model minus comparator) with a 95% interval",
        "fraction of splits on which the model is lower",
    ],
    "correction": (
        "none; correlations are uncorrected and compare methods, not absolute reliability"
    ),
    "interpretation": (
        "if every method falls similarly the loss belongs to the INSTRUMENT; if the model "
        "falls substantially further it is a MODEL-TRANSFER issue"
    ),
    "never": [
        "report a bare split-half number without comparators on the same splits",
        "use an unbalanced or ordered split",
        "infer a model defect from a single split",
    ],
}

#: Uncertainty behaviour (frozen 2026-08-04, before DPQ_J is opened).
#:
#: The load-bearing comparison is PREDICTED width against EMPIRICAL variation.
#: A model can be non-monotone in width (L43) and still be calibrated; it can
#: also be perfectly monotone and badly overconfident. Only measuring how much
#: the estimate actually moves under masking, and comparing that to the width
#: the model claims, distinguishes them. L45 is exactly this gap, and a single
#: masking contrast cannot see it.
UNCERTAINTY_PROTOCOL: dict[str, Any] = {
    "protocol_id": "repeated_nested_mask_uncertainty",
    "version": "1.1",
    "frozen_on": "2026-08-04",
    "revised_on": "2026-08-04",
    "revision": {
        "from_version": "1.0",
        "superseded_sha256": (
            "the v1.0 hash is recorded in configs/model/phase6b1_protocols.v1.json and in the "
            "provenance block of every run made under it; those runs remain valid under v1.0"
        ),
        "what_changed": (
            "one 'never' clause only. No parameter, seed, mask fraction, component or reported "
            "quantity was touched, so no run's numbers change and no result is reinterpreted."
        ),
        "why": (
            "v1.0 said 'never read predicted width as an error bar BEFORE the calibration ratio "
            "is known'. That conditional was correct when written - the ratio was unknown. It is "
            "now known, published in the same artifacts that carry this clause, and above 1 on "
            "every dataset. A reader reaching the clause today satisfies its condition and finds "
            "it grants exactly the permission the contract forbids. The prohibition is "
            "unconditional and is now written that way."
        ),
    },
    "n_repetitions": 20,
    "mask_fractions": [0.0, 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89],
    "nesting": "within a repetition each level removes a superset of the previous level's items",
    "seed": 6_500,
    "components": {
        "predicted_posterior_width": (
            "the model's own interval width per participant at each mask level"
        ),
        "empirical_variation": (
            "the spread of masked estimates around the full-item estimate, across repetitions "
            "at the same mask level - what the uncertainty actually is"
        ),
        "calibration_ratio": (
            "predicted / empirical. 1.0 is calibrated; below 1 is overconfident. This is the "
            "single number the protocol exists to produce"
        ),
        "held_out_proper_score": (
            "log loss and Brier score on the masked items, scored from the model's own "
            "category probabilities - not a correlation, so it cannot be flattered by scale"
        ),
        "item_prediction_degradation": (
            "correlation between the estimate and the held-out responses, by mask level"
        ),
        "participant_level_uncertainty_error": (
            "within participant, across mask levels: does a wider predicted interval "
            "accompany a larger actual error?"
        ),
        "dataset_level_uncertainty_error": (
            "across mask levels: does mean predicted width track RMSE?"
        ),
        "comparator_diagnostics": (
            "the same quantities for MML graded response using its EAP posterior standard "
            "deviation, where R/mirt is available"
        ),
    },
    "reported_per_dataset": [
        "the full curve for every component, never endpoints",
        "participant-level monotonicity rate alongside pooled medians",
        "mean and median curves together",
        "local non-monotonicity with location and magnitude",
        "cross-sectional, trait and state uncertainty separately where applicable",
    ],
    "never": [
        "summarise uncertainty behaviour with a single masking contrast",
        # v1.1: unconditional. See `revision` above for why the v1.0 conditional
        # form had to go.
        "read predicted width as an error bar, under any circumstances, whether or not the "
        "calibration ratio is known - knowing the ratio is what established that the width is "
        "not an error bar, not a licence to treat it as one",
        "recalibrate the frozen model in response to what this protocol finds",
    ],
}


#: The complete analysis specification, frozen before DPQ_J is opened. Freezing
#: only the two procedures would have left the mapping, the comparator set and
#: the metric definitions free to move - and any of those can change a result
#: as much as a procedure can. This block names everything a later run is bound
#: by, so replication means "the same analysis", not "a similar one".
ANALYSIS_FREEZE: dict[str, Any] = {
    "freeze_id": "phase6b1_analysis_freeze",
    "version": "1.0",
    "frozen_on": "2026-08-04",
    "frozen_before_opening": ["nhanes_2017_2018_dpq_j"],
    "model_version": "v0.6.0-beta.1",
    "mapping": {
        "config": "configs/mappings/nhanes_2015_2016_dpq.yaml",
        "reused_for_dpq_j": True,
        "rationale": (
            "DPQ_J carries the identical DPQ variable block - SEQN, DPQ010-DPQ090, the same "
            "0-3 categories and the same 7/9 administrative codes - so the DPQ_I mapping "
            "applies unchanged. Reusing it is what makes the comparison a replication rather "
            "than two separate analyses; only dataset_id differs."
        ),
        "excluded_variables": ["DPQ100"],
    },
    "comparators": ["sum_score", "principal_component", "mml_grm_eap"],
    "metrics": {
        "point_estimate_transfer": [
            "pearson agreement with each comparator",
            "rank agreement with each comparator",
            "cronbach alpha of the item set",
            "item discrimination summary",
            "dimensionality diagnostics",
        ],
        "stability": ["balanced paired split-half, per STABILITY_PROTOCOL"],
        "uncertainty": ["predicted-versus-empirical calibration, per UNCERTAINTY_PROTOCOL"],
        "labelling": (
            "every comparator quantity is AGREEMENT with a conventional estimate, never "
            "accuracy against truth"
        ),
    },
    "replication_questions": [
        "does strong point-estimate transfer replicate in a later population sample?",
        "does the uncertainty mismatch (L45: conservative intervals, ratio ~2) replicate?",
        "does the L43 non-monotonicity replicate?",
        "does the L44 instrument-not-model conclusion replicate?",
    ],
    "prohibited_after_opening": [
        "changing the mapping, any protocol parameter, the comparator set or any metric",
        "retraining, fine-tuning, recalibrating or re-thresholding the model",
        "pooling DPQ_J with DPQ_I",
        "re-running the locked cycle",
    ],
    "known_hazard": {
        "P_DPQ.xpt": (
            "the NHANES 2017-March 2020 pre-pandemic file OVERLAPS DPQ_J: its respondents "
            "include the 2017-2018 cycle. Using both would double-count those people. "
            "P_DPQ is therefore NOT part of this cycle and is not opened."
        )
    },
}


#: Real-world capability contract, revised after the ESS Round 7 evaluation.
#:
#: Capabilities are recorded SEPARATELY and carry different statuses, because
#: they earned different evidence. One status for "the model on real data"
#: would let the strong results launder the weak ones.
#:
#: v3.0 makes two structural changes to how uncertainty is recorded. The v2.0
#: form was a single entry with two status axes; that packed a usable capability
#: and an unusable one into one name, so any citation of it had to be qualified.
#: They are now two capabilities with their own names and statuses:
#:
#:   real_world_relative_uncertainty  supported_experimental - the ORDERING
#:   real_world_interval_scale        not_calibrated         - the UNITS
#:
#: The split is the same finding, said so that each half can be cited alone
#: without dragging the other along. Ranking is supported; the scale is not.
#:
#: DIF is also split, by group count. The 21-country ESS design established
#: that the many-group case is not merely unvalidated but INEXPRESSIBLE by the
#: frozen head, which is a stronger statement and deserves its own entry.
#:
#: None of these status values is one of the five OUTPUT statuses in
#: `interface.Status`. These are capability statuses.
#:
#: v3.4 (2026-08-10) records the Milestone 10A/10B multimodal closures as six
#: SEPARATE capabilities, on the same principle as the v3.0 uncertainty split:
#: one status for "multimodal" would let the strong results (rejection,
#: fallback, precision honesty) launder the weak one (utility). The status
#: vocabulary for these entries is the M10 adjudication vocabulary
#: (supported / demonstrated / not_established), owner-dictated at
#: ratification; both closures are permanent and owner-ratified, H2/H3 are
#: permanently unopened for M10B, and no further multimodal development is
#: chartered under Milestone 10.
#: v3.5 (2026-08-11) records the SHARE external-validation adjudication under
#: the frozen master charter (SHARE_MASTER_CHARTER_FROZEN, 2418ffd):
#: SHARE-A passed its pre-registered gate on both chartered waves - the first
#: PASS under a numeric pre-registered external gate - earning a dedicated
#: supported entry scoped to what was tested; SHARE-B failed its frozen
#: noninferiority gate on the raw-sum arms, replicating ELSA, earning a
#: not_established entry that records the two-dataset negative evidence.
#: Conservative adjudication: no unrelated boundary moved; the
#: cross-sectional entry gains SHARE evidence without a status upgrade.
#: v3.6 (2026-08-18) applies the P5 preregistered EXIT_2_RELABEL, owner-
#: authorized: the truth-referenced synthetic calibration study (P5, 400
#: held-out worlds) failed its conditional-coverage band in 10 of 12
#: prespecified strata, so the prespecified default exit fired - the
#: uncertainty output is released as a RELATIVE RELIABILITY INDICATOR and the
#: posterior standard deviation is withheld from the released public
#: interface. TEXT-ONLY revision: no status changed, no permitted or
#: prohibited use changed, no numerical output of the frozen model changed,
#: and no calibration multiplier is applied at runtime.
REAL_WORLD_CAPABILITY: dict[str, Any] = {
    "contract_version": "3.6",
    "set_on": "2026-08-18",
    "supersedes": "3.5 (which superseded 3.4, 3.3, 3.2, 3.1, 3.0, 2.0)",
    "model_version": "v0.6.0-beta.1",
    "real_world_cross_sectional_point_estimation": {
        "status": "supported_experimental",
        "evidence": {
            "instruments": ["PHQ-9", "CES-D8", "EURO-D"],
            "datasets": [
                "NHANES_2015_2016",
                "NHANES_2017_2018_locked",
                "ESS_Round_7",
                "SHARE_Release_9_wave9_locked",
                "SHARE_Release_9_wave8_locked_replication",
            ],
            "countries_or_cycles": 78,
        },
        "measured": (
            "agreement with conventional estimators 0.92-0.94 on the two NHANES cycles and "
            "0.97-0.998 on ESS7, no schema revision on any dataset, and 0.95-0.99 against "
            "MML-GRM in every one of the 21 ESS countries taken separately; SHARE Release "
            "9.0.0 (v3.5): pooled METER-vs-MML-GRM 0.9851 over 66,812 persons in 28 "
            "countries (wave 9) and 0.9847 over 51,732 in 27 countries (wave 8), "
            "per-country range 0.961-0.990, zero countries excluded"
        ),
        "scope": (
            "cross-sectional only, two instrument families, one construct family (depressive "
            "symptomatology), 23 country-cycles. Agreement with conventional estimates is NOT "
            "accuracy against truth: the model and its comparators share graded-response "
            "assumptions and can be wrong together. No criterion measure has been used"
        ),
    },
    "real_world_relative_uncertainty": {
        "status": "supported_experimental",
        "public_label": "relative reliability indicator",
        "permitted_use": [
            "ordinal_uncertainty_ranking",
            "comparative_diagnostic",
        ],
        "evidence": (
            "the uncertainty-error relationship is strongly positive WITHIN participants: "
            "median r 0.907 (DPQ_I), 0.88 (DPQ_J) and 0.861 (ESS7), positive for 100%, 100% "
            "and 98.7% of participants. Across ESS countries the model's own median relative "
            "uncertainty predicts where it agrees least with MML-GRM at r = -0.803, which is "
            "this capability doing useful work rather than merely being self-consistent"
        ),
        "exit_2_relabel": (
            "v3.6: the P5 preregistered exit rule selected EXIT_2_RELABEL (conditional "
            "coverage outside the prespecified band in 10 of 12 strata), so this output is "
            "released under the public label 'relative reliability indicator' and the "
            "posterior standard deviation is withheld from the released public interface: "
            "release/meter exposes scale-free within-result midrank percentiles of the "
            "withheld widths, supporting exactly the two permitted ordinal uses and nothing "
            "scale-dependent. The underlying posterior standard deviation is preserved "
            "internally for computation and provenance. No numerical output of the frozen "
            "model changed and no calibration multiplier is applied at runtime"
        ),
    },
    "real_world_interval_scale": {
        "status": "not_calibrated",
        "prohibited_uses": [
            "confidence_interval",
            "coverage_interval",
            "literal error_bar",
            "probability statement",
            "threshold-based individual decision",
        ],
        "limitations": {"scale": "L45", "shape": "L43 (dataset-dependent, see L47)"},
        "evidence": (
            "calibration ratio (predicted width / empirical variation) median 2.19 on DPQ_I, "
            "2.18 on DPQ_J, and 3.12-10.54 across the ESS7 masking curve - above 1 at every "
            "level of every dataset, i.e. consistently conservative. The shape failure is "
            "dataset-dependent: monotonicity 0.10 and 0.08 on the NHANES cycles against 0.98 "
            "on ESS7 under an identical measurement (L47)"
        ),
        "truth_referenced_calibration_study": (
            "P5 (preregistered, 400 held-out synthetic worlds with known truth, seed region "
            "disjoint from all training and prior evaluation): empirical coverage of the "
            "nominal 0.95 interval ranged 0.785-0.983 across the 12 prespecified design "
            "strata and fell outside the prespecified [0.93, 0.97] acceptance band in 10 of "
            "12, with directionally different calibration requirements across response "
            "formats - binary widths too narrow, 5- and 7-category widths too wide. A single "
            "global scalar calibration is therefore not supported across the prespecified P5 "
            "strata, and per the preregistered rule exit 1 (fit a calibration map) is closed "
            "and exit 2 (relabel) was applied. This is not a claim that calibration is "
            "impossible: the current flagship width is not absolutely calibrated, and a "
            "future stratum-conditional calibration system would constitute a separate "
            "capability requiring separate validation"
        ),
        "why_the_prohibitions_follow": {
            "confidence_interval": "a frequentist coverage guarantee was never established",
            "coverage_interval": "nominal coverage is unvalidated; the scale error alone "
            "would break it",
            "literal error_bar": "the width runs 2x to 10x the estimate's actual variation, so "
            "the bar drawn would not be the error",
            "probability statement": "no probability derived from this width is warranted - the "
            "scale it would be read off IS the miscalibrated quantity",
            "threshold-based individual decision": "misplaces every participant relative to a "
            "fixed cutoff, and on the NHANES datasets not even consistently between "
            "participants. The prohibition with direct consequences for people",
        },
        "warning": "real_world_interval_scale_not_calibrated",
    },
    "multicountry_dif": {
        "status": "not_supported",
        "reason": (
            "INEXPRESSIBLE, not merely unvalidated. dif.group_code_from_settings reduces every "
            "setting to (setting > 0), so a k-group design collapses to one group against the "
            "other k-1 pooled. A many-country request returns a confident number for a contrast "
            "nobody asked for"
        ),
        "limitation": "L46",
        "required_decomposition": (
            "prespecified two-group contrasts with an explicitly declared reference and focal "
            "group, each labelled with that scope"
        ),
    },
    #: Milestone 6B.2.3. The longitudinal boundaries below are no longer
    #: advisory: `longitudinal_design.route_longitudinal_design` applies them at
    #: the output boundary, and `integrated.infer` suppresses, renames and
    #: labels accordingly. The statuses here are the IN-SUPPORT ones; an
    #: out-of-support design receives the status in LONGITUDINAL_CONTRACTS.
    "longitudinal_enforcement": {
        "enforced_at": "human_measurement.longitudinal_design",
        "applied_by": "human_measurement.integrated._apply_longitudinal_boundaries",
        "milestone": "6B.2.3",
        "warnings": [
            "longitudinal_design_out_of_validated_support",
            "longitudinal_trait_share_binary_bias",
            "continuous_change_magnitude_attenuated",
            "longitudinal_design_metadata_incomplete",
            "longitudinal_trait_may_absorb_state_variance",
        ],
        "fail_safe": (
            "a longitudinal request with no design descriptor, or with routing-relevant "
            "metadata absent, is routed as OUT of support. An undescribed design is not "
            "thereby a supported one"
        ),
        "closes": "L50 at the interface; the underlying insensitivity is unchanged",
    },
    # --- longitudinal, from the Milestone 6B.2.2 synthetic audit -----------
    #
    # These carry NO real-data status: no real longitudinal dataset has been
    # opened. They record where the frozen heads are informative on synthetic
    # worlds shaped like the one about to be opened, so that the HRS analyses
    # are planned around measured behaviour rather than nominal thresholds.
    "longitudinal_trait": {
        "status": "experimental",
        "real_data_status": "none_established",
        "evidence": (
            "trait recovery r 0.75-0.94 across all 164 audited synthetic designs, including "
            "2-wave (0.71); degrades gracefully with fewer waves, lower trait share and fewer "
            "response categories rather than failing"
        ),
        "informative_when": "broadly, across every audited design",
        "limitation": ["L49"],
    },
    "longitudinal_state": {
        "status": "experimental",
        "real_data_status": "none_established",
        "evidence": (
            "state recovery r 0.63 at OU decay per wave below 0.05, 0.53 at 0.3-0.6, 0.29 at "
            "0.6-0.85 and 0.05 above 0.85 - weakly identified exactly where the state is "
            "persistent relative to the observation schedule"
        ),
        "informative_when": "decay per wave below ~0.6",
        "not_informative_when": (
            "decay above 0.85; a biennial schedule with realistic symptom persistence sits at "
            "r ~ 0.29"
        ),
        "limitation": ["L49"],
    },
    "longitudinal_trait_share": {
        "status": "not_calibrated",
        "real_data_status": "none_established",
        "evidence": (
            "estimated trait share exceeds the true share in every audited cell, by +0.032 to "
            "+0.685. Worst where the true trait share is low (+0.685 at 0.25) and where items "
            "are binary (+0.378 at 2 categories against -0.015 at 5). Insensitive to wave count "
            "and spacing, so more waves do not fix it"
        ),
        "prohibited_uses": ["variance decomposition reported without the measured bias"],
        "limitation": ["L49"],
    },
    "longitudinal_binary_change_detection": {
        "status": "experimental",
        "real_data_status": "none_established",
        "evidence": (
            "sensitivity 0.041, specificity 0.997, false-positive rate 0.0032, precision 0.400 "
            "pooled over 164 worlds; permanent 0.051 and temporary 0.021 against the same "
            "unchanged pool. Detection exceeds 0.3 only above three stationary standard "
            "deviations of realized change"
        ),
        "informative_when": "realized change exceeds ~3 stationary SDs",
        "prohibited_uses": [
            "reading a null result as evidence that no change occurred",
            "reading the continuous change score as a magnitude - attenuated 8-14x",
        ],
        "limitation": ["L48", "L50"],
    },
    "binary_group_dif": {
        "status": "experimental",
        "limitation": ["no_real_data_positive_control_in_ESS7"],
        "evidence": (
            "20 two-group reference-country contrasts on ESS7 returned zero flagged items with "
            "max |effect| 0.063-0.132 logits against a 0.20 substantive floor, stable under "
            "both size-matching and half-sample subsampling"
        ),
        "why_experimental": (
            "a uniform null is consistent with genuine invariance AND with an insensitive head, "
            "and the ESS7 design contains no contrast with known DIF that could distinguish "
            "them. Absence of a positive control is why this is experimental rather than "
            "supported, and a SYNTHETIC positive control does not discharge it - it would "
            "establish sensitivity on the data the head was trained against, not on real data"
        ),
    },
    # --- Milestone 7 closure (contract 3.3, 2026-08-06) ---------------------
    #: Unrestricted 1-to-5-factor structure discovery, closed WITHOUT
    #: demonstration after four smoke runs and three mechanism repairs. This is
    #: a SYNTHETIC-experiment status: no real dataset was ever opened for it
    #: and the 812-world confirmatory benchmark remains pinned and unspent.
    "battery_structure_discovery_1_to_5": {
        "status": "experimental_out_of_support",
        "real_data_status": "none_established",
        "milestone": "7 (closed 2026-08-06)",
        "evidence": (
            "four smoke runs (8d7f044, 4f3999a, 1dc6ef0, 4aead8c) each produced a "
            "single-column factor-count confusion matrix - every world reported as one "
            "factor - with balanced count accuracy 0.200 and item-clustering ARI 0.158, "
            "unchanged through gate-schedule repair, exact Poisson-binomial cardinality "
            "supervision, and slot-conditioned gate decoupling that verifiably removed "
            "the gate coupling (pairwise correlation 0.992+ down to 0.13-0.82). The "
            "DEVELOPMENT fit improved across all four runs while structure never appeared"
        ),
        "mechanism": (
            "the trained one-sample ELBO attenuates the likelihood's gate signal ~24x "
            "while parsimony is unattenuated, leaving every optional gate's optimum "
            "interior and below the activation threshold; see the exact gate-override "
            "audit (c2a6eee) and docs/milestone_7_closure.md"
        ),
        "prohibited_uses": [
            "reporting an automatically discovered factor count as a finding",
            "citing any Milestone 8 known-structure result as evidence for this capability",
        ],
        "reopening": (
            "docs/milestone_7_closure.md section 4 binds the starting point: count-weight "
            "recalibration under the frozen f8c9b70 procedure and the Stage A "
            "zero-gradient window are the pre-registered untried candidates"
        ),
    },
    # --- Milestone 10A/10B multimodal closures (contract 3.4, 2026-08-10) ---
    #: Both milestones are closed and owner-ratified. M10A: one locked
    #: real-data run (reports/m10a/). M10B: seven-candidate synthetic
    #: development chain closed at Candidate 7 (reports/m10b/m10b_closure.json,
    #: commit 7a28241, ratified). Every status below is scoped to the tested
    #: M10 architecture and operating region and must not be generalized.
    "real_world_safe_modality_rejection_and_fallback": {
        "status": "supported",
        "milestone": "10A (closed 2026-08-08, permanent)",
        "evidence": (
            "single locked NHANES 2005-2006 run (N=3,024, T=1): the wearable modality was "
            "rejected at every information level (gate snapped to 0 at 9/9 through 3/9) with "
            "bit-exact psychometric fallback everywhere (delta_r 0.0, relative precision "
            "inflation 1.0, median displacement 0.0); safety gates S1-S4 all held and the "
            "negative controls were rejected. All nine evaluability preconditions passed, so "
            "this is a scientific outcome, not a technical escape"
        ),
        "scope": (
            "one wearable modality, one construct (depressive symptomatology), one cycle, "
            "T=1. Rejection safety is what is supported - not that rejection was the correct "
            "scientific verdict on the modality (L52: the gate's own admission evidence "
            "degrades with the psychometric anchor)"
        ),
        "limitation": ["L52"],
    },
    "controlled_noise_conflict_person_mismatch_rejection": {
        "status": "supported",
        "milestone": "10B (synthetic only)",
        "evidence": (
            "at the calibrated Candidate-7 operating point: pure-noise (s1) and conflict (s4) "
            "rejection 1.0 with bit-exact fallback at every information level in both the "
            "native and T=1 regimes (G3/G4a); seeded between-person permutation rejection 1.0 "
            "with genuine same-construct admission 0.972-0.979 (G5). Held across the whole "
            "C1-C7 candidate chain and on the fresh calibration regions"
        ),
        "scope": "SYNTHETIC worlds only, the frozen M10B family set; real-data counterpart "
        "is the M10A rejection result above",
    },
    "exact_psychometric_fallback": {
        "status": "supported",
        "milestone": "10A (real data) and 10B (synthetic)",
        "evidence": (
            "every rejection returns the psychometric-only posterior bit-for-bit: M10A real "
            "data at all four information levels; M10B c7cal 2,577 rejections plus "
            "whole-modality-absence and person-absence probes, all exact (G6)"
        ),
    },
    "multimodal_precision_honesty_conservative_mechanism": {
        "status": "demonstrated",
        "milestone": "10B Candidate 7 (final conservative mechanism, M-6b)",
        "evidence": (
            "under frozen Amendments 9-11: G7b exceedance count ZERO at all 8 calibration "
            "grid points on the fresh region (allowed K(n)=4) and at the calibrated point on "
            "the development region (allowed K(n)=8), with G7a median RPI 0.977-0.988 against "
            "bounds at 1.000-1.001. Candidate 6, without the mechanism, showed 5 exceedances "
            "at every fresh grid point. The mechanism is a monotone conservative UCB ceiling "
            "deflation of claimed auxiliary precision where T=1 separation cannot be "
            "certified; it never loosens a claim"
        ),
        "scope": (
            "demonstrated within the tested M10B architecture and operating region only; "
            "this is a property of the final mechanism on the measured families, not a "
            "general warranty of honest precision"
        ),
    },
    "incremental_multimodal_utility": {
        "status": "not_established",
        "milestone": "10B (closed 2026-08-10, ratified; commit 7a28241)",
        "evidence": (
            "Candidate 6 met the sparse-utility requirement (G1 5/9 median delta_r 0.0254 "
            "dev / 0.0267 fresh, bar 0.02) but failed precision-tail honesty (5 G7b "
            "exceedances at every fresh grid point against an allowed count of 4). "
            "Candidate 7 repaired the identified precision defect (zero exceedances "
            "everywhere measured) but failed the frozen G1 5/9 utility requirement (0.0165 "
            "on the development region), replicated on an independent fresh region (0.0172). "
            "No measured operating point holds both sides of the contract. The shortfall is "
            "not harm: no-harm passes, admission is unchanged, and every paired per-world "
            "change is <= 0 - the mechanism shrinks fusion it cannot price honestly at T=1 "
            "sparsity"
        ),
        "scope": (
            "the tested M10B architecture and operating region ONLY - the frozen 5E anchor, "
            "the M-6/M-6b defence family, the frozen family set and bars. Not evidence about "
            "other architectures, mechanisms, regimes, or bars, and not to be generalized"
        ),
        "prohibited_uses": [
            "citing the precision-honesty demonstration as utility evidence",
            "citing Candidate 6's utility pass without its precision-tail failure",
        ],
        "limitation": ["L52", "L53"],
    },
    # --- SHARE external validation (contract 3.5, 2026-08-11) ----------------
    #: Adjudicated from the frozen master charter's locked one-shot runs;
    #: evidence at reports/share/share_a_w9_result.json, share_a_w8_result.json,
    #: share_b_result.json. Statuses use the M10 adjudication vocabulary.
    "external_multinational_1d_zero_shot_transfer": {
        "status": "supported",
        "milestone": "SHARE-A (charter frozen 2418ffd; executed 2026-08-11)",
        "evidence": (
            "first PASS under a numeric pre-registered external gate (GATE_SA, the 6B.5 "
            "Gate A thresholds 0.95/0.95/0.94 imported unchanged): wave 9 pooled "
            "METER-vs-specialist pearson 0.9851 (participant bootstrap [0.9849, 0.9854]), "
            "midrank spearman 0.9896, n=66,812, 28/28 countries evaluable, median country "
            "agreement 0.9842, worst 0.9674; wave 8 temporal replication executed exactly "
            "as frozen: 0.9847 (lo 0.9844), rho 0.9886, n=51,732, 27/27 countries, worst "
            "0.9607. Zero-shot: no dataset-specific METER fitting of any kind"
        ),
        "scope": (
            "frozen 1D scoring of short binary-scored distress instruments (EURO-D), "
            "cross-sectional, interviewer-administered, 27-28 European countries + Israel. "
            "The wave-8 result is a TEMPORAL/WAVE replication on an overlapping panel "
            "(38,071 shared persons of 51,732) - robustness across administrations and "
            "time, NOT an independent-cohort replication, per the charter terminology "
            "lock. Household-couple dependence disclosed; clustered sensitivity interval "
            "concordant"
        ),
        "prohibited_uses": [
            "describing wave 8 as an independent or participant-independent replication",
            "generalizing beyond short 1D distress instruments or to any capability the "
            "charter did not test",
        ],
    },
    "real_world_longitudinal_prospective_noninferiority": {
        "status": "not_established",
        "milestone": "SHARE-B (charter frozen 2418ffd; executed 2026-08-11) + 6B.5 ELSA",
        "evidence": (
            "TWO independent frozen locked replications now fail the same pre-registered "
            "noninferiority gate on the raw-sum arm: ELSA CES-D8 (delta lower bounds "
            "-0.0237/-0.0236 vs -0.02) and SHARE EURO-D (primary arm n=20,132: -0.0236; "
            "sensitivity arm n=8,913: -0.0243; replicated at -0.0237/-0.0233) - same "
            "direction, same magnitude, across instrument, dataset, and 27+ countries. "
            "The vs-specialist (MML random intercept) noninferiority PASSED on SHARE "
            "(-0.0088/-0.0093) and every guardrail passed on both datasets: convergence "
            "with the specialist 0.987 (SHARE Gate A), stability spread 0.0007, "
            "missingness robustness intact"
        ),
        "interpretation": (
            "for predicting a held-out future outcome from sparse longitudinal input, "
            "the raw-sum person mean retains a small (~0.01-0.02 r) advantage that the "
            "frozen -0.02 margin does not tolerate. METER's longitudinal scores converge "
            "with the specialist model; the shortfall is specific to prospective "
            "prediction against the simplest baseline. This is a characterized boundary, "
            "replicated twice, not sampling noise"
        ),
        "prohibited_uses": [
            "claiming METER longitudinal scores are noninferior to raw sums for "
            "prospective prediction",
            "presenting the passing convergence guardrail as a supported longitudinal "
            "utility claim",
        ],
    },
    "real_world_multimodal_utility": {
        "status": "not_established",
        "milestone": "10A (permanent) and 10B (H2/H3 permanently unopened)",
        "evidence": (
            "M10A utility gates U1/U2/U4 failed: the single locked real-data run rejected "
            "the wearable modality at every level, so no real-world fusion utility was "
            "observed; and the M10B synthetic closure removes the synthetic basis on which "
            "a real-world utility claim would have been staged. The M10B real-data holdouts "
            "(H2/H3) are permanently unopened under the ratified closure"
        ),
        "scope": "one wearable modality, one construct, one dataset tested; no real-world "
        "multimodal utility claim of any kind is supported",
    },
}


def protocol_sha256(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()


def frozen_protocols() -> dict[str, Any]:
    """Both protocols with their hashes, for the provenance of every run."""
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "stability": {
            "protocol": STABILITY_PROTOCOL,
            "sha256": protocol_sha256(STABILITY_PROTOCOL),
        },
        "uncertainty": {
            "protocol": UNCERTAINTY_PROTOCOL,
            "sha256": protocol_sha256(UNCERTAINTY_PROTOCOL),
        },
        "analysis_freeze": {
            "specification": ANALYSIS_FREEZE,
            "sha256": protocol_sha256(ANALYSIS_FREEZE),
        },
        "real_world_capability": {
            "contract": REAL_WORLD_CAPABILITY,
            "sha256": protocol_sha256(REAL_WORLD_CAPABILITY),
        },
    }
