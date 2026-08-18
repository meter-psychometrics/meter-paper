"""Milestone 6B.2.3: longitudinal design description and operating-boundary routing.

Milestone 6B.2.2 measured where the frozen longitudinal heads are actually
informative. This module makes those boundaries *enforceable at the output
boundary* rather than documented in a report nobody reads at the point of use.

Nothing here changes weights, latent representations, heads, or numerical
inference. It changes what is emitted, under which capability status, with
which warning, and what is withheld.

**Why the new statuses are not members of `interface.Status`.** The five output
statuses were frozen at Milestone 6A and a test asserts no sixth exists. The
statuses this milestone introduces - `experimental_out_of_support`,
`not_interpretable_for_design`, `not_supported_for_primary_inference`,
`ordinal_only` - are *capability* statuses: they describe what an output means
for a particular design, not how the request was routed. That is the same
distinction the real-world capability contract already draws with
`not_calibrated` and `not_supported`, and it keeps one frozen vocabulary intact
while letting a second grow. Every result carries both.

**What is inferred and what is not.** Every field of the descriptor is an
OBSERVABLE of the delivered data: counts, dates, spacings, missingness
patterns, category structure. The latent-state half-life is *not* estimated
here and must not be, because it is exactly the quantity whose confounding with
the trait the audit found. Observable spacing mismatch supports a *warning* -
this design is outside the region where the model was trained and evaluated -
and never a claim that the model is wrong on it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

DESIGN_SCHEMA_VERSION = "6b23-longitudinal-design-v1"

#: The wave-spacing region every Milestone 5E training family occupied
#: (20-30 day mean spacing). A design far outside it is out of the region the
#: model was trained and evaluated on - an observable fact about coverage, not
#: a claim about correctness.
TRAINING_SPACING_DAYS = (20.0, 30.0)

#: Beyond this multiple of the training band's upper edge, the mismatch is
#: recorded as MATERIAL. HRS at ~730 days sits at roughly 24x.
MATERIAL_SPACING_MULTIPLE = 3.0

#: Below this many response categories an item is binary. The 6B.2.2 audit
#: measured +0.378 trait-share bias at two categories against -0.015 at five,
#: so the binary case is called out by name rather than folded into a range.
BINARY_CATEGORY_COUNT = 2

#: Capability statuses this milestone introduces. Deliberately NOT members of
#: `interface.Status` - see the module docstring.
CAPABILITY_STATUSES = (
    "supported_experimental",
    "experimental_out_of_support",
    "not_interpretable_for_design",
    "not_supported_for_primary_inference",
    "ordinal_only",
)

#: Warning identifiers. Strings are the contract; they are asserted by test and
#: referenced from the capability contract, so they cannot drift.
OUT_OF_SUPPORT_WARNING = "longitudinal_design_out_of_validated_support"
TRAIT_SHARE_BIAS_WARNING = "longitudinal_trait_share_binary_bias"
CHANGE_ATTENUATION_WARNING = "continuous_change_magnitude_attenuated"
DESIGN_METADATA_INCOMPLETE_WARNING = "longitudinal_design_metadata_incomplete"
TRAIT_ABSORBS_STATE_WARNING = "longitudinal_trait_may_absorb_state_variance"

#: The exact text the out-of-support warning carries. Worded to state coverage,
#: not error: the model may well be fine here, and nothing in 6B.2.2 shows it
#: is certainly wrong on any real design.
OUT_OF_SUPPORT_TEXT = (
    "This design is materially outside the frozen longitudinal model's validated training "
    "and evaluation region. Outputs are routed according to empirically established operating "
    "boundaries and must not be interpreted beyond their capability-specific contract."
)


@dataclass(frozen=True, slots=True)
class LongitudinalDesign:
    """A machine-readable description of a longitudinal design.

    Every field is an observable of the delivered data. Optional fields are
    ``None`` when the caller did not supply the metadata, and a ``None`` in any
    routing-relevant field forces the conservative route rather than a guess.
    """

    # --- size and instrument ------------------------------------------------
    n_participants: int
    n_items: int
    item_response_type: str  # "binary" | "ordinal" | "mixed" | "unknown"
    n_categories_per_item: tuple[int, ...] = ()

    # --- schedule -----------------------------------------------------------
    n_waves: int = 1
    wave_elapsed_days: tuple[float, ...] = ()
    median_wave_spacing_days: float | None = None
    min_wave_spacing_days: float | None = None
    max_wave_spacing_days: float | None = None
    irregular_spacing: bool | None = None

    # --- participation ------------------------------------------------------
    proportion_by_observed_waves: dict[str, float] = field(default_factory=dict)
    missing_wave_burden: float | None = None
    monotone_dropout_burden: float | None = None
    item_level_missingness: float | None = None
    participant_level_missingness: float | None = None

    # --- declared metadata --------------------------------------------------
    proxy_indicator_supplied: bool | None = None
    repeated_item_consistency: bool | None = None
    coding_consistent_across_waves: bool | None = None

    schema_version: str = DESIGN_SCHEMA_VERSION

    # --- derived observables ------------------------------------------------

    def is_binary_items(self) -> bool:
        if self.item_response_type == "binary":
            return True
        if not self.n_categories_per_item:
            return False
        return max(self.n_categories_per_item) <= BINARY_CATEGORY_COUNT

    def spacing_multiple_of_training(self) -> float | None:
        """How far the median spacing sits beyond the training band's top."""
        if self.median_wave_spacing_days is None:
            return None
        return float(self.median_wave_spacing_days / TRAINING_SPACING_DAYS[1])

    def spacing_materially_outside_training(self) -> bool | None:
        multiple = self.spacing_multiple_of_training()
        if multiple is None:
            return None
        return multiple >= MATERIAL_SPACING_MULTIPLE

    def missing_routing_metadata(self) -> tuple[str, ...]:
        """Routing-relevant fields the caller did not supply.

        Deliberately narrow: only fields that change a routing decision. A
        design missing a proxy indicator is incompletely *described*, but the
        route does not turn on it, so demanding it would train callers to
        supply placeholder values for the fields that do matter.
        """
        missing = []
        if self.median_wave_spacing_days is None:
            missing.append("median_wave_spacing_days")
        if self.item_response_type == "unknown" and not self.n_categories_per_item:
            missing.append("item_response_type/n_categories_per_item")
        if self.n_waves < 1:
            missing.append("n_waves")
        return tuple(missing)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def trace_entries(self) -> tuple[str, ...]:
        """Compact, human-readable entries for the route trace."""
        spacing = (
            "unknown"
            if self.median_wave_spacing_days is None
            else f"{self.median_wave_spacing_days:.0f}d"
        )
        multiple = self.spacing_multiple_of_training()
        response_type = "binary" if self.is_binary_items() else self.item_response_type
        return (
            f"design_waves={self.n_waves}",
            f"design_items={self.n_items}",
            f"design_response_type={response_type}",
            f"design_median_spacing={spacing}",
            f"design_spacing_x_training={'unknown' if multiple is None else f'{multiple:.1f}'}",
            f"design_sha={self.sha256()[:12]}",
        )


def describe_from_canonical(dataset: Any, **declared: Any) -> LongitudinalDesign:
    """Build a descriptor from a canonical dataset plus declared metadata.

    Anything the data can show is measured; anything only the operator knows -
    proxy indicators, whether item wording changed between waves - must be
    DECLARED and is ``None`` otherwise. The descriptor never guesses those,
    for the same reason the ingestion layer never guesses item direction.
    """
    import numpy as np

    responses = np.asarray(dataset.responses)
    mask = np.asarray(dataset.response_mask)
    n_participants, n_waves, n_items = responses.shape

    categories = tuple(int(c) for c in np.asarray(dataset.n_categories).ravel())
    response_type = (
        "binary"
        if categories and max(categories) <= BINARY_CATEGORY_COUNT
        else "ordinal"
        if categories
        else "unknown"
    )

    wave_observed = mask.any(axis=2)  # (P, T)
    observed_counts = wave_observed.sum(axis=1)
    proportion = {
        str(int(count)): round(float((observed_counts == count).mean()), 6)
        for count in sorted(set(int(c) for c in observed_counts))
    }

    # Monotone dropout: a person whose observed waves form a prefix and who is
    # absent at the last wave. Distinguished from intermittent absence, which
    # is a different mechanism and a different bias.
    monotone = 0
    intermittent = 0
    for row in wave_observed:
        indices = np.flatnonzero(row)
        if indices.size == 0 or indices.size == n_waves:
            continue
        is_prefix = indices.size == indices[-1] + 1 and indices[0] == 0
        if is_prefix:
            monotone += 1
        else:
            intermittent += 1

    times = np.asarray(dataset.wave_time, dtype=float)
    elapsed: tuple[float, ...] = ()
    median_spacing = min_spacing = max_spacing = None
    irregular = None
    if times.size and np.isfinite(times).any():
        per_wave = np.nanmedian(times, axis=0) if times.ndim > 1 else times
        per_wave = np.asarray(per_wave, dtype=float).ravel()
        if np.isfinite(per_wave).sum() >= 2:
            finite = per_wave[np.isfinite(per_wave)]
            elapsed = tuple(round(float(v), 3) for v in finite)
            gaps = np.diff(finite)
            if gaps.size:
                median_spacing = float(np.median(gaps))
                min_spacing = float(gaps.min())
                max_spacing = float(gaps.max())
                irregular = bool(
                    median_spacing > 0 and (max_spacing - min_spacing) / median_spacing > 0.25
                )

    return LongitudinalDesign(
        n_participants=int(n_participants),
        n_items=int(n_items),
        item_response_type=response_type,
        n_categories_per_item=categories,
        n_waves=int(n_waves),
        wave_elapsed_days=elapsed,
        median_wave_spacing_days=median_spacing,
        min_wave_spacing_days=min_spacing,
        max_wave_spacing_days=max_spacing,
        irregular_spacing=irregular,
        proportion_by_observed_waves=proportion,
        missing_wave_burden=round(float(1.0 - wave_observed.mean()), 6),
        monotone_dropout_burden=round(float(monotone / max(n_participants, 1)), 6),
        item_level_missingness=round(float(1.0 - mask.mean()), 6),
        participant_level_missingness=round(float((observed_counts < n_waves).mean()), 6),
        proxy_indicator_supplied=declared.get("proxy_indicator_supplied"),
        repeated_item_consistency=declared.get("repeated_item_consistency"),
        coding_consistent_across_waves=declared.get("coding_consistent_across_waves"),
    )


#: The five longitudinal capability contracts, keyed by capability name. Each
#: records the status for an out-of-support design, what the output may be used
#: for, what it may not, and which limitation it comes from.
LONGITUDINAL_CONTRACTS: dict[str, dict[str, Any]] = {
    "longitudinal_trait": {
        "status_in_support": "supported_experimental",
        "status_out_of_support": "supported_experimental",
        "permitted_uses": [
            "relative stable-trait scoring",
            "comparison across wave subsets",
            "agreement with longitudinal comparators",
            "later-response prediction",
            "exploratory external-validity analyses",
        ],
        "prohibited_uses": [
            "treating the estimate as ground truth",
            "assuming the estimated trait contains no state variance",
            "interpreting trait share from the same run as valid",
        ],
        "disclosure": (
            "Stable-trait estimates may absorb time-varying variance under the audited design: "
            "Milestone 6B.2.2 measured trait-share over-attribution in every cell, up to +0.685."
        ),
        "evidence": "trait recovery r 0.75-0.94 across 164 synthetic designs, 0.71 at two waves",
        "limitation": ["L49"],
    },
    "longitudinal_state": {
        "status_in_support": "supported_experimental",
        "status_out_of_support": "experimental_out_of_support",
        "permitted_uses": [
            "exploratory comparison with within-person response variation",
            "diagnostic comparison with fitted longitudinal models",
            "investigation of possible responsiveness",
        ],
        "prohibited_uses": [
            "validated wave-specific state measurement",
            "individual monitoring",
            "clinical or operational decisions",
            "literal separation of trait and state",
            "headline efficacy or validity claims",
        ],
        "disclosure": (
            "State recovery was r ~ 0.29 in the HRS-like audit region and r ~ 0.05 where the "
            "state is highly persistent relative to the schedule."
        ),
        "evidence": "state recovery r 0.63 at decay <0.05, 0.29 at 0.6-0.85, 0.05 above 0.85",
        "limitation": ["L49"],
    },
    "longitudinal_trait_share": {
        "status_in_support": "not_calibrated",
        "status_out_of_support": "not_interpretable_for_design",
        "permitted_uses": [
            "internal diagnostic artifact, clearly marked, never as a proportion",
        ],
        "prohibited_uses": [
            "reporting as an interpretable proportion",
            "variance decomposition",
            "any user-facing numeric output",
        ],
        "suppress_numeric_output": True,
        "warning": TRAIT_SHARE_BIAS_WARNING,
        "disclosure": (
            "Trait share was biased upward in every audited cell, by +0.378 on binary items "
            "against -0.015 on five-category items, and up to +0.685 where the true trait "
            "share was low. Adding waves did not remove the bias."
        ),
        "limitation": ["L49"],
    },
    "binary_change_detection": {
        "status_in_support": "conditionally_supported_experimental",
        "status_out_of_support": "not_supported_for_primary_inference",
        "permitted_uses": [
            "exploratory description of only the most extreme changes",
            "comparison with the synthetic operating-point curve",
        ],
        "prohibited_uses": [
            "individual change classification",
            "clinically meaningful change",
            "inference of no change from no flag",
            "sensitivity claims below the established operating region",
        ],
        "required_disclosure": {
            "sensitivity_below_three_stationary_sd": 0.041,
            "specificity": 0.997,
            "score_is_calibrated_probability": False,
            "source": "frozen synthetic audit, Milestone 6B.2.2",
        },
        "limitation": ["L48", "L50"],
    },
    "continuous_change": {
        "status_in_support": "supported_experimental",
        "status_out_of_support": "ordinal_only",
        "output_rename": "continuous_change_rank_score",
        "permitted_uses": [
            "ranking relatively larger and smaller apparent changes",
            "exploratory associations",
        ],
        "prohibited_uses": [
            "literal magnitude",
            "standardized change",
            "effect size",
            "clinically meaningful change",
            "threshold-based individual decisions",
        ],
        "warning": CHANGE_ATTENUATION_WARNING,
        "disclosure": (
            "Magnitude was attenuated 8- to 14-fold in the Milestone 6B.2.2 audit "
            "(OLS slope of estimate on truth, 0.07-0.13). The ordinal information survives; "
            "the scale does not."
        ),
        "limitation": ["L50"],
    },
}


@dataclass(frozen=True, slots=True)
class DesignRouting:
    """The routing decision for one longitudinal design."""

    design: LongitudinalDesign
    out_of_support: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    capability_statuses: dict[str, str]
    metadata_incomplete: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "design": self.design.as_dict(),
            "design_sha256": self.design.sha256(),
            "out_of_support": self.out_of_support,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "capability_statuses": dict(self.capability_statuses),
            "metadata_incomplete": self.metadata_incomplete,
            "out_of_support_text": OUT_OF_SUPPORT_TEXT if self.out_of_support else None,
            "schema_version": DESIGN_SCHEMA_VERSION,
        }


def route_longitudinal_design(design: LongitudinalDesign) -> DesignRouting:
    """Decide capability statuses and warnings from observable design features.

    Fail-safe: when a routing-relevant field is absent the design is treated as
    OUT of support. A design nobody described is not thereby a supported one,
    and the cost of a spurious warning is far below the cost of a silent
    ordinary-looking number on a design nobody checked.
    """
    reasons: list[str] = []
    warnings: list[str] = []

    missing = design.missing_routing_metadata()
    metadata_incomplete = bool(missing)
    if metadata_incomplete:
        warnings.append(DESIGN_METADATA_INCOMPLETE_WARNING)
        reasons.append(
            "routing metadata absent (" + ", ".join(missing) + "); conservative route taken"
        )

    spacing_outside = design.spacing_materially_outside_training()
    if spacing_outside:
        multiple = design.spacing_multiple_of_training() or 0.0
        reasons.append(
            f"median wave spacing is {multiple:.1f}x the top of the "
            f"{TRAINING_SPACING_DAYS[0]:.0f}-{TRAINING_SPACING_DAYS[1]:.0f} day training band"
        )

    binary = design.is_binary_items()
    if binary:
        reasons.append(
            "binary repeated items; the trait-share decoder was not validated on them "
            "(6B.2.2 measured +0.378 share bias at two categories)"
        )

    out_of_support = bool(metadata_incomplete or spacing_outside or binary)
    if out_of_support:
        warnings.append(OUT_OF_SUPPORT_WARNING)

    statuses: dict[str, str] = {}
    for capability, contract in LONGITUDINAL_CONTRACTS.items():
        statuses[capability] = str(
            contract["status_out_of_support"] if out_of_support else contract["status_in_support"]
        )

    # Capability-specific warnings, attached only where they apply.
    if binary or out_of_support:
        warnings.append(TRAIT_SHARE_BIAS_WARNING)
    if out_of_support:
        warnings.append(CHANGE_ATTENUATION_WARNING)
        warnings.append(TRAIT_ABSORBS_STATE_WARNING)

    return DesignRouting(
        design=design,
        out_of_support=out_of_support,
        reasons=tuple(reasons),
        warnings=tuple(dict.fromkeys(warnings)),
        capability_statuses=statuses,
        metadata_incomplete=metadata_incomplete,
    )
