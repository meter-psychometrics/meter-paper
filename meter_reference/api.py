"""The METER research scoring API.

A thin, gated wrapper over the frozen ``human_measurement`` inference stack.
Nothing here trains, fits, recalibrates, or modifies a threshold: every number
is produced by the frozen model (``v0.6.0-beta.1``, state ``224a701b...``,
capability contract v3.6), and this layer decides only whether a request is
inside the validated support region, refusing or flagging it when it is not.

The load-bearing rule, inherited from the frozen interface, is **suppression**:
a refused request returns NO numeric estimate at all, because an
ordinary-looking number is exactly what gets mistaken for a validated one.

What is refused outright (capability gates run BEFORE any inference):

* unrestricted structure discovery (``battery_structure_discovery_1_to_5`` is
  ``experimental_out_of_support``; Milestone 7 closure);
* more than five supplied factors (K_MAX = 5);
* sparse multidimensional administrations (M8 closeout: only DENSE
  known-structure 3-5-factor inference passed the confirmatory benchmark);
* multidimensional longitudinal requests (L36; unsupported in the frozen
  interface);
* instruments outside the frozen category support (more than 7 categories).

What is flagged rather than refused:

* cognition-like construct declarations (cross-domain cognitive transfer is
  NOT ESTABLISHED; D9D permanent FAIL);
* any reliability output (released as a RELATIVE RELIABILITY INDICATOR,
  ordinal use only; the interval scale is NOT calibrated on real data - L45 -
  and per the P5 preregistered EXIT_2 the posterior standard deviation is
  withheld from this public interface).

What is never exposed here at all:

* real-data factor-correlation (Phi) interpretation - NOT SUPPORTED (D9B
  replication failed both Phi gates); Phi is suppressed unless the caller
  positively declares the data synthetic;
* multimodal fusion - ``meter.score`` accepts no continuous modality, because
  incremental multimodal measurement utility is NOT ESTABLISHED (M10A/M10B).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# frozen identity pins. These are assertions, not configuration: a mismatch
# with the loaded artifacts raises rather than running a different model.
# --------------------------------------------------------------------------

#: METER_PRE_SHARE_FREEZE frozen model state (the 5E longitudinal checkpoint;
#: 42,899 parameters). Every 1D and longitudinal score comes from this state.
FROZEN_MODEL_STATE_SHA256 = "224a701b8f6981e1e3eeeb64426a9bc8ab0122d7780bb74bb606a0a630397fc8"

#: The frozen known-structure (supplied-mapping) model state, as pinned in the
#: D9B/D9E charters. Reproduced deterministically from frozen seeds (seed
#: 8100), never fitted to any dataset; the reproduction is hash-asserted.
MAPPED_MODEL_STATE_SHA256 = "8d42752009e49694def7c5dfbd77b328c28e337e85c67274058ec111dd2cf7b4"

#: The capability contract this release is pinned to. `score` refuses to run
#: against any other contract version. 3.6 is the P5 EXIT_2 relabel revision
#: (text-only: no status, boundary or numerical output changed).
CAPABILITY_CONTRACT_VERSION = "3.6"

MODEL_VERSION = "v0.6.0-beta.1"

FREEZE_TAGS = (
    "METER_FOUNDATIONAL_EVIDENCE_FREEZE",
    "METER_PRE_SHARE_FREEZE",
)

#: Verbatim warning identifier from the frozen inference (L45).
INTERVAL_SCALE_WARNING = "real_world_interval_scale_not_calibrated"

#: The human-readable half of the L45 warning, attached to every reliability
#: output this API returns.
RELATIVE_RELIABILITY_EXPLANATION = (
    "These values are a RELATIVE RELIABILITY INDICATOR, usable ORDINALLY "
    "only: a higher value genuinely means less reliable (within-participant "
    "uncertainty-error r 0.86-0.91 across three real datasets), but no "
    "absolute magnitude is claimed. The underlying interval scale is not "
    "calibrated - predicted widths ran 2x to 10x the empirical variation on "
    "real data (L45), and the preregistered truth-referenced P5 study found "
    "conditional coverage outside its acceptance band in 10 of 12 design "
    "strata with directionally different miscalibration across response "
    "formats, so no single global scalar correction is supported. Per the "
    "P5 EXIT_2 rule the posterior standard deviation is withheld from this "
    "interface; the values are within-result midrank percentiles of the "
    "withheld widths and are not comparable across separate calls. "
    "Prohibited uses: confidence intervals, coverage intervals, literal "
    "error bars, probability statements, and threshold-based individual "
    "decisions."
)

#: The largest number of supplied factors the frozen backbone expresses.
K_MAX = 5

#: The frozen decoder's category support (CMAX in the model features).
MAX_CATEGORIES = 7

#: Construct-declaration substrings that trigger the D9D not-established flag.
COGNITION_MARKERS = (
    "cognit",
    "intelligen",
    "ability",
    "aptitude",
    "reasoning",
    "iq",
    "memory",
    "executive function",
)

_VALID_PROVENANCE = ("synthetic", "real", "undeclared")


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RelativeReliabilityIndicator:
    """The model's reliability output, released under the P5 EXIT_2 relabel.

    The posterior standard deviation is WITHHELD from this public interface
    (P5 preregistered EXIT_2; it is preserved internally, unmodified, for
    computation and provenance). ``values`` are scale-free midrank
    percentiles in (0, 1] of the withheld widths, computed within this
    result only - per factor for multidimensional results. Higher means
    LESS reliable relative to the other estimates in the same call. They
    carry no unit and no absolute magnitude, are not comparable across
    separate calls, and the warning travels with the values so it cannot be
    dropped on the way to a user.
    """

    values: np.ndarray
    label: str = "relative reliability indicator"
    scale: str = "ordinal"
    warning: str = INTERVAL_SCALE_WARNING
    explanation: str = RELATIVE_RELIABILITY_EXPLANATION
    permitted_uses: tuple[str, ...] = (
        "ordinal_uncertainty_ranking",
        "comparative_diagnostic",
    )
    prohibited_uses: tuple[str, ...] = (
        "confidence_interval",
        "coverage_interval",
        "literal error_bar",
        "probability statement",
        "threshold-based individual decision",
    )

    def ranks(self) -> np.ndarray:
        """Midranks (1..n, ties averaged) recovered from the percentiles."""
        array = np.asarray(self.values, dtype=float)
        return array * array.shape[0]


def _midrank_percentiles(widths: np.ndarray) -> np.ndarray:
    """Column-wise midrank percentiles in (0, 1]; exact ties share one value.

    This is the ONLY transformation between the withheld posterior widths
    and the released indicator: rank-preserving, scale-destroying, no
    fitted parameter of any kind.
    """
    array = np.asarray(widths, dtype=float)
    flat = array.reshape(array.shape[0], -1)
    out = np.empty_like(flat)
    n = flat.shape[0]
    for column in range(flat.shape[1]):
        values = flat[:, column]
        order = np.argsort(values, kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        uniques, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        rank_sums = np.zeros(uniques.shape[0], dtype=float)
        np.add.at(rank_sums, inverse, ranks)
        out[:, column] = (rank_sums / counts)[inverse] / n
    return out.reshape(array.shape)


def _relative_reliability(widths: Any) -> RelativeReliabilityIndicator:
    """Build the released indicator; the raw widths do not leave this frame."""
    raw = widths.detach().numpy() if hasattr(widths, "detach") else np.asarray(widths)
    return RelativeReliabilityIndicator(values=_midrank_percentiles(raw))


@dataclass(frozen=True)
class MeterResult:
    """One scoring outcome: either estimates with their contract, or a refusal.

    ``refusal_reason`` is ``None`` exactly when ``scores`` is present. A
    refused result carries no numeric estimate anywhere - what was withheld is
    named in ``suppressed`` with its reason.
    """

    scores: np.ndarray | None
    relative_reliability: RelativeReliabilityIndicator | None
    support_status: dict[str, str]
    warnings: tuple[str, ...]
    refusal_reason: str | None
    model_version: str
    provenance: dict[str, Any]
    capability: str | None = None
    status: str | None = None
    #: Design-specific capability status for longitudinal results (e.g.
    #: ``supported_experimental``, ``ordinal_only``); distinct from ``status``.
    capability_status: str | None = None
    suppressed: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.refusal_reason is not None


# --------------------------------------------------------------------------
# frozen-stack access (lazy; hash-verified by the loaders themselves)
# --------------------------------------------------------------------------

_MODELS: dict[str, Any] | None = None
_MAPPED_MODEL: Any | None = None


def _capability_contract() -> dict[str, Any]:
    from human_measurement.workbench.protocols import REAL_WORLD_CAPABILITY

    version = str(REAL_WORLD_CAPABILITY.get("contract_version"))
    if version != CAPABILITY_CONTRACT_VERSION:
        raise RuntimeError(
            f"meter {MODEL_VERSION} is pinned to capability contract "
            f"{CAPABILITY_CONTRACT_VERSION}; found {version}. Refusing to run "
            "against a contract this release was not adjudicated under."
        )
    return REAL_WORLD_CAPABILITY


def capability_support_map() -> dict[str, str]:
    """Per-capability status, verbatim from the frozen v3.6 contract.

    Every entry that carries a status is included - the supported ones and
    the not-established ones alike, because one map that omitted the negative
    entries would let the strong results launder the weak ones.
    """
    contract = _capability_contract()
    statuses = {
        name: str(entry["status"])
        for name, entry in contract.items()
        if isinstance(entry, dict) and "status" in entry
    }
    statuses["contract_version"] = str(contract["contract_version"])
    return statuses


#: Directory holding the frozen weight files. Defaults to the paper
#: package's ``checkpoints/`` directory; a caller may point it at another
#: verified checkpoint directory BEFORE the first model load. This is a
#: paper-reproducibility loader edit (path resolution only); the loaded
#: state is hash-asserted below exactly as in the frozen reference loaders.
_CHECKPOINT_DIR: Path | None = None


def _checkpoint_dir() -> Path:
    if _CHECKPOINT_DIR is not None:
        return _CHECKPOINT_DIR
    return Path(__file__).resolve().parents[1] / "checkpoints"


def _models() -> dict[str, Any]:
    """The frozen 5E scoring artifact, hash-verified at load.

    Research-package trim (identical to the paper-reproducibility loader):
    the research loader (``ingestion.cli._load_models``) additionally loads
    the DIF, linking and fusion artifacts, none of which is reachable
    through ``meter.score``; this loader replicates the frozen
    ``phase5e1_run.frozen_longitudinal()`` behaviour exactly (state key
    ``best_state``, sorted-key SHA-256 assertion, ``gate_floor = 0.0``, no
    ``eval()`` call) for the single artifact the scoring path uses.
    Fixture tests assert bit-identical outputs against the research loader.
    """
    global _MODELS
    if _MODELS is None:
        import torch

        from human_measurement.model.longitudinal import (
            EXPERIMENTAL_MODEL_5E,
            LongitudinalAmortizedGrm,
        )

        if EXPERIMENTAL_MODEL_5E["model_state_sha256"] != FROZEN_MODEL_STATE_SHA256:
            raise RuntimeError(
                "the frozen 5E state pin does not match this release's pin; "
                "refusing to score with an unverified model"
            )
        checkpoint = _checkpoint_dir() / "phase5e-arm5_combined.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"frozen checkpoint not found at {checkpoint}; obtain the "
                "checkpoint distribution and verify it against the bundle "
                "hashes.json before scoring"
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["best_state"]
        if _state_sha256(state) != FROZEN_MODEL_STATE_SHA256:
            raise RuntimeError(
                "the 5E checkpoint on disk fails the frozen hash assertion; "
                "refusing to score with an unverified model"
            )
        model = LongitudinalAmortizedGrm(hidden=64)
        model.load_state_dict(state)
        model.gate_floor = 0.0
        _MODELS = {"longitudinal": model}
    return _MODELS


def _state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].numpy().tobytes())
    return digest.hexdigest()


def _mapped_model() -> Any:
    """The frozen known-structure model, loaded from the bundle, verified.

    The M8/D9 design carries no fitted checkpoint for this model: the
    research repository reproduces it deterministically from frozen seeds
    (never fitted to any dataset) and hash-asserts the resulting state
    against the charter pin. This package distributes that reproduced,
    hash-asserted state as a bundle weight file; the same charter pin is
    asserted here before any inference.
    """
    global _MAPPED_MODEL
    if _MAPPED_MODEL is not None:
        return _MAPPED_MODEL

    import torch

    from human_measurement.mapped.model import KnownStructureGrm

    # Research-package trim (identical to the paper-reproducibility loader):
    # the research package reproduces this state from its frozen seed
    # procedure when the cache is absent. The reproduction code (trainer +
    # world generator) is not part of the released inference surface, so
    # this package distributes the reproduced, hash-asserted state as a
    # checkpoint instead. The digest below is the same charter pin the
    # research reproduction is asserted against (D9B/D9E charters).
    cache = _checkpoint_dir() / "mapped_known_structure_seed8100.pt"
    if not cache.exists():
        raise FileNotFoundError(
            f"frozen known-structure state not found at {cache}; obtain the "
            "checkpoint distribution and verify it against "
            "checkpoints/CHECKSUMS.sha256 before scoring"
        )
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    state = payload["state"]
    if _state_sha256(state) != MAPPED_MODEL_STATE_SHA256:
        raise RuntimeError(
            f"known-structure state at {cache} fails the frozen hash "
            "assertion; refusing to score with an unverified model"
        )
    model = KnownStructureGrm(hidden=64)
    model.load_state_dict(state)
    model.eval()
    _MAPPED_MODEL = model
    return model


def _base_provenance() -> dict[str, Any]:
    return {
        "package": "meter-psychometrics 0.9.0rc1",
        "model_version": MODEL_VERSION,
        "model_state_sha256": FROZEN_MODEL_STATE_SHA256,
        "capability_contract_version": CAPABILITY_CONTRACT_VERSION,
        "freeze_tags": list(FREEZE_TAGS),
        "per_dataset_fitting": "none - frozen forward inference only",
    }


def _full_provenance() -> dict[str, Any]:
    from human_measurement.interface import frozen_provenance
    from human_measurement.workbench.protocols import protocol_sha256

    provenance = _base_provenance()
    provenance["capability_contract_sha256"] = protocol_sha256(_capability_contract())
    provenance["frozen_provenance"] = frozen_provenance()
    return provenance


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


def _as_response_array(responses: Any) -> np.ndarray:
    array = np.asarray(responses, dtype=float)
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3:
        raise ValueError(
            f"responses must be (persons, items) or (persons, waves, items); got shape "
            f"{np.asarray(responses).shape}"
        )
    observed = np.isfinite(array) & (array >= 0)
    if observed.any():
        values = array[observed]
        if not np.allclose(values, np.round(values)):
            raise ValueError(
                "observed responses must be integer category codes (0-based); "
                "found non-integer values"
            )
    return array


def _resolve_categories(
    array: np.ndarray, observed: np.ndarray, metadata: dict[str, Any]
) -> np.ndarray:
    n_items = array.shape[2]
    declared = metadata.get("n_categories")
    if declared is not None:
        if np.isscalar(declared):
            categories = np.full(n_items, int(declared), dtype=np.int64)
        else:
            categories = np.asarray(declared, dtype=np.int64)
            if categories.shape != (n_items,):
                raise ValueError(
                    f"metadata['n_categories'] must be a scalar or length-{n_items} sequence"
                )
    else:
        categories = np.empty(n_items, dtype=np.int64)
        for j in range(n_items):
            cells = array[:, :, j][observed[:, :, j]]
            categories[j] = max(int(cells.max()) + 1, 2) if cells.size else 2
    if (categories < 2).any():
        raise ValueError("every item needs at least two categories")
    return categories


def _canonical_dataset(
    array: np.ndarray,
    observed: np.ndarray,
    categories: np.ndarray,
    wave_time: np.ndarray,
) -> Any:
    from human_measurement.ingestion.canonical import CanonicalDataset

    n_persons, n_waves, n_items = array.shape
    responses = np.where(observed, np.nan_to_num(array, nan=0.0), -1.0).astype(np.int64)
    return CanonicalDataset(
        responses=responses,
        response_mask=observed,
        n_categories=categories,
        item_instrument=np.zeros(n_items, dtype=np.int64),
        item_names=tuple(f"item_{j:03d}" for j in range(n_items)),
        instrument_names=("supplied",),
        participant_ids=tuple(f"row_{i}" for i in range(n_persons)),
        wave_time=wave_time,
        wave_labels=tuple(f"w{t + 1}" for t in range(n_waves)),
        setting_of_person=np.zeros(n_persons, dtype=np.int64),
        setting_labels=("all",),
    )


def _wave_times(array: np.ndarray, observed: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    n_persons, n_waves, _ = array.shape
    declared = metadata.get("wave_times_days")
    if declared is None:
        times = np.full((n_persons, n_waves), np.nan)
        times[:, 0] = 0.0 if n_waves == 1 else times[:, 0]
    else:
        declared_array = np.asarray(declared, dtype=float)
        if declared_array.shape == (n_waves,):
            times = np.tile(declared_array, (n_persons, 1))
        elif declared_array.shape == (n_persons, n_waves):
            times = declared_array.copy()
        else:
            raise ValueError(
                f"metadata['wave_times_days'] must have shape ({n_waves},) or "
                f"({n_persons}, {n_waves})"
            )
    wave_observed = observed.any(axis=2)
    times[~wave_observed] = np.nan
    if n_waves == 1:
        times = np.where(wave_observed, np.nan_to_num(times, nan=0.0), np.nan)
    return times


def _parse_structure(structure: Any, n_items: int) -> tuple[tuple[int, ...], int, tuple[str, ...]]:
    labels: tuple[str, ...] | None = None
    if isinstance(structure, dict):
        raw_labels = structure.get("labels") or structure.get("factor_labels")
        labels = tuple(str(x) for x in raw_labels) if raw_labels is not None else None
        structure = structure.get("item_factor")
        if structure is None:
            raise ValueError("a structure dict must carry 'item_factor'")
    item_factor = tuple(int(f) for f in np.asarray(structure).ravel())
    if len(item_factor) != n_items:
        raise ValueError(
            f"structure assigns {len(item_factor)} items; responses carry {n_items} items"
        )
    if min(item_factor) < 0:
        raise ValueError("structure factor indices must be 0-based and non-negative")
    k = max(item_factor) + 1
    if labels is None:
        labels = tuple(f"factor_{i + 1}" for i in range(k))
    if len(labels) != k:
        raise ValueError(f"{len(labels)} labels supplied for k={k} factors")
    return item_factor, k, labels


def _discovery_requested(metadata: dict[str, Any]) -> bool:
    if bool(metadata.get("discover_structure")):
        return True
    for key in ("n_factors", "n_dimensions", "dimensions"):
        value = metadata.get(key)
        if value is not None and int(value) > 1:
            return True
    return False


def _construct_flags(metadata: dict[str, Any]) -> tuple[str, ...]:
    construct = str(metadata.get("construct", "") or "").lower()
    if not construct:
        return ()
    if any(marker in construct for marker in COGNITION_MARKERS):
        return (
            "meter:construct_transfer_not_established - the declared construct "
            f"({metadata.get('construct')!r}) is cognition-like; cross-domain "
            "cognitive transfer is NOT ESTABLISHED (D9D permanent FAIL: median "
            "0.9023 vs the 0.93 gate on ICAR-16). Scores are emitted for "
            "methodological comparison only and support no substantive claim "
            "about this construct.",
        )
    return ()


# --------------------------------------------------------------------------
# refusal construction
# --------------------------------------------------------------------------


def _refusal(
    reason: str,
    withheld: dict[str, str],
    extra_warnings: tuple[str, ...] = (),
    decisions: dict[str, Any] | None = None,
) -> MeterResult:
    diagnostics: dict[str, Any] = {}
    if decisions:
        diagnostics["output_decisions"] = {
            name: decision.as_dict() for name, decision in decisions.items()
        }
    return MeterResult(
        scores=None,
        relative_reliability=None,
        support_status=capability_support_map(),
        warnings=tuple(extra_warnings),
        refusal_reason=reason,
        model_version=MODEL_VERSION,
        provenance=_base_provenance(),
        capability=None,
        status="unsupported",
        suppressed=dict(withheld),
        diagnostics=diagnostics,
    )


_STRUCTURE_DISCOVERY_REFUSAL = (
    "structure_discovery_refused: unrestricted 1-to-5-factor structure "
    "discovery is 'experimental_out_of_support' (Milestone 7 closure; "
    "REAL_WORLD_CAPABILITY['battery_structure_discovery_1_to_5']). Four "
    "preserved smoke runs reported every world as one factor; reporting an "
    "automatically discovered factor count as a finding is a prohibited use. "
    "METER scores a structure you SUPPLY (structure=item-to-factor mapping); "
    "it does not discover one."
)

_SPARSE_MULTIDIM_REFUSAL = (
    "sparse_multidimensional_refused: multidimensional scoring of sparse or "
    "incomplete administrations is unsupported (M8 closeout: only DENSE "
    "known-structure 3-5-factor inference passed the confirmatory benchmark; "
    "sparse multidimensional administration is explicitly out of support). "
    "Provide fully observed responses, or score each factor separately as a "
    "1D instrument."
)

_K_TOO_LARGE_REFUSAL = (
    "k_gt_5_refused: the frozen known-structure backbone expresses at most "
    "K=5 supplied factors (K_MAX=5; M8B). Larger batteries must be decomposed "
    "into instruments of at most five factors, scored separately."
)

_MULTIDIM_LONGITUDINAL_REFUSAL = (
    "multidimensional_longitudinal_refused: multidimensional longitudinal "
    "inference degrades and is not validated (L36); the frozen interface "
    "routes it as unsupported and METER does not offer it."
)

_SINGLE_WAVE_LONGITUDINAL_REFUSAL = (
    "single_wave_longitudinal_refused: a longitudinal request needs responses "
    "shaped (persons, waves, items) with at least two observed waves. A "
    "single wave carries no within-person information to separate."
)

_CATEGORY_SUPPORT_REFUSAL = (
    "category_support_refused: the frozen decoder supports items with 2 to "
    f"{MAX_CATEGORIES} response categories; this instrument exceeds that. No "
    "capability exists for it, so no score is emitted."
)


# --------------------------------------------------------------------------
# the public entry point
# --------------------------------------------------------------------------


def score(
    responses: Any,
    structure: Any = None,
    *,
    longitudinal: bool = False,
    metadata: dict[str, Any] | None = None,
) -> MeterResult:
    """Score an instrument with the frozen METER model, or refuse.

    Parameters
    ----------
    responses:
        ``(persons, items)`` or ``(persons, waves, items)`` integer category
        codes, 0-based. ``NaN`` (or any negative value) marks an unobserved
        cell.
    structure:
        ``None`` for unidimensional scoring, or a supplied item-to-factor
        mapping for known-structure multidimensional scoring: a length-J
        sequence of 0-based factor indices, or a dict with ``item_factor``
        and optional ``labels``. Structure is an INPUT here, never an output.
    longitudinal:
        Declare a longitudinal trait request. Requires at least two waves;
        the frozen operating boundaries (6B.2.3) are enforced on the result.
    metadata:
        Optional declarations. Recognised keys: ``construct`` (str),
        ``data_provenance`` (``"synthetic"`` | ``"real"`` | ``"undeclared"``,
        fail-safe default ``"undeclared"``), ``n_categories`` (int or
        per-item sequence), ``wave_times_days`` (length-T sequence or (P, T)
        array), ``proxy_indicator_supplied``, ``repeated_item_consistency``,
        ``coding_consistent_across_waves`` (bools, longitudinal routing),
        ``discover_structure`` / ``n_factors`` (a discovery ask - refused).

    Returns
    -------
    MeterResult
        Estimates with their usage contract, or a refusal that names what was
        withheld and why. ``result.refused`` distinguishes the two.
    """
    metadata = dict(metadata or {})
    provenance_declared = str(metadata.get("data_provenance", "undeclared"))
    if provenance_declared not in _VALID_PROVENANCE:
        raise ValueError(
            f"metadata['data_provenance'] must be one of {_VALID_PROVENANCE}; "
            f"got {provenance_declared!r}"
        )

    array = _as_response_array(responses)
    observed = np.isfinite(array) & (array >= 0)
    n_waves = array.shape[1]
    construct_flags = _construct_flags(metadata)

    # ---- capability gates. All of them run BEFORE any inference. ----------
    if structure is None and _discovery_requested(metadata):
        from human_measurement.capability_router import (
            DesignDescriptor,
            supplied_structure_decisions,
        )

        try:
            categories_preview = _resolve_categories(array, observed, metadata)
        except ValueError:
            categories_preview = np.asarray([], dtype=np.int64)
        discovery_descriptor = DesignDescriptor(
            n_participants=int(array.shape[0]),
            n_waves=int(array.shape[1]),
            n_items=int(array.shape[2]),
            response_category_counts=tuple(int(c) for c in categories_preview),
            observed_cell_fraction=float(observed.mean()),
            n_constructs=1,
            latent_dimensionality=None,
            factor_count=None,
            supplied_structure_present=False,
            longitudinal_design=bool(longitudinal or n_waves > 1),
            auxiliary_modality_present=False,
            requested_output_types=("item_to_factor_structure", "factor_count_discovery"),
            data_provenance=provenance_declared,
        )
        discovery_decisions = supplied_structure_decisions(discovery_descriptor)
        return _refusal(
            _STRUCTURE_DISCOVERY_REFUSAL,
            {
                "factor_count": "structure discovery out of support (M7 closure)",
                "item_assignment": "structure discovery out of support (M7 closure)",
                "person_score": "no validated capability answers this request",
            },
            extra_warnings=construct_flags,
            decisions={
                name: discovery_decisions[name]
                for name in ("item_to_factor_structure", "factor_count_discovery")
            },
        )

    categories = _resolve_categories(array, observed, metadata)
    if (categories > MAX_CATEGORIES).any():
        return _refusal(
            _CATEGORY_SUPPORT_REFUSAL,
            {"person_score": "instrument outside the frozen category support"},
            extra_warnings=construct_flags,
        )

    if structure is not None:
        from human_measurement.capability_router import (
            SupportAction,
            validate_supplied_structure,
        )

        item_factor, k, labels = _parse_structure(structure, array.shape[2])
        requested_k = next(
            (
                metadata[key]
                for key in ("n_factors", "n_dimensions", "dimensions")
                if metadata.get(key) is not None
            ),
            None,
        )
        try:
            validate_supplied_structure(
                array.shape[2],
                item_factor,
                None if requested_k is None else int(requested_k),
            )
        except ValueError as error:
            return _refusal(
                f"structure_inconsistent_refused: {error}",
                {"person_score": "the supplied structure is internally inconsistent"},
                extra_warnings=construct_flags,
            )
        if k >= 2:
            # Output-specific authorization: the canonical router evaluates
            # every predicate executably and its decision drives emission.
            # The same input may permit factor scores while refusing Phi,
            # structure discovery and factor-count discovery.
            decisions = _supplied_structure_decisions(
                array, observed, categories, k, longitudinal, provenance_declared
            )
            score_decision = decisions["factor_scores"]
            if score_decision.action is SupportAction.REFUSE:
                refusal_text = {
                    "predicate_failed:supplied_k_within_frozen_backbone": _K_TOO_LARGE_REFUSAL,
                    "predicate_failed:multidimensional_request_is_cross_sectional": (
                        _MULTIDIM_LONGITUDINAL_REFUSAL
                    ),
                    "predicate_failed:dense_observation_required_for_multidimensional_scoring": (
                        _SPARSE_MULTIDIM_REFUSAL
                    ),
                    "predicate_failed:response_categories_within_decoder_support": (
                        _CATEGORY_SUPPORT_REFUSAL
                    ),
                }[score_decision.refusal_code]
                withheld = {
                    _K_TOO_LARGE_REFUSAL: {
                        "person_score": f"K={k} exceeds the frozen K_MAX={K_MAX}"
                    },
                    _MULTIDIM_LONGITUDINAL_REFUSAL: {
                        "trait": "unsupported combination (L36)",
                        "state": "unsupported (L36)",
                    },
                    _SPARSE_MULTIDIM_REFUSAL: {
                        "person_score": "sparse multidimensional administration out of support"
                    },
                    _CATEGORY_SUPPORT_REFUSAL: {
                        "person_score": "instrument outside the frozen category support"
                    },
                }[refusal_text]
                return _refusal(
                    refusal_text,
                    withheld,
                    extra_warnings=construct_flags,
                    decisions=decisions,
                )
            return _score_known_structure(
                array,
                observed,
                categories,
                item_factor,
                k,
                labels,
                provenance_declared,
                construct_flags,
                decisions,
            )
        # k == 1 is the unidimensional case stated as a trivial mapping.

    if longitudinal and n_waves < 2:
        return _refusal(
            _SINGLE_WAVE_LONGITUDINAL_REFUSAL,
            {"trait": "no repeated observation supplied"},
            extra_warnings=construct_flags,
        )

    return _score_unidimensional(
        array, observed, categories, metadata, provenance_declared, construct_flags
    )


def _supplied_structure_decisions(
    array: np.ndarray,
    observed: np.ndarray,
    categories: np.ndarray,
    k: int,
    longitudinal: bool,
    provenance_declared: str,
) -> dict[str, Any]:
    """Canonical per-output router decisions for a supplied-structure request."""
    from human_measurement.capability_router import (
        DesignDescriptor,
        supplied_structure_decisions,
    )

    descriptor = DesignDescriptor(
        n_participants=int(array.shape[0]),
        n_waves=int(array.shape[1]),
        n_items=int(array.shape[2]),
        response_category_counts=tuple(int(c) for c in categories),
        observed_cell_fraction=float(observed.mean()),
        n_constructs=int(k),
        latent_dimensionality=int(k),
        factor_count=int(k),
        supplied_structure_present=True,
        longitudinal_design=bool(longitudinal or array.shape[1] > 1),
        auxiliary_modality_present=False,
        requested_output_types=("factor_scores", "factor_correlation_phi"),
        data_provenance=provenance_declared,
    )
    return supplied_structure_decisions(descriptor)


# --------------------------------------------------------------------------
# execution paths (reached only after every gate has passed)
# --------------------------------------------------------------------------


def _score_unidimensional(
    array: np.ndarray,
    observed: np.ndarray,
    categories: np.ndarray,
    metadata: dict[str, Any],
    provenance_declared: str,
    construct_flags: tuple[str, ...],
) -> MeterResult:
    import torch

    from human_measurement.ingestion.routing import to_world_tensors
    from human_measurement.integrated import infer
    from human_measurement.interface import InferenceRequest

    n_waves = array.shape[1]
    wave_time = _wave_times(array, observed, metadata)
    dataset = _canonical_dataset(array, observed, categories, wave_time)

    design = None
    meter_flags: list[str] = list(construct_flags)
    if n_waves >= 2:
        from human_measurement.longitudinal_design import describe_from_canonical

        design = describe_from_canonical(
            dataset,
            proxy_indicator_supplied=metadata.get("proxy_indicator_supplied"),
            repeated_item_consistency=metadata.get("repeated_item_consistency"),
            coding_consistent_across_waves=metadata.get("coding_consistent_across_waves"),
        )
        meter_flags.append(
            "meter:longitudinal_route - repeated waves route to the "
            "longitudinal trait capability; the 6B.2.3 operating boundaries "
            "are enforced on this result"
        )

    request = InferenceRequest(
        tensors=to_world_tensors(dataset),
        data_provenance=provenance_declared,
        longitudinal_design=design,
    )
    with torch.no_grad():
        result = infer(request, _models())

    if result.capability == "longitudinal_trait_state":
        scores = result.outputs["trait"].detach().numpy()
        widths = result.outputs.get("level_sd")
    else:
        scores = result.outputs["person_score"].detach().numpy().ravel()
        widths = result.outputs.get("person_sd")

    # P5 EXIT_2: the posterior widths stay in this frame; only the
    # scale-free relative reliability indicator leaves it.
    reliability = None
    if widths is not None:
        reliability = _relative_reliability(widths)

    diagnostics: dict[str, Any] = {"route": list(result.route)}
    if result.output_decisions:
        diagnostics["output_decisions"] = [
            decision.as_dict() if hasattr(decision, "as_dict") else decision
            for decision in result.output_decisions
        ]
    if result.design is not None:
        diagnostics["longitudinal_design_routing"] = result.design
    if result.capability == "longitudinal_trait_state":
        diagnostics["level"] = result.outputs["level"].detach().numpy()
        diagnostics["state_contract"] = (
            "wave-specific state estimates are diagnostics only under the "
            "frozen longitudinal contracts; see support_status and warnings"
        )

    return MeterResult(
        scores=scores,
        relative_reliability=reliability,
        support_status=capability_support_map(),
        warnings=(*result.warnings, *meter_flags),
        refusal_reason=None,
        model_version=MODEL_VERSION,
        provenance=_full_provenance(),
        capability=result.capability,
        status=str(result.status.value),
        capability_status=result.capability_status,
        suppressed=dict(result.suppressed),
        diagnostics=diagnostics,
    )


def _score_known_structure(
    array: np.ndarray,
    observed: np.ndarray,
    categories: np.ndarray,
    item_factor: tuple[int, ...],
    k: int,
    labels: tuple[str, ...],
    provenance_declared: str,
    construct_flags: tuple[str, ...],
    decisions: dict[str, Any],
) -> MeterResult:
    import torch

    from human_measurement.ingestion.routing import to_world_tensors
    from human_measurement.mapped.mapping import SuppliedMapping
    from human_measurement.mapped.model import factor_adequacy

    wave_time = np.zeros((array.shape[0], 1))
    dataset = _canonical_dataset(array, observed, categories, wave_time)
    mapping = SuppliedMapping(
        item_factor=item_factor,
        k=k,
        labels=labels,
        provenance="user_supplied:meter.score",
    )
    model = _mapped_model()
    with torch.no_grad():
        forward = model(to_world_tensors(dataset), mapping)

    scores = forward.mu.detach().numpy()
    # P5 EXIT_2: forward.sd is withheld; only its within-result midrank
    # percentiles are released.
    reliability = _relative_reliability(forward.sd)
    adequacy = factor_adequacy(forward, mapping, observed[:, 0, :])

    warnings: list[str] = list(construct_flags)
    if provenance_declared != "synthetic":
        warnings.append(INTERVAL_SCALE_WARNING)
    if k == 2:
        warnings.append(
            "meter:k2_outside_confirmatory_range - the M8B confirmatory "
            "benchmark established K=3..5; a two-factor mapping is expressible "
            "but carries no confirmatory evidence of its own"
        )

    suppressed: dict[str, str] = {}
    diagnostics: dict[str, Any] = {
        "factor_labels": list(labels),
        "factor_adequacy": [
            {
                "label": entry.label,
                "status": entry.status,
                "reliability": entry.reliability,
                "n_items": entry.n_items,
            }
            for entry in adequacy
        ],
        "evidence_basis": (
            "M8B_PASS (dense known-structure K=3-5 synthetic confirmatory); "
            "D9E_PASS (IPIP-FFM cross-instrument transfer, median 0.9791); "
            "D9B score gates passed (0.924-0.975) with BOTH Phi gates failed "
            "(REPLICATION_FAIL)"
        ),
    }
    from human_measurement.capability_router import SupportAction

    diagnostics["output_decisions"] = {
        name: decision.as_dict() for name, decision in decisions.items()
    }
    phi_decision = decisions["factor_correlation_phi"]
    if phi_decision.action is SupportAction.LIMIT:
        phi, shrink_weight = forward.positive_definite_phi()
        diagnostics["factor_correlation_phi"] = phi.detach().numpy()
        diagnostics["phi_shrink_weight"] = shrink_weight
    else:
        suppressed["factor_correlation_phi"] = (
            "real-data Phi interpretation is NOT SUPPORTED: the D9B locked "
            "replication failed both frozen Phi gates (max 0.2271 vs 0.2; "
            "mean 0.1333 vs 0.12; verdict REPLICATION_FAIL). Phi is exposed "
            "only for positively declared synthetic data."
        )

    return MeterResult(
        scores=scores,
        relative_reliability=reliability,
        support_status=capability_support_map(),
        warnings=tuple(warnings),
        refusal_reason=None,
        model_version=MODEL_VERSION,
        provenance={
            **_full_provenance(),
            "mapped_model_state_sha256": MAPPED_MODEL_STATE_SHA256,
        },
        capability="known_structure_multidimensional_3_5",
        status="supported_experimental",
        suppressed=suppressed,
        diagnostics=diagnostics,
    )
