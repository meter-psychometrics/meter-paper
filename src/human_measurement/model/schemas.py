"""Typed input and output schemas for the amortized measurement model.

The central discipline of Milestone 4A lives in this module: **inference inputs
and synthetic truth are different types.** ``WorldTensors`` carries only
quantities observable in a real study; ``TruthBundle`` carries everything the
generator knows; ``ModelOutput`` is the stable inference product a later API
can pin. A function that needs truth must ask for a ``TruthBundle`` in its
signature, which makes truth flow auditable by reading call sites — and
testable, which ``tests/unit/test_model_leakage.py`` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

#: Version of the inference output contract. Bump on any field change.
#: 4.1: calibrated-uncertainty fields added (raw fields unchanged, never
#: overwritten; calibration provenance carried in metadata).
OUTPUT_SCHEMA_VERSION = "4.1"

#: Names that must never appear as ``WorldTensors`` fields. The leakage test
#: asserts against this list, so extending the input schema forces a deliberate
#: decision here rather than a quiet addition.
FORBIDDEN_INPUT_FIELDS: tuple[str, ...] = (
    "theta",
    "eta",
    "zeta",
    "discrimination",
    "thresholds",
    "loadings",
    "alpha",
    "beta",
    "dimension",
    "n_dimensions",
    "is_decoy",
    "is_anchor",
    "has_uniform_dif",
    "class_of_person",
    "missingness",
)


@dataclass(frozen=True, slots=True)
class WorldTensors:
    """One measurement world as the model is allowed to see it.

    Participants and items exist only as within-world row and column indices;
    the opaque string identifiers never enter. ``responses`` holds 0-based
    categories with ``-1`` at unobserved cells; ``mask`` is the single source
    of truth for observedness and every consumer must respect it.
    """

    responses: torch.Tensor  # (P, T, J) int64, -1 where unobserved
    mask: torch.Tensor  # (P, T, J) bool
    n_categories: torch.Tensor  # (J,) int64
    item_instrument: torch.Tensor  # (J,) int64
    setting_of_person: torch.Tensor  # (P,) int64
    wave_time: torch.Tensor  # (P, T) float32, NaN where the wave is absent

    def __post_init__(self) -> None:
        p, t, j = self.responses.shape
        checks: list[tuple[bool, str]] = [
            (self.responses.dtype == torch.int64, "responses must be int64"),
            (self.mask.dtype == torch.bool, "mask must be bool"),
            (tuple(self.mask.shape) == (p, t, j), "mask shape must match responses"),
            (tuple(self.n_categories.shape) == (j,), "n_categories must be (J,)"),
            (tuple(self.item_instrument.shape) == (j,), "item_instrument must be (J,)"),
            (tuple(self.setting_of_person.shape) == (p,), "setting_of_person must be (P,)"),
            (tuple(self.wave_time.shape) == (p, t), "wave_time must be (P, T)"),
            (
                bool((self.responses[self.mask] >= 0).all()) if self.mask.any() else True,
                "observed responses must be non-negative",
            ),
            (
                bool((self.responses[~self.mask] == -1).all()) if (~self.mask).any() else True,
                "unobserved cells must hold -1, not stale values",
            ),
            (bool((self.n_categories >= 2).all()), "every item needs at least two categories"),
        ]
        for passed, message in checks:
            if not passed:
                raise ValueError(f"WorldTensors: {message}")

    @property
    def n_persons(self) -> int:
        return int(self.responses.shape[0])

    @property
    def n_waves(self) -> int:
        return int(self.responses.shape[1])

    @property
    def n_items(self) -> int:
        return int(self.responses.shape[2])


@dataclass(frozen=True, slots=True)
class TruthBundle:
    """Everything the generator knows, aligned to ``WorldTensors`` order.

    Consumed by supervised loss terms and by evaluation. Never an argument to
    the model's forward pass — the leakage tests enforce that separation.
    """

    theta: np.ndarray  # (P, T, K_total) latent positions, decoy dim included
    eta: np.ndarray  # (P, K_total) stable traits
    n_substantive_dimensions: int
    discrimination: np.ndarray  # (J, K_total)
    thresholds: np.ndarray  # (J, Cmax-1) generator logit scale, NaN padded
    item_dimension: np.ndarray  # (J,) construct index per item
    is_decoy_item: np.ndarray  # (J,) bool
    world_seed: int
    # Milestone 5A group-comparability truth (evaluation and training losses
    # only, like everything else here). ``dif_shift`` is the generator's
    # uniform threshold DIF collapsed to one logit shift per item between the
    # two settings (setting 1 minus setting 0, category-averaged);
    # ``group_impact`` is the true latent mean difference per dimension.
    has_uniform_dif: np.ndarray | None = None  # (J,) bool
    dif_shift: np.ndarray | None = None  # (J,) logits, setting1 - setting0
    group_impact: np.ndarray | None = None  # (K_total,) setting-mean differences
    # Milestone 5E: the generator's mean threshold drift per wave, in logits,
    # relative to wave 1 (evaluation and training losses only).
    threshold_drift: np.ndarray | None = None  # (T,)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """The stable inference product (requirement H). Numpy, not tensors.

    Wave-state fields are present from day one so the schema does not change
    shape when Phase 5 fills them; until then they are ``None`` and
    ``trait_unidentified`` explains why where relevant.
    """

    schema_version: str
    # world-level structure
    dimensionality_probabilities: np.ndarray  # (2,) P(K=1), P(K=2)
    item_discrimination: np.ndarray  # (J,) positive
    item_thresholds: np.ndarray  # (J, Cmax-1), NaN beyond an item's categories
    item_loadings: np.ndarray  # (J, K_est); K_est = 1 for the prototype
    # person-level posteriors. ``person_sd`` is always the RAW model
    # uncertainty and is never overwritten; the calibrated fields live at the
    # end of the schema and are filled by the frozen calibration layer.
    person_mean: np.ndarray  # (P, K_est)
    person_sd: np.ndarray  # (P, K_est) strictly positive, raw
    # longitudinal (Phase 5)
    state_mean: np.ndarray | None
    state_sd: np.ndarray | None
    # reconstruction
    reconstruction_log_likelihood: float
    response_probabilities: np.ndarray | None  # (P, T, J, Cmax) on request
    # diagnostics
    diagnostics: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    converged: bool = True
    inference_seconds: float = 0.0
    # Frozen post-hoc calibration (schema 4.1). Raw ``person_sd`` is never
    # overwritten; these are additive. ``calibration`` carries version,
    # multiplier, source partition, model checkpoint hash and manifest hash.
    person_sd_calibrated: np.ndarray | None = None
    calibration: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != OUTPUT_SCHEMA_VERSION:
            raise ValueError(
                f"ModelOutput schema {self.schema_version!r} != {OUTPUT_SCHEMA_VERSION!r}"
            )
        if not np.all(self.person_sd > 0):
            raise ValueError("person_sd must be strictly positive everywhere")
        if self.person_sd_calibrated is not None:
            if self.calibration is None:
                raise ValueError("calibrated uncertainty requires calibration provenance")
            if not np.all(self.person_sd_calibrated > 0):
                raise ValueError("person_sd_calibrated must be strictly positive everywhere")
        probabilities = self.dimensionality_probabilities
        if probabilities.shape != (2,) or not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
            raise ValueError("dimensionality_probabilities must be a length-2 simplex")
