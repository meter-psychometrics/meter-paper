"""Frozen-model handles for the two METER paper pathways.

``load_model`` returns a handle whose ``score`` performs frozen inference
through the exact code path the paper's numbers came from (``meter.score``
semantics: capability gates first, then the frozen forward pass). The handle
never exposes training hooks, optimizer state, or gradient updates — there is
nothing of that kind in this package to expose.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from meter_reference import api

#: The models the paper reports. Keys are the public identifiers.
PAPER_MODELS = {
    "meter-1d-paper": {
        "description": (
            "Frozen 1-D scoring pathway (LongitudinalAmortizedGrm, 42,899 "
            "parameters, state 224a701b…): every real-data unidimensional "
            "result in Table 1 (NHANES, ESS, SHARE, SAPA scale-wise scoring)."
        ),
        "state_sha256": api.FROZEN_MODEL_STATE_SHA256,
        "structure": False,
    },
    "meter-multidim-paper": {
        "description": (
            "Frozen supplied-structure multidimensional pathway "
            "(KnownStructureGrm, 30,986 parameters, state 8d427520…): MIDUS, "
            "IPIP-50 and ICAR-16 factor scoring. Requires an externally "
            "supplied item-to-factor map; performs no structure discovery."
        ),
        "state_sha256": api.MAPPED_MODEL_STATE_SHA256,
        "structure": True,
    },
}


class PaperModel:
    """A frozen paper model handle. Scoring only; no training surface."""

    def __init__(self, name: str) -> None:
        if name not in PAPER_MODELS:
            raise KeyError(
                f"unknown paper model {name!r}; available: {sorted(PAPER_MODELS)}"
            )
        self.name = name
        self.info = dict(PAPER_MODELS[name])

    def score(
        self,
        X: Any,
        mask: Any = None,
        factor_map: Any = None,
        *,
        longitudinal: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> api.MeterResult:
        """Frozen inference on a response matrix; returns estimates or a refusal.

        ``X`` is ``(persons, items)`` or ``(persons, waves, items)`` 0-based
        integer category codes. ``mask`` (optional, same shape, boolean) marks
        OBSERVED cells; cells that are False are treated as unobserved. With no
        ``mask``, NaN or negative entries mark unobserved cells (the frozen
        convention). ``factor_map`` supplies the item-to-factor structure for
        the multidimensional pathway and must be omitted for the 1-D pathway.
        """
        wants_structure = bool(self.info["structure"])
        if wants_structure and factor_map is None:
            raise ValueError(
                f"{self.name} scores a SUPPLIED structure: pass factor_map "
                "(a length-n_items sequence of 0-based factor indices, or a "
                "dict with 'item_factor' and optional 'labels')"
            )
        if not wants_structure and factor_map is not None:
            raise ValueError(f"{self.name} is unidimensional; factor_map is not accepted")
        responses = _apply_mask(X, mask)
        return api.score(
            responses,
            factor_map,
            longitudinal=longitudinal,
            metadata=metadata,
        )


def _apply_mask(X: Any, mask: Any) -> np.ndarray:
    """Fold an explicit observed-mask into the frozen NaN convention."""
    array = np.asarray(X, dtype=float)
    if mask is None:
        return array
    observed = np.asarray(mask, dtype=bool)
    if observed.shape != array.shape:
        raise ValueError(
            f"mask shape {observed.shape} does not match responses shape {array.shape}"
        )
    return np.where(observed, array, np.nan)


def load_model(name: str) -> PaperModel:
    """Load a frozen paper model by public identifier.

    The underlying checkpoint is loaded lazily on first ``score`` call and is
    SHA-256-asserted against the frozen paper pin before any inference.
    """
    return PaperModel(name)


def load_phase3_reference():
    """The frozen Phase 3 1-D core (31,051 parameters, state 0d571638…).

    This module produced the paper's synthetic recovery claim (median r =
    0.9388 on 40 unseen worlds) and the uncertainty-calibration scalar. It is
    NOT the real-data scoring pathway — see ``PAPER_MODELS['meter-1d-paper']``
    for that — and is exposed for the synthetic-evaluation reproduction only.

    Returns ``(model, state_sha256)``; raises if the checkpoint is absent or
    fails the frozen hash assertion.
    """
    import hashlib

    import torch

    from human_measurement.model.network import AmortizedGrm

    expected = "0d571638a8eb29f64fc0202a67d72620faf2b7ec1229670bc977ab643bac36a4"
    checkpoint = api._repo_root() / "checkpoints" / "phase3-stage2-arm1_beta05.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"frozen Phase 3 checkpoint not found at {checkpoint}; obtain the "
            "checkpoint distribution and verify it against checkpoints/CHECKSUMS.sha256"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["best_state"]
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].numpy().tobytes())
    if digest.hexdigest() != expected:
        raise RuntimeError("frozen Phase 3 checkpoint hash mismatch - refusing to load")
    model = AmortizedGrm(hidden=64)
    model.load_state_dict(state)
    return model, digest.hexdigest()
