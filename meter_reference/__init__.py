"""METER paper-reproducibility reference API.

The smallest usable surface over the frozen METER paper models:

    from meter_reference import load_model

    model = load_model("meter-1d-paper")
    result = model.score(X, mask=M)

    model = load_model("meter-multidim-paper")
    result = model.score(X, mask=M, factor_map=A)

``result`` is a :class:`meter_reference.api.MeterResult`: frozen estimates with
their usage contract, or a refusal naming what was withheld and why. All
numbers come from the frozen checkpoints identified in ``paper_manifest.json``;
nothing here trains, fits, recalibrates, or updates a parameter, and the test
suite asserts that scoring performs no model-parameter update.
"""

from __future__ import annotations

from meter_reference.api import (
    CAPABILITY_CONTRACT_VERSION,
    FROZEN_MODEL_STATE_SHA256,
    INTERVAL_SCALE_WARNING,
    MAPPED_MODEL_STATE_SHA256,
    MODEL_VERSION,
    MeterResult,
    RelativeReliabilityIndicator,
    capability_support_map,
    score,
)
from meter_reference.inference import PaperModel, load_model

__version__ = "0.1.0"

__all__ = [
    "CAPABILITY_CONTRACT_VERSION",
    "FROZEN_MODEL_STATE_SHA256",
    "INTERVAL_SCALE_WARNING",
    "MAPPED_MODEL_STATE_SHA256",
    "MODEL_VERSION",
    "MeterResult",
    "RelativeReliabilityIndicator",
    "PaperModel",
    "capability_support_map",
    "load_model",
    "score",
]
