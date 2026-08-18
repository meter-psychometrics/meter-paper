"""Paper-reproducibility trim of ``ingestion/canonical.py``.

The research module additionally implements the mapping-driven construction of
a canonical dataset from raw files (a polars/pydantic surface used by the
internal ingestion CLI). The paper package constructs ``CanonicalDataset``
directly from arrays, so only the canonical container itself is released. The
dataclass below is verbatim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CANONICAL_VERSION = "6b0-canonical-v1"
#: Unobserved ordinal cell marker, matching the frozen WorldTensors convention.
UNOBSERVED = -1


@dataclass(frozen=True, slots=True)
class CanonicalDataset:
    """One dataset in the model's canonical form. Containers stay separate."""

    #: (P, T, J) int64 category codes, 0-based, UNOBSERVED where absent.
    responses: np.ndarray
    #: (P, T, J) bool observedness mask - the single source of truth.
    response_mask: np.ndarray
    #: (J,) number of categories per item.
    n_categories: np.ndarray
    #: (J,) instrument index per item.
    item_instrument: np.ndarray
    item_names: tuple[str, ...]
    instrument_names: tuple[str, ...]

    #: (P,) opaque participant labels. Never emitted into reports.
    participant_ids: tuple[str, ...]
    #: (P, T) float wave times; NaN where the wave is absent for that person.
    wave_time: np.ndarray
    wave_labels: tuple[str, ...]

    #: (P,) setting index, and the label lookup behind it.
    setting_of_person: np.ndarray
    setting_labels: tuple[str, ...]
    #: variable -> (P,) integer codes, with labels alongside.
    group_codes: dict[str, np.ndarray] = field(default_factory=dict)
    group_labels: dict[str, tuple[str, ...]] = field(default_factory=dict)

    #: modality_id -> (P, T, F) values, mask, feature names, declared relation.
    continuous: dict[str, dict[str, Any]] = field(default_factory=dict)

    provenance: dict[str, Any] = field(default_factory=dict)
    version: str = CANONICAL_VERSION

    @property
    def n_participants(self) -> int:
        return int(self.responses.shape[0])

    @property
    def n_waves(self) -> int:
        return int(self.responses.shape[1])

    @property
    def n_items(self) -> int:
        return int(self.responses.shape[2])

    def structure(self) -> dict[str, Any]:
        """Shape-only summary. Contains no participant-level value."""
        return {
            "n_participants": self.n_participants,
            "n_waves": self.n_waves,
            "n_items": self.n_items,
            "n_instruments": len(self.instrument_names),
            "item_names": list(self.item_names),
            "instrument_names": list(self.instrument_names),
            "wave_labels": list(self.wave_labels),
            "n_categories": self.n_categories.tolist(),
            "observed_cells": int(self.response_mask.sum()),
            "observed_fraction": float(self.response_mask.mean())
            if self.response_mask.size
            else 0.0,
            "group_variables": sorted(self.group_codes),
            "setting_labels": list(self.setting_labels),
            "continuous_modalities": {
                key: {
                    "n_features": len(block["feature_names"]),
                    "feature_names": list(block["feature_names"]),
                    "construct_relation": block["construct_relation"],
                    "observed_fraction": float(block["mask"].mean()) if block["mask"].size else 0.0,
                }
                for key, block in self.continuous.items()
            },
            "version": self.version,
        }

    def sha256(self) -> str:
        """Hash of the canonical content, for provenance.

        Includes the arrays so a silent content change is detectable, but the
        digest itself discloses nothing.
        """
        digest = hashlib.sha256()
        digest.update(json.dumps(self.structure(), sort_keys=True).encode("utf-8"))
        for array in (
            self.responses,
            self.response_mask,
            self.n_categories,
            self.item_instrument,
            self.setting_of_person,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(np.ascontiguousarray(np.nan_to_num(self.wave_time, nan=-1.0)).tobytes())
        for key in sorted(self.continuous):
            digest.update(key.encode("utf-8"))
            digest.update(np.ascontiguousarray(self.continuous[key]["values"]).tobytes())
            digest.update(np.ascontiguousarray(self.continuous[key]["mask"]).tobytes())
        return digest.hexdigest()


class CanonicalError(ValueError):
    """Raised when a mapping cannot be applied. Carries no participant values."""
