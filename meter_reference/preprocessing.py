"""The frozen preprocessing path, re-exported for inspection and reuse.

These are the exact functions ``meter_reference.api.score`` calls between a
user response matrix and the frozen model input. Nothing here is new code:
every function is the frozen implementation (see ``api.py`` and
``human_measurement.ingestion``), re-exported so a reviewer can run and read
the preprocessing step in isolation.

Semantics fixed by the frozen implementation and asserted by the tests:

* observedness — a cell is observed iff it is finite AND non-negative; the
  canonical container stores ``-1`` at unobserved cells and carries a boolean
  mask as the single source of truth;
* categories — per-item counts are declared via ``metadata['n_categories']``
  or inferred as ``max(observed)+1`` (floor 2); items are normalised to a
  shared scale with a decoder ceiling of 7 categories;
* wave times — optional ``metadata['wave_times_days']``; unobserved waves
  carry NaN.
"""

from __future__ import annotations

from human_measurement.ingestion.canonical import UNOBSERVED, CanonicalDataset
from human_measurement.ingestion.routing import to_world_tensors
from human_measurement.model.features import item_features, normalized_values, world_features
from human_measurement.model.schemas import WorldTensors

from meter_reference.api import (
    _as_response_array as as_response_array,
)
from meter_reference.api import (
    _canonical_dataset as canonical_dataset,
)
from meter_reference.api import (
    _resolve_categories as resolve_categories,
)
from meter_reference.api import (
    _wave_times as wave_times,
)
from meter_reference.inference import _apply_mask as apply_mask

__all__ = [
    "UNOBSERVED",
    "CanonicalDataset",
    "WorldTensors",
    "apply_mask",
    "as_response_array",
    "canonical_dataset",
    "item_features",
    "normalized_values",
    "resolve_categories",
    "to_world_tensors",
    "wave_times",
    "world_features",
]
