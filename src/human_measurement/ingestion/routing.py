"""Paper-reproducibility trim of ``ingestion/routing.py``.

The research module additionally implements the validated multi-capability
workflow used by the internal ingestion CLI (validation reports, route
planning, aggregate serialisation). The paper package needs exactly the
canonical-dataset-to-model-input conversion; ``to_world_tensors`` below is
verbatim.
"""

from __future__ import annotations

import numpy as np
import torch

from human_measurement.ingestion.canonical import CanonicalDataset
from human_measurement.model.schemas import WorldTensors

ROUTING_VERSION = "6b0-routing-v1"


def to_world_tensors(dataset: CanonicalDataset) -> WorldTensors:
    """Canonical dataset -> the frozen model's input container."""
    return WorldTensors(
        responses=torch.from_numpy(dataset.responses).to(torch.int64),
        mask=torch.from_numpy(dataset.response_mask),
        n_categories=torch.from_numpy(dataset.n_categories).to(torch.int64),
        item_instrument=torch.from_numpy(dataset.item_instrument).to(torch.int64),
        setting_of_person=torch.from_numpy(dataset.setting_of_person).to(torch.int64),
        wave_time=torch.from_numpy(np.nan_to_num(dataset.wave_time, nan=0.0)).to(torch.float32),
    )
