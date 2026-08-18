"""Frozen amortized neural-psychometric model architectures (inference only).

Paper-reproducibility trim: the research package's ``__init__`` additionally
exposes the world-tensor builders and the training entry points; those are not
part of the frozen inference surface and are not released.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "human_measurement.model requires PyTorch; install it with: pip install torch"
    ) from error

from human_measurement.model.network import AmortizedGrm
from human_measurement.model.schemas import (
    FORBIDDEN_INPUT_FIELDS,
    OUTPUT_SCHEMA_VERSION,
    ModelOutput,
    TruthBundle,
    WorldTensors,
)

__all__ = [
    "FORBIDDEN_INPUT_FIELDS",
    "OUTPUT_SCHEMA_VERSION",
    "AmortizedGrm",
    "ModelOutput",
    "TruthBundle",
    "WorldTensors",
]
