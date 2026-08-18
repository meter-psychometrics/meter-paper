"""Paper-reproducibility trim of ``battery/model.py``.

Only ``spectral_item_features`` (and the constant it depends on) is released:
it is the per-item spectral feature builder consumed by the frozen
supplied-structure model. The structure-DISCOVERY model that shares the
research file is closed (`experimental_out_of_support`, Milestone 7) and is
not part of the paper's inference surface. The function body is verbatim; the
fixture tests assert numerical identity with the frozen implementation.
"""

from __future__ import annotations

import torch

from human_measurement.model.features import normalized_values
from human_measurement.model.schemas import WorldTensors

#: verbatim: N_SLOTS = 5 in the research module; features are N_SLOTS - 1.
N_SPECTRAL_FEATURES = 4


def spectral_item_features(tensors: WorldTensors) -> torch.Tensor:
    """Per-item components of the top ``N_SPECTRAL_FEATURES`` eigenvectors.

    Same rationale as the two-slot version: which items covary together is a
    pairwise property no per-item marginal statistic can express, so without
    this the assignment head has no signal to separate equal blocks. Signs are
    canonicalised per world (largest-magnitude entry positive) - arbitrary but
    deterministic, and slot labels are exchangeable anyway.
    """
    values, mask = normalized_values(tensors)
    counts = mask.sum(dim=0).clamp(min=1).to(torch.float32)
    mean = values.sum(dim=0) / counts
    filled = torch.where(mask, values, mean[None, :].expand_as(values))
    centered = filled - filled.mean(dim=0, keepdim=True)
    sd = centered.std(dim=0).clamp(min=1e-6)
    standardized = centered / sd
    correlation = (standardized.T @ standardized) / max(values.shape[0] - 1, 1)
    correlation = 0.5 * (correlation + correlation.T)
    _, eigenvectors = torch.linalg.eigh(correlation)
    columns = []
    for offset in range(1, N_SPECTRAL_FEATURES + 1):
        position = eigenvectors.shape[1] - offset
        if position < 0:  # fewer items than requested components
            columns.append(torch.zeros_like(eigenvectors[:, 0]))
            continue
        vector = eigenvectors[:, position]
        anchor = int(vector.abs().argmax())
        if vector[anchor] < 0:
            vector = -vector
        columns.append(vector)
    scale = float(tensors.n_items) ** 0.5
    return torch.stack(columns, dim=1) * scale
