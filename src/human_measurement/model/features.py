"""Deterministic order-invariant statistics computed from model inputs.

These are the "symmetric statistics" of the architecture record: every feature
is a masked moment, histogram, or spectral summary that is *algebraically*
invariant under participant permutation and equivariant under item
permutation. They are computed from ``WorldTensors`` only — this module has no
import path to truth — and under ``no_grad``, because they are inputs to the
learned components, not part of them.

Why hand-specified: at Milestone 4A scale these are close to the sufficient
statistics of a GRM world, they cost nothing to verify, and they keep the
learned networks small enough to diagnose. The ADR records the condition under
which they would be replaced by learned set encoders.
"""

from __future__ import annotations

import torch

from human_measurement.model.schemas import WorldTensors

#: Fixed feature widths, part of the model contract.
N_EIGEN_FEATURES = 4
CMAX = 7  # largest response scale in the corpus (categories 2..7)
ITEM_FEATURE_DIM = CMAX + 6  # histogram + [observed rate, mean, sd, item-rest r, C/Cmax, log P]
WORLD_FEATURE_DIM = N_EIGEN_FEATURES + 3


def normalized_values(tensors: WorldTensors) -> tuple[torch.Tensor, torch.Tensor]:
    """Responses mapped to (-1, 1) by scale position, flattened over waves.

    Returns ``(values, mask)`` of shape (P*T, J). Normalizing by each item's
    own category count is what lets a 4-point and a 7-point item share one
    embedding space.
    """
    p, t, j = tensors.responses.shape
    flat = tensors.responses.reshape(p * t, j).to(torch.float32)
    mask = tensors.mask.reshape(p * t, j)
    categories = tensors.n_categories.to(torch.float32).clamp(min=2.0)
    values = 2.0 * (flat + 0.5) / categories - 1.0
    return torch.where(mask, values, torch.zeros_like(values)), mask


@torch.no_grad()
def item_features(tensors: WorldTensors) -> torch.Tensor:
    """Per-item summary, shape (J, ITEM_FEATURE_DIM). Person-permutation invariant."""
    values, mask = normalized_values(tensors)
    rows, n_items = values.shape
    counts = mask.sum(dim=0).clamp(min=1).to(torch.float32)

    histogram = torch.zeros(n_items, CMAX)
    for category in range(CMAX):
        hits = (tensors.responses == category) & tensors.mask
        histogram[:, category] = hits.reshape(rows, n_items).sum(dim=0).to(torch.float32)
    histogram = histogram / counts[:, None]

    mean = values.sum(dim=0) / counts
    centered = torch.where(mask, values - mean[None, :], torch.zeros_like(values))
    variance = (centered**2).sum(dim=0) / counts
    sd = variance.clamp(min=1e-8).sqrt()

    # Item-rest correlation: each item's values against the masked mean of the
    # *other* items on the same person-wave row.
    row_sum = values.sum(dim=1, keepdim=True)
    row_count = mask.sum(dim=1, keepdim=True).to(torch.float32)
    rest_count = (row_count - mask.to(torch.float32)).clamp(min=1.0)
    rest_mean = (row_sum - values) / rest_count
    item_rest = torch.zeros(n_items)
    for j in range(n_items):
        observed = mask[:, j] & (row_count[:, 0] >= 2)
        if int(observed.sum()) >= 8:
            x = values[observed, j]
            y = rest_mean[observed, j]
            x = x - x.mean()
            y = y - y.mean()
            denominator = (x.norm() * y.norm()).clamp(min=1e-8)
            item_rest[j] = (x @ y) / denominator

    observed_rate = mask.to(torch.float32).mean(dim=0)
    scale = tensors.n_categories.to(torch.float32) / float(CMAX)
    log_persons = torch.full((n_items,), float(torch.log(torch.tensor(rows + 1.0))) / 10.0)
    return torch.cat(
        [
            histogram,
            observed_rate[:, None],
            mean[:, None],
            sd[:, None],
            item_rest[:, None],
            scale[:, None],
            log_persons[:, None],
        ],
        dim=1,
    )


@torch.no_grad()
def world_features(tensors: WorldTensors) -> torch.Tensor:
    """World summary, shape (WORLD_FEATURE_DIM,). Fully permutation invariant.

    The leading features are the top eigenvalue *shares* of the inter-item
    correlation matrix — the signal parallel analysis reads for
    dimensionality, provided here so the dimensionality head has something to
    learn from without pairwise learned interactions.
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
    eigenvalues = torch.linalg.eigvalsh(correlation).flip(0)
    shares = eigenvalues[:N_EIGEN_FEATURES] / eigenvalues.sum().clamp(min=1e-8)
    padded = torch.zeros(N_EIGEN_FEATURES)
    padded[: shares.shape[0]] = shares

    extras = torch.tensor(
        [
            float(torch.log(torch.tensor(float(tensors.n_persons + 1)))) / 10.0,
            float(torch.log(torch.tensor(float(tensors.n_items + 1)))) / 10.0,
            float(mask.to(torch.float32).mean()),
        ]
    )
    return torch.cat([padded, extras])
