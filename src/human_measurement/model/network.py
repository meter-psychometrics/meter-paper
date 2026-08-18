"""The minimal amortized GRM: two set encoders and a constrained decoder.

Design per ``docs/milestone_4_architecture.md``: Deep Sets over symmetric
statistics, no attention, parameter count well under 100k. The decoder is the
graded response model itself — the networks only produce its parameters, under
hard constraints (positive discrimination via softplus, ordered thresholds via
cumulative softplus). There is no path from person representations to response
probabilities that bypasses the GRM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from human_measurement.model.features import (
    CMAX,
    ITEM_FEATURE_DIM,
    WORLD_FEATURE_DIM,
    item_features,
    normalized_values,
    world_features,
)
from human_measurement.model.schemas import WorldTensors

A_FLOOR = 0.05  # smallest admissible discrimination; sign convention anchor
DELTA_FLOOR = 0.01  # smallest admissible threshold gap; enforces strict order


@dataclass(slots=True)
class Forward:
    """One forward pass: everything the losses and the output schema need."""

    a: torch.Tensor  # (J,) positive discriminations
    b: torch.Tensor  # (J, CMAX-1) ordered thresholds; entries past C_j-1 unused
    b_valid: torch.Tensor  # (J, CMAX-1) bool validity mask
    mu: torch.Tensor  # (P,) posterior means
    logvar: torch.Tensor  # (P,) posterior log variances
    dim_logits: torch.Tensor  # (2,) K=1 vs K=2

    @property
    def sd(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)


def _mlp(dim_in: int, hidden: int, dim_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim_in, hidden), nn.SiLU(), nn.Linear(hidden, dim_out))


class AmortizedGrm(nn.Module):
    """Dataset-level and participant-level encoders around a GRM decoder."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        # Dataset-level: per-item map with a pooled-context second stage, so
        # each item's parameters can depend on the world it sits in without
        # any ordering entering.
        self.item_stage_one = _mlp(ITEM_FEATURE_DIM, hidden, hidden)
        self.item_stage_two = _mlp(2 * hidden, hidden, hidden)
        self.head_a = nn.Linear(hidden, 1)
        self.head_b_first = nn.Linear(hidden, 1)
        self.head_b_deltas = nn.Linear(hidden, CMAX - 2)
        # Participant-level: per-observation map pooled per person.
        self.person_cell = _mlp(1 + hidden, hidden, hidden)
        self.person_head = _mlp(hidden + 2, hidden, 2)
        # World-level dimensionality head (present now, trained in Phase 4).
        self.dim_head = _mlp(WORLD_FEATURE_DIM, hidden // 2, 2)

    def forward(self, tensors: WorldTensors) -> Forward:
        phi = item_features(tensors)  # (J, F) no-grad statistics
        h = self.item_stage_one(phi)
        pooled = h.mean(dim=0, keepdim=True).expand_as(h)
        h = self.item_stage_two(torch.cat([h, pooled], dim=1))  # (J, H)

        a = nn.functional.softplus(self.head_a(h).squeeze(-1)) + A_FLOOR
        first = self.head_b_first(h)  # (J, 1)
        deltas = nn.functional.softplus(self.head_b_deltas(h)) + DELTA_FLOOR
        b = torch.cat([first, first + deltas.cumsum(dim=1)], dim=1)  # (J, CMAX-1) ordered
        categories = torch.arange(1, CMAX, device=b.device)[None, :]
        b_valid = categories < tensors.n_categories[:, None]

        values, mask = normalized_values(tensors)  # (P*T, J)
        cell = self.person_cell(
            torch.cat([values[:, :, None], h[None, :, :].expand(values.shape[0], -1, -1)], dim=2)
        )  # (P*T, J, H)
        weights = mask.to(cell.dtype)[:, :, None]
        row_pooled = (cell * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        p, t = tensors.n_persons, tensors.n_waves
        person_pooled = row_pooled.reshape(p, t, -1).mean(dim=1)  # trait-only prototype
        observed_per_person = tensors.mask.reshape(p, -1).sum(dim=1).to(cell.dtype) / float(
            t * tensors.n_items
        )
        wave_share = torch.tensor(1.0 / t, dtype=cell.dtype).expand(p)
        posterior = self.person_head(
            torch.cat([person_pooled, observed_per_person[:, None], wave_share[:, None]], dim=1)
        )
        mu = posterior[:, 0]
        logvar = posterior[:, 1].clamp(min=-8.0, max=4.0)

        dim_logits = self.dim_head(world_features(tensors))
        return Forward(a=a, b=b, b_valid=b_valid, mu=mu, logvar=logvar, dim_logits=dim_logits)


def category_log_probabilities(
    a: torch.Tensor, b: torch.Tensor, n_categories: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Masked-category GRM log probabilities, shape (P, J, CMAX).

    ``P(Y >= c) = sigmoid(a * theta - b_c)`` with ordered ``b`` gives strictly
    positive category masses; invalid categories (c >= C_j) get -inf so a
    downstream gather on a valid response never touches them.
    """
    linear = a[None, :, None] * theta[:, None, None] - b[None, :, :]  # (P, J, CMAX-1)
    cumulative = torch.sigmoid(linear)
    # Force cumulative probabilities past an item's last real threshold to 0
    # so P(y = C_j - 1) closes the telescope correctly for short scales.
    categories = torch.arange(1, CMAX, device=b.device)[None, :]
    valid = (categories < n_categories[:, None])[None, :, :]
    cumulative = torch.where(valid, cumulative, torch.zeros_like(cumulative))
    ones = torch.ones_like(cumulative[:, :, :1])
    zeros = torch.zeros_like(cumulative[:, :, :1])
    padded = torch.cat([ones, cumulative, zeros], dim=2)  # (P, J, CMAX+1)
    masses = (padded[:, :, :-1] - padded[:, :, 1:]).clamp(min=1e-8)  # (P, J, CMAX)
    log_masses = masses.log()
    category_index = torch.arange(CMAX, device=b.device)[None, None, :]
    return torch.where(
        category_index < n_categories[None, :, None],
        log_masses,
        torch.full_like(log_masses, float("-inf")),
    )


def masked_log_likelihood(
    tensors: WorldTensors, forward: Forward, theta: torch.Tensor
) -> torch.Tensor:
    """Mean GRM log likelihood over observed cells, given sampled theta (P,)."""
    p, t, j = tensors.responses.shape
    log_probabilities = category_log_probabilities(
        forward.a, forward.b, tensors.n_categories, theta
    )  # (P, J, CMAX)
    expanded = log_probabilities[:, None, :, :].expand(p, t, j, CMAX)
    safe_responses = tensors.responses.clamp(min=0)
    gathered = expanded.gather(3, safe_responses[:, :, :, None]).squeeze(3)
    observed = tensors.mask
    return gathered[observed].mean() if bool(observed.any()) else gathered.sum() * 0.0


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL(q(theta) || N(0,1)) — the identification anchor for scale."""
    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).mean()


def reparameterized_sample(
    mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    noise = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    return mu + torch.exp(0.5 * logvar) * noise


def sanity_flags(forward: Forward) -> list[str]:
    """Decoder-validity warnings for the output schema; empty means healthy."""
    flags: list[str] = []
    if not bool((forward.a > 0).all()):
        flags.append("non_positive_discrimination")
    gaps = forward.b[:, 1:] - forward.b[:, :-1]
    if bool((gaps <= 0).any()):
        flags.append("unordered_thresholds")
    if not bool(torch.isfinite(forward.mu).all() & torch.isfinite(forward.logvar).all()):
        flags.append("non_finite_posterior")
    if float(forward.sd.std()) < 1e-6:
        flags.append("uncertainty_collapsed_to_constant")
    return flags


def orient_positive(mu: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The documented sign convention: discriminations positive on average.

    With softplus the sign of ``a`` is already positive, so this is an
    assertion more than a transformation — kept explicit so the convention has
    one named home. ``math`` guards against silent NaN.
    """
    if not math.isfinite(float(a.mean())):
        raise ValueError("cannot orient a non-finite discrimination vector")
    return mu, a
