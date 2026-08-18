"""The two-slot multidimensional amortized GRM (Phase 4).

A separate module by design: the Phase 3 one-dimensional network stays
byte-identical so the frozen reference and its calibration remain
reproducible. Everything here follows the ADR §6a: soft item-to-slot
assignment over a single positive magnitude, a differentiable activation gate
on the second slot, a shared person head across slots (label exchangeability),
and the same hard decoder constraints as the one-dimensional model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
from human_measurement.model.network import A_FLOOR, DELTA_FLOOR, _mlp
from human_measurement.model.schemas import WorldTensors

N_SLOTS = 2
N_SPECTRAL_FEATURES = 2


@torch.no_grad()
def spectral_item_features(tensors: WorldTensors) -> torch.Tensor:
    """Per-item components of the top-two correlation eigenvectors, (J, 2).

    Stage 1 diagnosis (ADR §6a): per-item *marginal* statistics are
    block-exchangeable in a balanced two-construct world — which items covary
    together is a pairwise property no per-item summary can express, so the
    assignment head had literally no input signal to separate equal blocks.
    The second eigenvector's sign pattern is the classical minimal signal for
    a two-block split. Sign is canonicalised per world (largest-magnitude
    entry positive): arbitrary but deterministic, and slot labels are
    exchangeable anyway. Computed here, not in ``features.py`` — the frozen
    one-dimensional model's feature contract is untouched.
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
    for position in (-1, -2):
        vector = eigenvectors[:, position]
        anchor = int(vector.abs().argmax())
        if vector[anchor] < 0:
            vector = -vector
        columns.append(vector)
    scale = float(tensors.n_items) ** 0.5  # entries ~O(1) regardless of J
    return torch.stack(columns, dim=1) * scale


@dataclass(slots=True)
class Forward2:
    """One two-slot forward pass."""

    a: torch.Tensor  # (J,) positive magnitudes
    assign: torch.Tensor  # (J, 2) softmax slot probabilities
    b: torch.Tensor  # (J, CMAX-1) ordered thresholds
    b_valid: torch.Tensor  # (J, CMAX-1) bool
    mu: torch.Tensor  # (P, 2)
    logvar: torch.Tensor  # (P, 2)
    gate_logit: torch.Tensor  # () activation of slot 2
    gate_floor: float = 0.0  # training-schedule floor; 0 at inference

    @property
    def sd(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)

    @property
    def gate(self) -> torch.Tensor:
        """Learned gate, lifted by the annealed floor during early training.

        ``g + (1 - g) * floor`` keeps the gradient path to the logit alive at
        every floor value, unlike a hard max.
        """
        learned = torch.sigmoid(self.gate_logit)
        return learned + (1.0 - learned) * self.gate_floor

    @property
    def loadings(self) -> torch.Tensor:
        """Effective (J, 2) loadings: magnitude x assignment, slot 2 gated."""
        raw = self.a[:, None] * self.assign
        return torch.stack([raw[:, 0], raw[:, 1] * self.gate], dim=1)


class MultidimAmortizedGrm(nn.Module):
    """Two construct slots, one gate, the same constrained GRM decoder."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.item_stage_one = _mlp(ITEM_FEATURE_DIM + N_SPECTRAL_FEATURES, hidden, hidden)
        self.item_stage_two = _mlp(2 * hidden, hidden, hidden)
        self.head_a = nn.Linear(hidden, 1)
        self.head_assign = nn.Linear(hidden, N_SLOTS)
        self.head_b_first = nn.Linear(hidden, 1)
        self.head_b_deltas = nn.Linear(hidden, CMAX - 2)
        self.person_cell = _mlp(1 + hidden, hidden, hidden)
        # Shared across slots: exchangeable labels by construction.
        self.person_head = _mlp(hidden + 2, hidden, 2)
        self.gate_head = _mlp(WORLD_FEATURE_DIM, hidden // 2, 1)
        # Symmetry breaking (ADR §6a, Stage 1 diagnosis, two iterations):
        # with uniform initial assignments both slots pool identical
        # representations, the second slot is collinear with the first, the
        # gate closes, and the closed gate removes the gradient that would
        # ever differentiate the assignments. A warm-open *initialisation*
        # proved insufficient — early in training the sampled second latent is
        # noise, the likelihood punishes it, and the gate logit saturates
        # negative before the posterior becomes informative. The remedy is a
        # **gate floor annealed by the trainer** (``gate_floor`` attribute,
        # 1 → 0 over the early epochs): slot 2 stays exposed to gradient while
        # its posterior learns; afterwards the learned gate takes over and the
        # parsimony penalty can still close it on one-dimensional worlds.
        # ``gate_floor`` is training-loop state; inference leaves it at 0.
        nn.init.normal_(self.head_assign.weight, std=1.0)
        with torch.no_grad():
            self.gate_head[-1].bias.fill_(1.5)
        self.gate_floor: float = 0.0

    def forward(self, tensors: WorldTensors) -> Forward2:
        phi = torch.cat([item_features(tensors), spectral_item_features(tensors)], dim=1)
        h = self.item_stage_one(phi)
        pooled = h.mean(dim=0, keepdim=True).expand_as(h)
        h = self.item_stage_two(torch.cat([h, pooled], dim=1))

        a = nn.functional.softplus(self.head_a(h).squeeze(-1)) + A_FLOOR
        assign = torch.softmax(self.head_assign(h), dim=1)  # (J, 2)
        first = self.head_b_first(h)
        deltas = nn.functional.softplus(self.head_b_deltas(h)) + DELTA_FLOOR
        b = torch.cat([first, first + deltas.cumsum(dim=1)], dim=1)
        categories = torch.arange(1, CMAX, device=b.device)[None, :]
        b_valid = categories < tensors.n_categories[:, None]

        values, mask = normalized_values(tensors)  # (P*T, J)
        cell = self.person_cell(
            torch.cat([values[:, :, None], h[None, :, :].expand(values.shape[0], -1, -1)], dim=2)
        )  # (P*T, J, H)
        p, t = tensors.n_persons, tensors.n_waves
        wave_share = torch.tensor(1.0 / t, dtype=cell.dtype).expand(p)

        posterior_by_slot = []
        for k in range(N_SLOTS):
            weights = (mask.to(cell.dtype) * assign[None, :, k])[:, :, None]
            row_pooled = (cell * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
            person_pooled = row_pooled.reshape(p, t, -1).mean(dim=1)
            # Slot-specific observed fraction (analysis diagnosis: a *global*
            # fraction made sigma respond to total information only, so
            # masking one construct's items raised both slots' uncertainty
            # equally). Each slot's uncertainty input is the share of ITS
            # assignment-weighted items that this person actually answered.
            slot_observed = (mask.to(cell.dtype) * assign[None, :, k]).sum(dim=1) / assign[
                :, k
            ].sum().clamp(min=1e-6)
            slot_fraction = slot_observed.reshape(p, t).mean(dim=1)
            posterior_by_slot.append(
                self.person_head(
                    torch.cat([person_pooled, slot_fraction[:, None], wave_share[:, None]], dim=1)
                )
            )
        stacked = torch.stack(posterior_by_slot, dim=1)  # (P, 2, 2)
        mu = stacked[:, :, 0]
        logvar = stacked[:, :, 1].clamp(min=-8.0, max=4.0)

        gate_logit = self.gate_head(world_features(tensors)).squeeze()
        return Forward2(
            a=a,
            assign=assign,
            b=b,
            b_valid=b_valid,
            mu=mu,
            logvar=logvar,
            gate_logit=gate_logit,
            gate_floor=float(self.gate_floor),
        )


def category_log_probabilities_2(
    forward: Forward2, n_categories: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """(P, J, CMAX) GRM log probabilities under the two-slot linear predictor."""
    loadings = forward.loadings  # (J, 2)
    linear = torch.einsum("pk,jk->pj", theta, loadings)[:, :, None] - forward.b[None, :, :]
    cumulative = torch.sigmoid(linear)
    categories = torch.arange(1, CMAX, device=forward.b.device)[None, :]
    valid = (categories < n_categories[:, None])[None, :, :]
    cumulative = torch.where(valid, cumulative, torch.zeros_like(cumulative))
    ones = torch.ones_like(cumulative[:, :, :1])
    zeros = torch.zeros_like(cumulative[:, :, :1])
    padded = torch.cat([ones, cumulative, zeros], dim=2)
    masses = (padded[:, :, :-1] - padded[:, :, 1:]).clamp(min=1e-8)
    log_masses = masses.log()
    index = torch.arange(CMAX, device=forward.b.device)[None, None, :]
    return torch.where(
        index < n_categories[None, :, None],
        log_masses,
        torch.full_like(log_masses, float("-inf")),
    )


def masked_log_likelihood_2(
    tensors: WorldTensors, forward: Forward2, theta: torch.Tensor
) -> torch.Tensor:
    """Mean observed-cell log likelihood, theta of shape (P, 2)."""
    p, t, j = tensors.responses.shape
    log_probabilities = category_log_probabilities_2(forward, tensors.n_categories, theta)
    expanded = log_probabilities[:, None, :, :].expand(p, t, j, CMAX)
    gathered = expanded.gather(3, tensors.responses.clamp(min=0)[:, :, :, None]).squeeze(3)
    return gathered[tensors.mask].mean() if bool(tensors.mask.any()) else gathered.sum() * 0.0


def gate_closed(forward: Forward2) -> Forward2:
    """The same forward with slot 2 forced inactive, for the likelihood diagnostic."""
    return Forward2(
        a=forward.a,
        assign=forward.assign,
        b=forward.b,
        b_valid=forward.b_valid,
        mu=forward.mu,
        logvar=forward.logvar,
        gate_logit=torch.tensor(-30.0),
    )


def dimensionality_diagnostics(tensors: WorldTensors, forward: Forward2) -> dict[str, float]:
    """Truth-free Phase 4 diagnostics, computed at inference (ADR §6a)."""
    with torch.no_grad():
        open_loglik = float(masked_log_likelihood_2(tensors, forward, forward.mu))
        closed_loglik = float(masked_log_likelihood_2(tensors, gate_closed(forward), forward.mu))
        gate = float(forward.gate)
        slot_mass = forward.assign.mean(dim=0)
        entropy = float(-(forward.assign.clamp(min=1e-8).log() * forward.assign).sum(dim=1).mean())
        mu = forward.mu.numpy()
        correlation = float(torch.corrcoef(forward.mu.T)[0, 1]) if mu[:, 0].std() > 1e-9 else 0.0
    # Posterior signal share per slot — a truth-free reliability proxy.
    with torch.no_grad():
        signal = forward.mu.var(dim=0)
        noise = (forward.logvar.exp()).mean(dim=0)
        reliability = (signal / (signal + noise).clamp(min=1e-8)).clamp(min=0.05, max=1.0)
        disattenuated = correlation / float(torch.sqrt(reliability[0] * reliability[1]))
        disattenuated = float(np.clip(disattenuated, -1.5, 1.5))

    # The dimensionality DECISION (ADR §6a, Stage 3 revision): the raw gate is
    # NOT identified — the decoder sees only a·π₂·gate, so the magnitude head
    # absorbs any gate value and the parsimony term then drives the gate down
    # regardless of the world (Stage 3: likelihood-only sensitivity 0.00 while
    # slot-2 recovery stood at r 0.9). The identified evidence is separation:
    # two slots measuring ONE latent have disattenuated correlation ≈ 1 at any
    # noise level; distinct constructs at the corpus ceiling (0.7) sit near
    # 0.7. Declared two-dimensional iff both slots carry assignment mass, the
    # disattenuated correlation is below 0.85, and removing slot 2 costs
    # reconstruction likelihood. The gate remains a reported diagnostic and a
    # training-exposure device; it no longer decides.
    predicted_two = bool(
        float(slot_mass.min()) >= 0.10
        and abs(disattenuated) < 0.85
        and (open_loglik - closed_loglik) > 0.02
    )
    # Explicit ambiguity flag for highly correlated constructs (Phase 4
    # acceptance requirement): near the decision threshold the classification
    # is a boundary call and downstream users must see that, whichever side
    # it lands on. The band brackets the threshold; it does not change the
    # decision and is not a new separation statistic.
    dimensionality_ambiguous = bool(
        float(slot_mass.min()) >= 0.10 and 0.75 <= abs(disattenuated) <= 0.95
    )
    return {
        "gate": gate,
        "predicted_two_dimensional": predicted_two,
        "dimensionality_ambiguous": dimensionality_ambiguous,
        "loglik_gate_open": open_loglik,
        "loglik_gate_closed": closed_loglik,
        "loglik_gain_from_slot2": open_loglik - closed_loglik,
        "slot1_mass": float(slot_mass[0]),
        "slot2_mass": float(slot_mass[1]),
        "assignment_entropy": entropy,
        "inter_slot_correlation": correlation,
        "inter_slot_correlation_disattenuated": disattenuated,
        "slot_reliability_1": float(reliability[0]),
        "slot_reliability_2": float(reliability[1]),
    }


def structure_flags(diagnostics: dict[str, float]) -> list[str]:
    """Named failure modes; thresholds fixed here, exercised by tests."""
    flags: list[str] = []
    if diagnostics.get("dimensionality_ambiguous"):
        flags.append("dimensionality_ambiguous")
    two_dimensional = diagnostics["gate"] > 0.5
    if two_dimensional and abs(diagnostics["inter_slot_correlation"]) > 0.95:
        flags.append("duplicate_factors")
    if two_dimensional and min(diagnostics["slot1_mass"], diagnostics["slot2_mass"]) < 0.05:
        flags.append("collapse_suspected_empty_slot")
    if not two_dimensional and diagnostics["loglik_gain_from_slot2"] > 0.05:
        flags.append("gate_closed_despite_likelihood_gain")
    return flags


def parameter_count_2(model: MultidimAmortizedGrm) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
