"""Milestone 5E: longitudinal trait-state model (ADR section 2e).

theta_ikt = trait_ik + state_ikt, with trait ~ N(0, 1) and state following a
stationary mean-reverting process whose Delta-t aware transition is the prior.
The likelihood sees only the sum, so the split is identified by the
lag-autocorrelation profile: trait contributes correlation 1 at every lag,
state contributes phi^Delta.

The decoder is the unchanged constrained GRM; the transition is an explicit
scored prior. There is no recurrent prediction network anywhere, and
prospective prediction is the transition applied forward, never a learned
extrapolator. Truth never enters this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from human_measurement.model.features import (
    ITEM_FEATURE_DIM,
    item_features,
    normalized_values,
)
from human_measurement.model.multidim import (
    A_FLOOR,
    CMAX,
    DELTA_FLOOR,
    N_SLOTS,
    N_SPECTRAL_FEATURES,
    _mlp,
    spectral_item_features,
)
from human_measurement.model.schemas import WorldTensors

LONGITUDINAL_VERSION = "5e-v1"

#: Persistence cap (ADR 2e). phi -> 1 makes the state a random walk, a random
#: walk absorbs a constant, and trait/state collapse at no cost to the ELBO.
#: An identification device, not a tuning knob.
PHI_MAX = 0.98

#: Number of world-level dynamics features fed to the transition head.
N_DYNAMICS_FEATURES = 8

#: Milestone 5E acceptance (v0.5.0-exp.5e.1): an **experimental
#: longitudinal-core checkpoint**, not the universal Human Measurement Model
#: and not a passed milestone. The 5E gate was PARTIAL (criteria 6 and 9
#: failed, criterion 11 partial); the capability flags below are the accepted
#: contract. Outputs in ``unsupported_outputs`` must never be exposed as
#: validated results.
EXPERIMENTAL_MODEL_5E: dict[str, object] = {
    "arm": "arm5_combined",
    "status": "experimental_partial",
    "model_state_sha256": "224a701b8f6981e1e3eeeb64426a9bc8ab0122d7780bb74bb606a0a630397fc8",
    "config_sha256": "56f11a9b3834637cefe492eb9fb834e9ab1f575955f4cc42d9e278a38b53d884",
    "benchmark_manifest_sha256": (
        "f0e5c79f048846e49390b2b4513c0cfa7624d243b265e60c7014df680f94b0af"
    ),
    "longitudinal_version": "5e-v1",
    "scope": "one substantive construct; two to six waves; ordinal responses",
    "supported_outputs": (
        "stable_trait_posterior",
        "wave_state_posterior",
        "continuous_change_score",
        "persistence_and_recovery_parameters",
        "trait_state_variance_share",
        "dropout_aware_population_trajectory",
    ),
    "unsupported_outputs": (
        "individual_binary_change_flag",  # L34
        "unconstrained_drift_classification",  # L35
        "multidimensional_longitudinal_inference",  # L36
        "decoy_temporal_inference",  # L36
    ),
}


@dataclass(frozen=True, slots=True)
class Forward5E:
    """One longitudinal forward pass."""

    a: torch.Tensor  # (J,) positive magnitudes
    assign: torch.Tensor  # (J, 2) slot probabilities
    b: torch.Tensor  # (J, CMAX-1) ordered thresholds
    b_valid: torch.Tensor  # (J, CMAX-1) bool
    trait_mu: torch.Tensor  # (P, 2)
    trait_logvar: torch.Tensor  # (P, 2)
    state_mu: torch.Tensor  # (P, T, 2)
    state_logvar: torch.Tensor  # (P, T, 2)
    phi: torch.Tensor  # () persistence in (0, PHI_MAX)
    log_q: torch.Tensor  # () log innovation variance
    phi_log_sd: torch.Tensor  # () uncertainty for proper scoring
    q_log_sd: torch.Tensor  # ()
    drift: torch.Tensor  # (T,) additive threshold shift, drift[0] == 0
    gate_logit: torch.Tensor  # () activation of slot 2
    delta_t: torch.Tensor  # (T,) normalised spacing, delta_t[0] == 0
    wave_observed: torch.Tensor  # (P, T) bool
    gate_floor: float = 0.0

    @property
    def trait_sd(self) -> torch.Tensor:
        return torch.exp(0.5 * self.trait_logvar)

    @property
    def state_sd(self) -> torch.Tensor:
        return torch.exp(0.5 * self.state_logvar)

    @property
    def gate(self) -> torch.Tensor:
        learned = torch.sigmoid(self.gate_logit)
        return learned + (1.0 - learned) * self.gate_floor

    @property
    def loadings(self) -> torch.Tensor:
        raw = self.a[:, None] * self.assign
        return torch.stack([raw[:, 0], raw[:, 1] * self.gate], dim=1)

    @property
    def level_mu(self) -> torch.Tensor:
        """Total latent level by wave: trait + state. Never overwrites trait."""
        return self.trait_mu[:, None, :] + self.state_mu

    @property
    def stationary_variance(self) -> torch.Tensor:
        return self.log_q.exp() / (1.0 - self.phi**2).clamp(min=1e-4)


@torch.no_grad()
def dynamics_features(tensors: WorldTensors) -> torch.Tensor:
    """World-level lag statistics of the observed responses.

    Symmetric statistics only: per-wave person means of normalised responses,
    their lag-1..3 autocorrelations, the within/between variance ratio (the
    raw ICC - admissible as an observable *input*, never as a target or an
    acceptance criterion, ADR 2e), and the schedule shape.
    """
    values, mask = normalized_values(tensors)
    p, t, j = tensors.responses.shape
    values = values.reshape(p, t, j)
    mask_3d = mask.reshape(p, t, j)
    counts = mask_3d.sum(dim=2).clamp(min=1).to(values.dtype)
    per_wave = values.sum(dim=2) / counts  # (P, T)
    observed = mask_3d.any(dim=2)  # (P, T)

    features = []
    for lag in (1, 2, 3):
        if t > lag:
            first, second = per_wave[:, :-lag], per_wave[:, lag:]
            pair = observed[:, :-lag] & observed[:, lag:]
            if int(pair.sum()) > 3:
                x, y = first[pair], second[pair]
                xc, yc = x - x.mean(), y - y.mean()
                denominator = (xc.std() * yc.std()).clamp(min=1e-6)
                features.append(((xc * yc).mean() / denominator).clamp(-1.0, 1.0))
            else:
                features.append(torch.zeros(()))
        else:
            features.append(torch.zeros(()))

    person_mean = (per_wave * observed).sum(dim=1) / observed.sum(dim=1).clamp(min=1)
    between = person_mean.var().clamp(min=1e-8)
    within = ((per_wave - person_mean[:, None]) ** 2 * observed).sum() / observed.sum().clamp(min=1)
    features.append(between / (between + within).clamp(min=1e-8))  # raw ICC (input only)
    features.append(within.sqrt().clamp(max=5.0))
    features.append(torch.tensor(float(t) / 6.0))
    features.append(observed.to(values.dtype).mean())
    spacing = tensors.wave_time.diff(dim=1) if t > 1 else tensors.wave_time
    features.append((spacing.std() / spacing.mean().clamp(min=1e-6)).clamp(max=3.0))
    return torch.stack(features)[:N_DYNAMICS_FEATURES]


def normalised_spacing(tensors: WorldTensors) -> torch.Tensor:
    """Per-wave elapsed time in median-spacing units; entry 0 is zero."""
    t = tensors.n_waves
    if t < 2:
        return torch.zeros(t)
    times = tensors.wave_time.to(torch.float32).mean(dim=0)  # (T,)
    gaps = times.diff().clamp(min=0.0)
    scale = gaps.median().clamp(min=1e-6)
    return torch.cat([torch.zeros(1), gaps / scale])


class LongitudinalAmortizedGrm(nn.Module):
    """Trait + state posteriors, an explicit transition, one GRM decoder."""

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
        # Trait: pooled over observed waves. State: per wave, conditioned on the
        # person's own pooled representation so it can express a deviation.
        # Both shared across slots - exchangeable labels by construction.
        self.trait_head = _mlp(hidden + 2, hidden, 2)
        self.state_head = _mlp(2 * hidden + 3, hidden, 2)
        self.transition_head = _mlp(N_DYNAMICS_FEATURES, hidden, 4)
        self.drift_head = _mlp(hidden + 2, hidden // 2, 1)
        self.gate_head = _mlp(N_DYNAMICS_FEATURES, hidden // 2, 1)
        nn.init.normal_(self.head_assign.weight, std=1.0)
        with torch.no_grad():
            self.gate_head[-1].bias.fill_(1.5)
            # Start persistent-but-not-degenerate and small-innovation, so the
            # first epochs do not begin inside the collapse regime.
            self.transition_head[-1].bias.copy_(torch.tensor([0.8, -1.0, -1.0, -1.0]))
        self.gate_floor: float = 0.0

    def forward(self, tensors: WorldTensors) -> Forward5E:
        phi_features = torch.cat([item_features(tensors), spectral_item_features(tensors)], dim=1)
        h = self.item_stage_one(phi_features)
        pooled_items = h.mean(dim=0, keepdim=True).expand_as(h)
        h = self.item_stage_two(torch.cat([h, pooled_items], dim=1))

        a = nn.functional.softplus(self.head_a(h).squeeze(-1)) + A_FLOOR
        assign = torch.softmax(self.head_assign(h), dim=1)
        first = self.head_b_first(h)
        deltas = nn.functional.softplus(self.head_b_deltas(h)) + DELTA_FLOOR
        b = torch.cat([first, first + deltas.cumsum(dim=1)], dim=1)
        categories = torch.arange(1, CMAX, device=b.device)[None, :]
        b_valid = categories < tensors.n_categories[:, None]

        p, t, j = tensors.responses.shape
        values, mask = normalized_values(tensors)  # (P*T, J)
        cell = self.person_cell(
            torch.cat([values[:, :, None], h[None, :, :].expand(values.shape[0], -1, -1)], dim=2)
        )  # (P*T, J, H)
        mask_3d = mask.reshape(p, t, j)
        wave_observed = mask_3d.any(dim=2)  # (P, T)
        delta_t = normalised_spacing(tensors)

        trait_out, state_out = [], []
        for k in range(N_SLOTS):
            weights = (mask.to(cell.dtype) * assign[None, :, k])[:, :, None]
            row = (cell * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
            per_wave = row.reshape(p, t, -1)  # (P, T, H)
            live = wave_observed.to(per_wave.dtype)[:, :, None]
            pooled = (per_wave * live).sum(dim=1) / live.sum(dim=1).clamp(min=1e-6)  # (P, H)

            slot_observed = (mask.to(cell.dtype) * assign[None, :, k]).sum(dim=1) / assign[
                :, k
            ].sum().clamp(min=1e-6)
            slot_fraction = slot_observed.reshape(p, t)  # (P, T)
            person_fraction = (slot_fraction * wave_observed).sum(dim=1) / wave_observed.sum(
                dim=1
            ).clamp(min=1)
            wave_share = wave_observed.to(per_wave.dtype).mean(dim=1)

            trait_out.append(
                self.trait_head(
                    torch.cat([pooled, person_fraction[:, None], wave_share[:, None]], dim=1)
                )
            )
            state_input = torch.cat(
                [
                    per_wave,
                    pooled[:, None, :].expand(-1, t, -1),
                    slot_fraction[:, :, None],
                    wave_observed.to(per_wave.dtype)[:, :, None],
                    delta_t[None, :, None].expand(p, -1, -1),
                ],
                dim=2,
            )
            state_out.append(self.state_head(state_input))

        trait_stacked = torch.stack(trait_out, dim=1)  # (P, 2, 2)
        state_stacked = torch.stack(state_out, dim=2)  # (P, T, 2, 2)
        trait_mu = trait_stacked[:, :, 0]
        trait_logvar = trait_stacked[:, :, 1].clamp(min=-8.0, max=4.0)
        state_mu = state_stacked[:, :, :, 0]
        state_logvar = state_stacked[:, :, :, 1].clamp(min=-8.0, max=4.0)

        dynamics = dynamics_features(tensors)
        transition = self.transition_head(dynamics)
        phi = PHI_MAX * torch.sigmoid(transition[0])
        log_q = transition[1].clamp(min=-6.0, max=3.0)
        phi_log_sd = transition[2].clamp(min=-4.0, max=2.0)
        q_log_sd = transition[3].clamp(min=-4.0, max=2.0)

        # Wave-level threshold drift: one shift per wave from the wave's own
        # pooled item representation; wave 1 is the reference (drift[0] = 0).
        wave_repr = []
        for wave in range(t):
            live = mask_3d[:, wave, :].to(cell.dtype)
            weight = live.sum().clamp(min=1e-6)
            block = cell.reshape(p, t, j, -1)[:, wave, :, :]
            wave_repr.append((block * live[:, :, None]).sum(dim=(0, 1)) / weight)
        wave_repr_tensor = torch.stack(wave_repr)  # (T, H)
        drift_raw = self.drift_head(
            torch.cat(
                [
                    wave_repr_tensor,
                    delta_t[:, None],
                    torch.arange(t, dtype=wave_repr_tensor.dtype)[:, None] / max(t - 1, 1),
                ],
                dim=1,
            )
        ).squeeze(-1)
        drift = drift_raw - drift_raw[0]

        gate_logit = self.gate_head(dynamics).squeeze()
        return Forward5E(
            a=a,
            assign=assign,
            b=b,
            b_valid=b_valid,
            trait_mu=trait_mu,
            trait_logvar=trait_logvar,
            state_mu=state_mu,
            state_logvar=state_logvar,
            phi=phi,
            log_q=log_q,
            phi_log_sd=phi_log_sd,
            q_log_sd=q_log_sd,
            drift=drift,
            gate_logit=gate_logit,
            delta_t=delta_t,
            wave_observed=wave_observed,
            gate_floor=float(self.gate_floor),
        )


def category_log_probabilities_5e(
    forward: Forward5E, n_categories: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """(P, T, J, CMAX) GRM log probabilities with per-wave threshold drift."""
    loadings = forward.loadings  # (J, 2)
    linear = torch.einsum("ptk,jk->ptj", theta, loadings)[:, :, :, None]
    thresholds = forward.b[None, None, :, :] + forward.drift[None, :, None, None]
    cumulative = torch.sigmoid(linear - thresholds)
    categories = torch.arange(1, CMAX, device=forward.b.device)
    valid = categories[None, None, None, :] < n_categories[None, None, :, None]
    cumulative = torch.where(valid, cumulative, torch.zeros_like(cumulative))
    ones = torch.ones_like(cumulative[:, :, :, :1])
    zeros = torch.zeros_like(cumulative[:, :, :, :1])
    padded = torch.cat([ones, cumulative, zeros], dim=3)
    masses = (padded[:, :, :, :-1] - padded[:, :, :, 1:]).clamp(min=1e-8)
    log_masses = masses.log()
    index = torch.arange(CMAX, device=forward.b.device)
    return torch.where(
        index[None, None, None, :] < n_categories[None, None, :, None],
        log_masses,
        torch.full_like(log_masses, float("-inf")),
    )


def masked_log_likelihood_5e(
    tensors: WorldTensors, forward: Forward5E, theta: torch.Tensor
) -> torch.Tensor:
    """Mean observed-cell log likelihood; theta of shape (P, T, 2)."""
    log_probabilities = category_log_probabilities_5e(forward, tensors.n_categories, theta)
    gathered = log_probabilities.gather(3, tensors.responses.clamp(min=0)[:, :, :, None]).squeeze(3)
    if not bool(tensors.mask.any()):
        return gathered.sum() * 0.0
    return gathered[tensors.mask].mean()


def transition_log_prior(forward: Forward5E, state: torch.Tensor) -> torch.Tensor:
    """Sum of log p(state) under the Delta-t aware stationary AR(1) prior."""
    phi = forward.phi.clamp(max=PHI_MAX)
    stationary = forward.stationary_variance.clamp(min=1e-6)
    total = (
        -0.5 * (state[:, 0, :] ** 2 / stationary + stationary.log() + math.log(2.0 * math.pi)).sum()
    )
    for wave in range(1, state.shape[1]):
        decay = phi ** forward.delta_t[wave].clamp(min=1e-3)
        variance = (stationary * (1.0 - decay**2)).clamp(min=1e-6)
        residual = state[:, wave, :] - decay * state[:, wave - 1, :]
        total = (
            total - 0.5 * (residual**2 / variance + variance.log() + math.log(2.0 * math.pi)).sum()
        )
    return total / max(state.shape[0] * state.shape[1], 1)


def decoder_flags_5e(forward: Forward5E) -> list[str]:
    flags: list[str] = []
    if not bool((forward.a > 0).all()):
        flags.append("non_positive_discrimination")
    if bool(((forward.b[:, 1:] - forward.b[:, :-1]) <= 0).any()):
        flags.append("unordered_thresholds")
    finite = (
        torch.isfinite(forward.trait_mu).all()
        & torch.isfinite(forward.state_mu).all()
        & torch.isfinite(forward.phi)
    )
    if not bool(finite):
        flags.append("non_finite_posterior")
    if float(forward.trait_sd.std()) < 1e-6 and float(forward.state_sd.std()) < 1e-6:
        flags.append("uncertainty_collapsed_to_constant")
    if float(forward.phi) >= PHI_MAX - 1e-3:
        flags.append("persistence_at_cap")
    return flags
