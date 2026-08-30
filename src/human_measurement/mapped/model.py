"""Milestone 8: the mapping-conditioned measurement model (EXPERIMENTAL).

The Milestone 7 battery architecture minus discovery. Removed outright: the
gate head, gates, gate floor, parsimony, count loss and the learned assignment
head - none of that code is imported here. Added: the supplied mapping enters
as a hard (J x K) simple-structure mask, so the effective loadings are
``a[:, None] * Q`` - the audited magnitude-times-assignment factorisation with
the assignment SUPPLIED rather than learned.

Retained from the audited M7 design, deliberately: the two-stage
permutation-invariant item encoder with spectral features, the shared person
head with the FACTOR-SPECIFIC observed fraction (the measured repair that makes
masking one factor's items raise that factor's uncertainty and not every
factor's), ordered thresholds, and the Cholesky-parameterised correlated prior
consumed by the multivariate KL - wired, per the M7 trainer-contract audit.

K in 1..5 rides one K_MAX = 5 backbone: the person head is shared across
factors and evaluated only for the K supplied columns, so a K = 3 world
computes a K = 3 posterior, a K = 3 prior submatrix and a K = 3 KL. Nothing is
padded with dead slots; unused backbone capacity simply is not evaluated.

Truth is unreachable from the forward pass: inputs are the response tensors
and the Q-matrix, nothing else. The T10 guarantee is held by test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from human_measurement.battery.model import spectral_item_features
from human_measurement.mapped.mapping import K_MAX, SuppliedMapping
from human_measurement.model.features import (
    CMAX,
    ITEM_FEATURE_DIM,
    item_features,
    normalized_values,
)
from human_measurement.model.network import A_FLOOR, DELTA_FLOOR, _mlp
from human_measurement.model.schemas import WorldTensors

#: Same width the battery encoder used; the spectral signal is an encoder
#: input, not discovery logic.
N_SPECTRAL_FEATURES = K_MAX - 1


@dataclass(slots=True)
class MappedForward:
    """One forward pass under one supplied mapping."""

    a: torch.Tensor  # (J,) positive magnitude
    b: torch.Tensor  # (J, CMAX-1) ordered thresholds
    b_valid: torch.Tensor  # (J, CMAX-1)
    mu: torch.Tensor  # (P, K) factor-specific posterior means
    logvar: torch.Tensor  # (P, K) factor-specific posterior log-variances
    q: torch.Tensor  # (J, K) the supplied simple-structure mask
    prior_cholesky: torch.Tensor  # (K, K) lower factor of the prior correlation

    @property
    def sd(self) -> torch.Tensor:
        return (0.5 * self.logvar).exp()

    @property
    def k(self) -> int:
        return int(self.q.shape[1])

    @property
    def loadings(self) -> torch.Tensor:
        """(J, K) effective loadings: magnitude times the SUPPLIED mask."""
        return self.a[:, None] * self.q

    @property
    def prior_correlation(self) -> torch.Tensor:
        """(K, K) prior factor correlation. Unit diagonal by construction."""
        return self.prior_cholesky @ self.prior_cholesky.T

    def factor_support_index(self) -> torch.Tensor:
        """(K,) continuous support index per SUPPLIED factor, in [0, 1].

        var(mu_k) / (var(mu_k) + mean(sigma_k^2)) - an estimated reliability.

        WHAT IT IS AND IS NOT, stated per the gate-derivation terminology
        ruling: this is EVIDENCE ABOUT SUPPORT for a supplied factor - whether
        the data let the model score it with signal above its posterior noise.
        It is NOT proof that the factor is substantively valid: a coherent
        blend of other factors scores high here (measured: phantoms carved
        from real items reached 0.26-0.82, indistinguishable from matched
        factors). Redundancy and mapping misspecification require the SEPARATE
        diagnostics: ``redundancy_indices`` for phantom/merged signatures, and
        the misassignment checks of the benchmark's credited behaviours.
        It is the AUROC axis for WEAK-factor detection only.
        """
        signal = self.mu.var(dim=0, unbiased=False)
        noise = self.logvar.exp().mean(dim=0)
        return signal / (signal + noise).clamp(min=1e-12)

    def redundancy_indices(self) -> torch.Tensor:
        """(K,) how much of each factor's score the OTHER factors already carry.

        Multiple R^2 of factor k's posterior mean on the other factors' means,
        computed from the RAW score correlation (unit diagonal, PSD by
        construction):  R^2_k = 1 - 1 / (C^-1)_kk.

        Added by the fixture-v2 revision, with the v1 failure on record: a
        phantom factor CARVED FROM REAL ITEMS carries real signal, so its
        reliability is not low - measured 0.26-0.82 across twelve worlds,
        indistinguishable from matched factors. What such a factor cannot be
        is DISTINCT: its score is a blend of its donors, and redundancy is the
        signature reliability structurally cannot see. Raw rather than
        disattenuated correlations, because the same v1 run measured
        disattenuated twin-phi to be unstable under the uncalibrated amortized
        sigma (fires at rho 0.20, misses at 0.45-0.70 - the L45 lesson again).
        """
        if self.k == 1:
            return torch.zeros(1, dtype=self.mu.dtype)
        correlation = self.factor_correlation(disattenuated=False)
        # A ridge keeps the inverse finite when two factors are near-collinear
        # - which is exactly the redundant case; R^2 saturates toward 1 there.
        precision = torch.linalg.inv(
            correlation + 1e-6 * torch.eye(self.k, dtype=correlation.dtype)
        )
        return (1.0 - 1.0 / torch.diagonal(precision).clamp(min=1.0)).clamp(min=0.0, max=1.0)

    def positive_definite_phi(self) -> tuple[torch.Tensor, float]:
        """The disattenuated Phi, shrunk toward the raw correlation until PD.

        Disattenuation divides by reliabilities and can push a matrix outside
        the PD cone - a known property, not a bug. The RAW correlation of the
        posterior means is PSD by construction, so shrinking toward it always
        terminates:  Phi(w) = w * disattenuated + (1 - w) * raw,  stepping w
        down from 1.0 by 0.05 until the smallest eigenvalue clears 1e-6.
        Returns the matrix and the w actually used; w < 1.0 is REPORTED so a
        repair is never silent.
        """
        raw = self.factor_correlation(disattenuated=False)
        disattenuated = self.factor_correlation(disattenuated=True)
        weight = 1.0
        while weight > 0.0:
            candidate = weight * disattenuated + (1.0 - weight) * raw
            candidate = candidate - torch.diag_embed(torch.diagonal(candidate)) + torch.eye(
                self.k, dtype=candidate.dtype
            )
            smallest = float(torch.linalg.eigvalsh(candidate).min())
            if smallest > 1e-6:
                return candidate, round(weight, 2)
            weight = round(weight - 0.05, 2)
        return raw, 0.0

    def factor_correlation(self, disattenuated: bool = True) -> torch.Tensor:
        """(K, K) estimated factor correlation from the posterior means.

        Raw Pearson correlation of the posterior means, optionally
        disattenuated by the estimated reliabilities and clamped to
        [-0.99, 0.99]. Both versions are reported; the disattenuated one is
        the primary Phi estimate (Stage 1 section 2.5). K = 1 returns the
        1 x 1 identity.
        """
        if self.k == 1:
            return torch.ones(1, 1, dtype=self.mu.dtype)
        centered = self.mu - self.mu.mean(dim=0, keepdim=True)
        sd = centered.std(dim=0, unbiased=False).clamp(min=1e-8)
        raw = (centered / sd).T @ (centered / sd) / self.mu.shape[0]
        if not disattenuated:
            return raw
        reliability = self.factor_support_index().clamp(min=1e-6)
        scale = torch.sqrt(torch.outer(reliability, reliability))
        adjusted = raw / scale.clamp(min=1e-6)
        adjusted = adjusted.clamp(min=-0.99, max=0.99)
        return adjusted - torch.diag_embed(torch.diagonal(adjusted)) + torch.eye(
            self.k, dtype=adjusted.dtype
        )


class KnownStructureGrm(nn.Module):
    """3-5 supplied factors, one shared person head, the same GRM decoder.

    No gate head, no assignment head, no parsimony state, no count state.
    The only structural input is the Q-matrix.
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.item_stage_one = _mlp(ITEM_FEATURE_DIM + N_SPECTRAL_FEATURES, hidden, hidden)
        self.item_stage_two = _mlp(2 * hidden, hidden, hidden)
        self.head_a = nn.Linear(hidden, 1)
        self.head_b_first = nn.Linear(hidden, 1)
        self.head_b_deltas = nn.Linear(hidden, CMAX - 2)
        self.person_cell = _mlp(1 + hidden, hidden, hidden)
        # Shared across factors: relabelling equivariance by construction.
        self.person_head = _mlp(hidden + 2, hidden, 2)
        # EQUICORRELATION prior: one learned rho, Sigma = (1-rho)I + rho 11^T.
        #
        # Increment 1 shipped a Cholesky prior over backbone slots. That was a
        # DEFECT the increment-2 contract exposed: a slot-indexed prior is not
        # exchangeable, so relabelling the supplied factors permutes mu but
        # not the prior, and the KL - the training OBJECTIVE - changes under a
        # pure renaming. An exchangeable prior is the only kind compatible
        # with the required relabelling equivariance. Expressiveness is not
        # lost where it matters: the per-world Phi ESTIMATE is the
        # disattenuated posterior correlation, not the prior; the prior is a
        # global regularizer and rho its single, permutation-symmetric knob.
        self.prior_rho_raw = nn.Parameter(torch.zeros(1))

    #: rho is mapped into (RHO_LOWER, RHO_UPPER). The lower bound keeps
    #: Sigma positive definite for EVERY k up to K_MAX (PD requires
    #: rho > -1/(k-1); -1/(K_MAX-1) + margin covers all of them), the upper
    #: bound keeps it comfortably nonsingular.
    RHO_LOWER = -1.0 / (K_MAX - 1) + 0.02
    RHO_UPPER = 0.95

    def prior_rho(self) -> torch.Tensor:
        span = self.RHO_UPPER - self.RHO_LOWER
        return self.RHO_LOWER + span * torch.sigmoid(self.prior_rho_raw).squeeze(0)

    def prior_cholesky_for(self, k: int) -> torch.Tensor:
        """(k, k) lower Cholesky factor of the equicorrelated prior.

        Unit diagonal, all off-diagonals equal to rho - invariant under any
        permutation of the k factors, which is what makes the KL (and with it
        the whole objective) exactly indifferent to factor labels.
        """
        rho = self.prior_rho()
        eye = torch.eye(k, dtype=rho.dtype)
        sigma = (1.0 - rho) * eye + rho * torch.ones(k, k, dtype=rho.dtype)
        return torch.linalg.cholesky(sigma)

    def forward(self, tensors: WorldTensors, mapping: SuppliedMapping) -> MappedForward:
        if mapping.n_items != int(tensors.n_items):
            raise ValueError(
                f"mapping covers {mapping.n_items} items; world has {int(tensors.n_items)}"
            )
        q = torch.from_numpy(mapping.q_matrix()).to(torch.float32)

        phi = torch.cat([item_features(tensors), spectral_item_features(tensors)], dim=1)
        h = self.item_stage_one(phi)
        pooled = h.mean(dim=0, keepdim=True).expand_as(h)
        h = self.item_stage_two(torch.cat([h, pooled], dim=1))

        a = nn.functional.softplus(self.head_a(h).squeeze(-1)) + A_FLOOR
        first = self.head_b_first(h)
        deltas = nn.functional.softplus(self.head_b_deltas(h)) + DELTA_FLOOR
        b = torch.cat([first, first + deltas.cumsum(dim=1)], dim=1)
        categories = torch.arange(1, CMAX, device=b.device)[None, :]
        b_valid = categories < tensors.n_categories[:, None]

        values, mask = normalized_values(tensors)
        cell = self.person_cell(
            torch.cat([values[:, :, None], h[None, :, :].expand(values.shape[0], -1, -1)], dim=2)
        )
        p, t = tensors.n_persons, tensors.n_waves
        wave_share = torch.tensor(1.0 / t, dtype=cell.dtype).expand(p)

        posterior_by_factor = []
        for k in range(mapping.k):
            weights = (mask.to(cell.dtype) * q[None, :, k])[:, :, None]
            row_pooled = (cell * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
            person_pooled = row_pooled.reshape(p, t, -1).mean(dim=1)
            # FACTOR-SPECIFIC observed fraction: masking one factor's items
            # must raise THAT factor's uncertainty, not every factor's. The
            # measured M7 repair, carried over unchanged.
            mapped_mass = q[:, k].sum().clamp(min=1e-6)
            factor_observed = (mask.to(cell.dtype) * q[None, :, k]).sum(dim=1) / mapped_mass
            factor_fraction = factor_observed.reshape(p, t).mean(dim=1)
            posterior_by_factor.append(
                self.person_head(
                    torch.cat(
                        [person_pooled, factor_fraction[:, None], wave_share[:, None]], dim=1
                    )
                )
            )
        stacked = torch.stack(posterior_by_factor, dim=1)
        return MappedForward(
            a=a,
            b=b,
            b_valid=b_valid,
            mu=stacked[:, :, 0],
            logvar=stacked[:, :, 1].clamp(min=-8.0, max=4.0),
            q=q,
            prior_cholesky=self.prior_cholesky_for(mapping.k),
        )


# --------------------------------------------------------------------------
# adequacy diagnostics. Reported, never learned. (Stage 1 section 2.6)
# --------------------------------------------------------------------------
#: Graded statuses for a SUPPLIED factor. Capability-style labels; not members
#: of the frozen five-value output enum, and a test should keep it that way.
ADEQUACY_STATUSES = (
    "factor_supported",
    "factor_weak_support",
    "factor_unsupported",
    "factor_not_estimable",
)

#: PROVISIONAL thresholds. Frozen on DEVELOPMENT synthetic worlds before any
#: evaluation, per the Stage 1 gate-derivation procedure; these placeholders
#: exist so the plumbing is testable, and carry a test asserting they are
#: revisited at freeze time.
RELIABILITY_SUPPORTED = 0.70
RELIABILITY_WEAK = 0.40
MIN_ITEMS_PER_FACTOR = 3
MIN_RESPONDENTS_WITH_COVERAGE = 50
MIN_OBSERVED_ITEMS_PER_RESPONDENT = 2


@dataclass(frozen=True)
class FactorAdequacy:
    """One supplied factor's adequacy verdict, with the evidence that made it."""

    label: str
    status: str
    reliability: float
    n_items: int
    n_respondents_with_coverage: int
    score_variance: float
    mean_posterior_variance: float


def factor_adequacy(
    forward: MappedForward, mapping: SuppliedMapping, mask: np.ndarray
) -> tuple[FactorAdequacy, ...]:
    """Adequacy of every SUPPLIED factor, from model state and coverage only.

    ``mask`` is the (P, J) observedness matrix (wave-collapsed). Statuses:

    * ``factor_not_estimable`` - too few mapped items, or too few respondents with
      minimal coverage on this factor's items: the factor cannot be scored,
      whatever the data say;
    * ``factor_unsupported`` - estimable, but the posterior stayed near the prior
      (reliability below the weak floor): the phantom-factor signature;
    * ``factor_weak_support`` - between the floors;
    * ``factor_supported`` - reliability at or above the supported threshold.
    """
    reliabilities = forward.factor_support_index().detach()
    mu = forward.mu.detach()
    logvar = forward.logvar.detach()
    verdicts = []
    for factor in range(mapping.k):
        items = mapping.items_of(factor)
        enough = mask[:, list(items)].sum(axis=1) >= MIN_OBSERVED_ITEMS_PER_RESPONDENT
        coverage = int(enough.sum())
        reliability = float(reliabilities[factor])
        if len(items) < MIN_ITEMS_PER_FACTOR or coverage < MIN_RESPONDENTS_WITH_COVERAGE:
            status = "factor_not_estimable"
        elif reliability < RELIABILITY_WEAK:
            status = "factor_unsupported"
        elif reliability < RELIABILITY_SUPPORTED:
            status = "factor_weak_support"
        else:
            status = "factor_supported"
        verdicts.append(
            FactorAdequacy(
                label=mapping.labels[factor],
                status=status,
                reliability=reliability,
                n_items=len(items),
                n_respondents_with_coverage=coverage,
                score_variance=float(mu[:, factor].var(unbiased=False)),
                mean_posterior_variance=float(logvar[:, factor].exp().mean()),
            )
        )
    return tuple(verdicts)


__all__ = [
    "ADEQUACY_STATUSES",
    "FactorAdequacy",
    "KnownStructureGrm",
    "MappedForward",
    "factor_adequacy",
]
