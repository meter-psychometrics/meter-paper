"""The structured graded-response decoders, re-exported for inspection.

METER's response decoder is Samejima's graded-response model (GRM): neural
heads emit latent means/log-variances, positive item discriminations and
strictly ordered thresholds, and response probabilities are evaluated through
the closed-form GRM link rather than an unconstrained classifier head. The
parameterisation (identical across pathways) is::

    a = softplus(head_a) + A_FLOOR                      # discrimination > 0
    b = [b1, b1 + cumsum(softplus(deltas) + DELTA_FLOOR)]  # ordered thresholds
    P(Y >= c | theta) = sigmoid(a * (theta - b_c))

Everything here is the frozen implementation re-exported; the functions are
the ones the paper's evaluation scripts call.

* ``category_log_probabilities`` / ``masked_log_likelihood`` — the Phase 3
  1-D core (synthetic-recovery claim);
* ``category_log_probabilities_2`` / ``masked_log_likelihood_2`` — the
  multidimensional base;
* ``category_log_probabilities_5e`` / ``masked_log_likelihood_5e`` — the 5E
  pathway behind every real-data 1-D result;
* the supplied-structure pathway emits ``a``/``b``/``b_valid`` in its forward
  container (``MappedForward``) under the same parameterisation.
"""

from __future__ import annotations

from human_measurement.model.longitudinal import (
    LongitudinalAmortizedGrm,
    category_log_probabilities_5e,
    masked_log_likelihood_5e,
)
from human_measurement.model.multidim import (
    MultidimAmortizedGrm,
    category_log_probabilities_2,
    masked_log_likelihood_2,
)
from human_measurement.model.network import (
    A_FLOOR,
    DELTA_FLOOR,
    AmortizedGrm,
    category_log_probabilities,
    masked_log_likelihood,
)

__all__ = [
    "A_FLOOR",
    "DELTA_FLOOR",
    "AmortizedGrm",
    "LongitudinalAmortizedGrm",
    "MultidimAmortizedGrm",
    "category_log_probabilities",
    "category_log_probabilities_2",
    "category_log_probabilities_5e",
    "masked_log_likelihood",
    "masked_log_likelihood_2",
    "masked_log_likelihood_5e",
]
