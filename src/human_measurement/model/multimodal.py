"""Paper-reproducibility trim of ``model/multimodal.py``.

The research module implements the experimental multimodal fusion pathway,
which is outside the METER foundational paper. The frozen 1-D/longitudinal
scoring path needs exactly one function from it: ``psychometric_message``,
which reads the accepted longitudinal posterior as a Gaussian message. It is
copied verbatim below; nothing else from the research module is released.
"""

from __future__ import annotations

import torch

from human_measurement.model.longitudinal import Forward5E


def psychometric_message(forward: Forward5E, *, slot: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The accepted longitudinal posterior as a Gaussian message, unchanged."""
    mean = forward.level_mu[:, :, slot]
    precision = 1.0 / (
        forward.trait_sd[:, slot][:, None] ** 2 + forward.state_sd[:, :, slot] ** 2
    ).clamp(min=1e-8)
    return mean, precision
