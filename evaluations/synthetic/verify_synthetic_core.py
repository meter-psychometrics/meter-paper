"""Verify the frozen Phase 3 synthetic-recovery core loads and decodes.

Loads the checkpoint behind the paper's r = 0.9388 claim (hash-asserted),
scores the packaged example matrix, and checks the structural decoder
properties the Methods describe: positive discriminations and strictly
ordered thresholds. Regenerating the r = 0.9388 evaluation itself requires
the research repository's world generator (see README.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "src"))


def main() -> None:
    from meter_reference.inference import load_phase3_reference
    from meter_reference.preprocessing import (
        as_response_array,
        canonical_dataset,
        resolve_categories,
        to_world_tensors,
    )

    model, digest = load_phase3_reference()
    print(f"frozen Phase 3 core loaded, state {digest[:16]}...")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    rows = (PACKAGE_ROOT / "examples" / "example_response_matrix.csv").read_text().splitlines()
    matrix = np.array(
        [[float(c) if c != "" else np.nan for c in row.split(",")] for row in rows[1:]]
    )
    array = as_response_array(matrix)
    observed = np.isfinite(array) & (array >= 0)
    categories = resolve_categories(array, observed, {})
    dataset = canonical_dataset(array, observed, categories, np.zeros((array.shape[0], 1)))
    with torch.no_grad():
        forward = model(to_world_tensors(dataset))

    a = forward.a.detach().numpy()
    b = forward.b.detach().numpy()
    b_valid = forward.b_valid.detach().numpy().astype(bool)
    assert (a > 0).all(), "discriminations must be positive (A_FLOOR)"
    for j in range(b.shape[0]):
        thresholds = b[j][b_valid[j]]
        assert (np.diff(thresholds) > 0).all(), "thresholds must be strictly ordered"
    print(f"decoder OK: {a.shape[0]} items, positive a, strictly ordered thresholds")
    print(f"person scores: shape {tuple(forward.mu.shape)}")


if __name__ == "__main__":
    main()
