"""End-to-end synthetic demonstration of the frozen paper models.

Shows, on fully synthetic data, that: the package loads; masking works; a
response matrix produces person scores; the model weights remain fixed; the
supplied-structure pathway scores factors; and refusals carry named reasons
instead of numbers. Run from the package root:

    python examples/synthetic_example.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from meter_reference import load_model  # noqa: E402
from meter_reference import api  # noqa: E402


def state_digest(model) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    rows = (PACKAGE_ROOT / "examples" / "example_response_matrix.csv").read_text().splitlines()
    matrix = np.array(
        [[float(c) if c != "" else np.nan for c in row.split(",")] for row in rows[1:]]
    )
    print(f"example matrix: {matrix.shape[0]} persons x {matrix.shape[1]} items, "
          f"{np.isnan(matrix).mean():.0%} cells missing")

    # 1) unidimensional scoring with missingness via the observation mask
    model = load_model("meter-1d-paper")
    result = model.score(matrix, metadata={"data_provenance": "synthetic"})
    print(f"\n1D scores: shape {result.scores.shape}, "
          f"first three {np.round(result.scores[:3], 4)}")
    print(f"uncertainty is {result.uncertainty.scale}-only "
          f"(permitted: {', '.join(result.uncertainty.permitted_uses)})")

    # 2) the weights did not move
    before = state_digest(api._models()["longitudinal"])
    model.score(matrix, metadata={"data_provenance": "synthetic"})
    assert state_digest(api._models()["longitudinal"]) == before
    print(f"\nweights fixed: state {before[:16]}... unchanged across calls")

    # 3) supplied-structure multidimensional scoring (structure is an INPUT)
    dense = np.nan_to_num(matrix, nan=1.0)
    factor_map = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    multidim = load_model("meter-multidim-paper")
    factors = multidim.score(dense, factor_map=factor_map,
                             metadata={"data_provenance": "synthetic"})
    print(f"\nfactor scores: shape {factors.scores.shape} (K=3 supplied factors)")

    # 4) out-of-support requests refuse with named reasons, not numbers
    refusal = api.score(dense, metadata={"n_factors": 3, "data_provenance": "synthetic"})
    assert refusal.refused and refusal.scores is None
    print(f"\nstructure DISCOVERY refused as designed:\n  {refusal.refusal_reason[:120]}...")


if __name__ == "__main__":
    main()
