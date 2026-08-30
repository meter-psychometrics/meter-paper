"""Generate the numerical-equivalence fixtures from the FROZEN research stack.

This script runs ONLY inside the METER research repository: it imports the
original release API (``release/meter/meter/api.py``), which loads the frozen
checkpoints through the original loaders (``ingestion.cli._load_models`` /
the seed-8100 reproduction), scores three synthetic fixture matrices, and
records inputs, outputs and SHA-256 digests of the float64 output bytes.

The public package's test suite then replays the same inputs through
``meter_reference`` and asserts bit-identical outputs. The fixtures are the
bridge that proves the released thin wrappers reproduce the frozen
implementation exactly; regenerating them requires the research repository by
design.

Fixture inputs are synthetic (seeded numpy draws) — no real data anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
#: The research checkout carrying the CANONICAL frozen stack at the pinned
#: contract version (v3.6). Defaults to the enclosing research worktree;
#: METER_FIXTURE_STACK overrides it when the canonical v3.6 stack lives in a
#: different checkout.
RESEARCH_ROOT = Path(os.environ.get("METER_FIXTURE_STACK", PACKAGE_ROOT.parents[0]))
FIXTURES = PACKAGE_ROOT / "fixtures"


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def make_inputs() -> dict[str, dict]:
    rng = np.random.default_rng(20_260_818)

    def matrix(p: int, j: int, cats: int, missing: float, seed_shift: int) -> np.ndarray:
        local = np.random.default_rng(20_260_818 + seed_shift)
        theta = local.normal(size=(p, 1))
        base = theta + local.normal(scale=1.0, size=(p, j))
        edges = np.quantile(base, np.linspace(0, 1, cats + 1)[1:-1])
        codes = np.digitize(base, edges).astype(float)
        drop = local.random(size=(p, j)) < missing
        codes[drop] = np.nan
        return codes

    x_1d = matrix(40, 9, 4, 0.15, 1)
    x_binary = matrix(30, 8, 2, 0.10, 2)
    x_multi = matrix(60, 15, 4, 0.0, 3)  # dense, as the supplied-structure gate requires
    factor_map = [0] * 5 + [1] * 5 + [2] * 5

    # NaN-vs-negative encodings of the SAME observations (mask-semantics test)
    x_1d_negative = np.where(np.isnan(x_1d), -1.0, x_1d)

    return {
        "fixture_1d": {"responses": x_1d, "structure": None},
        "fixture_1d_negative_encoding": {"responses": x_1d_negative, "structure": None},
        "fixture_1d_binary": {"responses": x_binary, "structure": None},
        "fixture_multidim_k3": {"responses": x_multi, "structure": factor_map},
        "_rng_note": {
            "seed_base": 20_260_818,
            "generator": "numpy default_rng, quantile-binned Gaussian draws",
        },
    }


def main() -> None:
    sys.path.insert(0, str(RESEARCH_ROOT / "src"))
    sys.path.insert(0, str(RESEARCH_ROOT / "release" / "meter"))
    import meter  # the ORIGINAL release API, original loaders

    FIXTURES.mkdir(parents=True, exist_ok=True)
    inputs = make_inputs()
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {
        "_provenance": {
            "generated_by": "tools/generate_fixtures_from_frozen.py",
            "stack": "research release/meter via ingestion.cli._load_models",
            "meter_version": meter.__version__,
        }
    }

    for name, spec in inputs.items():
        if name.startswith("_"):
            manifest[name] = spec
            continue
        responses = spec["responses"]
        arrays[f"{name}__responses"] = responses
        result = meter.score(
            responses,
            spec["structure"],
            metadata={"data_provenance": "synthetic"},
        )
        assert not result.refused, f"{name} unexpectedly refused: {result.refusal_reason}"
        scores = np.asarray(result.scores, dtype=np.float64)
        # v3.6 (P5 EXIT_2): the released reliability values are within-result
        # midrank percentiles of the withheld posterior widths. The fixture
        # records the RELEASED values; raw widths never leave the stack.
        reliability = np.asarray(result.relative_reliability.values, dtype=np.float64)
        arrays[f"{name}__scores"] = scores
        arrays[f"{name}__reliability"] = reliability
        manifest[name] = {
            "structure": spec["structure"],
            "capability": result.capability,
            "status": result.status,
            "scores_sha256_float64": _digest(scores),
            "reliability_sha256_float64": _digest(reliability),
            "scores_shape": list(scores.shape),
            "warnings": list(result.warnings),
            "suppressed": sorted(result.suppressed),
        }
        if name == "fixture_multidim_k3":
            phi = np.asarray(result.diagnostics["factor_correlation_phi"], dtype=np.float64)
            arrays[f"{name}__phi"] = phi
            manifest[name]["phi_sha256_float64"] = _digest(phi)

    # The frozen provenance dictionary, for the pin-identity test.
    from human_measurement.interface import frozen_provenance

    manifest["_frozen_provenance"] = frozen_provenance()

    np.savez_compressed(FIXTURES / "equivalence_fixtures.npz", **arrays)
    (FIXTURES / "equivalence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print("fixtures written:", sorted(arrays))
    for name, entry in manifest.items():
        if not name.startswith("_"):
            print(name, entry["capability"], entry["scores_sha256_float64"][:16])


if __name__ == "__main__":
    main()
