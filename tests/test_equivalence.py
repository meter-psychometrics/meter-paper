"""Release test 2: released inference == original frozen outputs, bit for bit.

The fixtures were produced by the FROZEN research stack (original release API,
original loaders) by ``tools/generate_fixtures_from_frozen.py``. This test
replays the identical inputs through the released ``meter_reference`` package
and asserts byte-identical float64 outputs — scores, uncertainty widths and
(for the supplied-structure fixture) the synthetic-provenance Phi matrix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PACKAGE_ROOT / "fixtures"


def _digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def _load():
    arrays = np.load(FIXTURES / "equivalence_fixtures.npz")
    manifest = json.loads((FIXTURES / "equivalence_manifest.json").read_text())
    return arrays, manifest


def _score(name, arrays, manifest):
    import meter_reference

    structure = manifest[name]["structure"]
    result = meter_reference.score(
        arrays[f"{name}__responses"],
        structure,
        metadata={"data_provenance": "synthetic"},
    )
    assert not result.refused, f"{name} refused: {result.refusal_reason}"
    return result


def test_unidimensional_outputs_bit_identical():
    arrays, manifest = _load()
    for name in ("fixture_1d", "fixture_1d_negative_encoding", "fixture_1d_binary"):
        result = _score(name, arrays, manifest)
        assert _digest(result.scores) == manifest[name]["scores_sha256_float64"], name
        assert (
            _digest(result.relative_reliability.values)
            == manifest[name]["reliability_sha256_float64"]
        ), name
        assert np.array_equal(
            np.asarray(result.scores, dtype=np.float64), arrays[f"{name}__scores"]
        ), name


def test_supplied_structure_outputs_bit_identical():
    arrays, manifest = _load()
    name = "fixture_multidim_k3"
    result = _score(name, arrays, manifest)
    assert _digest(result.scores) == manifest[name]["scores_sha256_float64"]
    assert (
        _digest(result.relative_reliability.values)
        == manifest[name]["reliability_sha256_float64"]
    )
    phi = result.diagnostics["factor_correlation_phi"]
    assert _digest(phi) == manifest[name]["phi_sha256_float64"]


def test_capability_and_status_match_frozen_run():
    arrays, manifest = _load()
    for name in ("fixture_1d", "fixture_1d_binary", "fixture_multidim_k3"):
        result = _score(name, arrays, manifest)
        assert result.capability == manifest[name]["capability"]
        assert result.status == manifest[name]["status"]
        assert sorted(result.suppressed) == manifest[name]["suppressed"]


def test_frozen_provenance_identical_except_documented_trim():
    """The packaged provenance equals the frozen one, except calibration_configs.

    The four calibration-config files belong to out-of-paper capabilities and
    are not shipped; their hash slots are None here by design. Everything else
    — interface versions, every model pin, every status — must be identical.
    """
    _, manifest = _load()
    from human_measurement.interface import frozen_provenance

    packaged = frozen_provenance()
    frozen = dict(manifest["_frozen_provenance"])
    packaged_configs = packaged.pop("calibration_configs")
    frozen.pop("calibration_configs")
    # score_is_calibrated_probability arrives as bool; JSON round-trip keeps it.
    assert {k: str(v) for k, v in packaged.items()} == {k: str(v) for k, v in frozen.items()}
    assert all(value is None for value in packaged_configs.values())
