"""Release test 1: the shipped checkpoints ARE the frozen paper checkpoints.

Two layers: file-level SHA-256 against ``CHECKSUMS.sha256``, and the frozen
STATE digests (sorted-key SHA-256 over tensor bytes) against the pins recorded
in the paper manifest and asserted by every frozen loader.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = PACKAGE_ROOT / "checkpoints"

STATE_PINS = {
    ("phase5e-arm5_combined.pt", "best_state"): (
        "224a701b8f6981e1e3eeeb64426a9bc8ab0122d7780bb74bb606a0a630397fc8"
    ),
    ("phase3-stage2-arm1_beta05.pt", "best_state"): (
        "0d571638a8eb29f64fc0202a67d72620faf2b7ec1229670bc977ab643bac36a4"
    ),
    ("mapped_known_structure_seed8100.pt", "state"): (
        "8d42752009e49694def7c5dfbd77b328c28e337e85c67274058ec111dd2cf7b4"
    ),
}


def _state_digest(state) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_file_checksums_match():
    recorded = {}
    for line in (CHECKPOINTS / "CHECKSUMS.sha256").read_text().splitlines():
        value, name = line.split()
        recorded[name.lstrip("*")] = value
    assert set(recorded) == {name for name, _ in STATE_PINS}
    for name, expected in recorded.items():
        actual = hashlib.sha256((CHECKPOINTS / name).read_bytes()).hexdigest()
        assert actual == expected, f"file hash mismatch for {name}"


def test_frozen_state_digests_match():
    for (name, key), expected in STATE_PINS.items():
        payload = torch.load(CHECKPOINTS / name, map_location="cpu", weights_only=False)
        assert _state_digest(payload[key]) == expected, f"state digest mismatch for {name}"


def test_manifest_pins_agree():
    manifest = json.loads((PACKAGE_ROOT / "paper_manifest.json").read_text())
    models = {m["checkpoint"]: m["state_sha256"] for m in manifest["models"]}
    for (name, _), expected in STATE_PINS.items():
        assert models[f"checkpoints/{name}"] == expected
