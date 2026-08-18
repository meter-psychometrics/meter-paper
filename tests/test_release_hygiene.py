"""Release tests 8-9: no restricted content, no unpublished-module imports.

Test 8 scans every file in the package for credentials, absolute local paths,
participant-data markers and restricted-dataset payloads. Test 9 asserts both
statically (no generator/trainer/DIF/linking/discovery files exist in the
package) and dynamically (a full scoring run imports none of those modules).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".cff", ".r", ".R", ".sha256"}

#: Modules that must never ship or import: generators, trainers, and
#: capabilities outside the paper.
FORBIDDEN_MODULE_FILES = [
    "simulation",
    "model/build.py",
    "model/training.py",
    "model/phase3.py",
    "model/phase4_train.py",
    "model/phase5d_train.py",
    "model/phase5e_train.py",
    "model/dif.py",
    "model/linking.py",
    "model/change_detection.py",
    "model/change_worlds.py",
    "model/linking_worlds.py",
    "model/longitudinal_worlds.py",
    "model/multimodal_worlds.py",
    "model/multimodal_arms.py",
    "mapped/train.py",
    "battery/worlds.py",
    "ingestion/cli.py",
    "ingestion/mapping.py",
    "ingestion/profile.py",
    "ingestion/suggest.py",
    "ingestion/templates.py",
    "ingestion/validation.py",
]

FORBIDDEN_IMPORT_PREFIXES = (
    "human_measurement.simulation",
    "human_measurement.model.build",
    "human_measurement.model.training",
    "human_measurement.model.dif",
    "human_measurement.model.linking",
    "human_measurement.model.change_detection",
    "human_measurement.model.change_worlds",
    "human_measurement.model.linking_worlds",
    "human_measurement.model.longitudinal_worlds",
    "human_measurement.model.multimodal_worlds",
    "human_measurement.model.multimodal_arms",
    "human_measurement.mapped.train",
    "human_measurement.battery.worlds",
    "human_measurement.ingestion.cli",
)

RESTRICTED_CONTENT_PATTERNS = [
    r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}",
    r"(?i)(secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----",
    r"C:\\Users\\[A-Za-z0-9]",
    r"/Users/[A-Za-z0-9]+/",
    r"data[/\\]private",
    # an actual SHARE person identifier VALUE (the column NAME appears in
    # schema documentation and is public vocabulary; values must never ship)
    r"['\"][A-Z]{2}-\d{6}-\d{2}['\"]",
    r"(?i)Tresorit",
]


def _package_files():
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        yield path


def test_no_restricted_content_anywhere():
    offenders = []
    scanner = Path(__file__).resolve()
    for path in _package_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == scanner:  # the pattern list itself
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in RESTRICTED_CONTENT_PATTERNS:
            if re.search(pattern, text):
                offenders.append((str(path.relative_to(PACKAGE_ROOT)), pattern))
    assert not offenders, f"restricted content found: {offenders}"


def test_no_dataset_payloads_shipped():
    """No SPSS/Stata/SAS/parquet data files and no large opaque payloads
    besides the three frozen checkpoints and the fixtures archive."""
    allowed_binaries = {
        "checkpoints/phase5e-arm5_combined.pt",
        "checkpoints/phase3-stage2-arm1_beta05.pt",
        "checkpoints/mapped_known_structure_seed8100.pt",
        "fixtures/equivalence_fixtures.npz",
    }
    for path in _package_files():
        rel = str(path.relative_to(PACKAGE_ROOT)).replace("\\", "/")
        assert path.suffix.lower() not in {".sav", ".dta", ".sas7bdat", ".parquet", ".zip"}, rel
        if path.name.startswith("."):  # dotfiles (.gitignore) are text
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix.lower() not in {".csv"}:
            assert rel in allowed_binaries, f"unexpected binary file: {rel}"


def test_forbidden_module_files_absent():
    base = PACKAGE_ROOT / "src" / "human_measurement"
    present = [name for name in FORBIDDEN_MODULE_FILES if (base / name).exists()]
    assert not present, f"unpublished modules shipped: {present}"


def test_full_scoring_run_imports_no_unpublished_module():
    from meter_reference import load_model

    rng = np.random.default_rng(3)
    x = rng.integers(0, 4, size=(20, 6)).astype(float)
    load_model("meter-1d-paper").score(x, metadata={"data_provenance": "synthetic"})
    dense = rng.integers(0, 4, size=(30, 9)).astype(float)
    load_model("meter-multidim-paper").score(
        dense, factor_map=[0, 0, 0, 1, 1, 1, 2, 2, 2], metadata={"data_provenance": "synthetic"}
    )

    loaded = [name for name in sys.modules if name.startswith("human_measurement")]
    offenders = [
        name
        for name in loaded
        for prefix in FORBIDDEN_IMPORT_PREFIXES
        if name == prefix or name.startswith(prefix + ".")
    ]
    assert not offenders, f"scoring imported unpublished modules: {offenders}"
