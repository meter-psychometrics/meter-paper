"""Milestone 6B.2.3: numeric-only digests of frozen evaluation artifacts.

An interface milestone must be able to prove it changed no result. Hashing the
artifact files whole cannot show that, because those files legitimately carry
warnings, route traces, timings and contract text that this kind of milestone
is *supposed* to change. A whole-file hash would fail for the right reasons and
therefore prove nothing.

So the digest covers **numbers only**: every float and int reachable in the
document, in key order, with the non-numeric scaffolding dropped. A relabelled
output, a new warning, a renamed field or a slower run leaves it untouched. A
changed estimate does not.

Booleans are excluded deliberately. In Python a bool is an int, and folding
them in would let a status flag flipping register as a numeric change - the
opposite of what this measures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def numeric_leaves(value: Any, path: str = "") -> list[tuple[str, float]]:
    """Every numeric leaf in a document, with its path. Sorted, deterministic."""
    found: list[tuple[str, float]] = []
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        # NaN is a legitimate result value here (an unavailable correlation),
        # so it is recorded as a token rather than dropped or coerced.
        found.append((path, value))
        return found
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            found.extend(numeric_leaves(value[key], f"{path}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(numeric_leaves(item, f"{path}[{index}]"))
    return found


def numeric_digest(document: Any) -> str:
    """A hash of a document's numbers, independent of its prose."""
    leaves = numeric_leaves(document)
    payload = "\n".join(
        f"{path}={'nan' if value != value else repr(value)}" for path, value in leaves
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def digest_file(path: str | Path) -> tuple[str, int]:
    """Numeric digest and numeric-leaf count for one JSON artifact."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    leaves = numeric_leaves(document)
    return numeric_digest(document), len(leaves)


def content_sha256(path: str | Path) -> str:
    """Whole-file sha256 over content, with line endings normalised to LF.

    Use this - never ``sha256(path.read_bytes())`` - for any pin that compares
    a hash recorded in this repository against a file checked out of it.

    A line ending is a property of the checkout, not of the content. This is a
    Windows working copy with ``core.autocrlf=true``, so git materialises text
    files with CRLF; the same commit on Linux, in CI, or in a freshly created
    linked worktree materialises LF. A pin taken over raw bytes therefore
    records which machine computed it, and the guarantee it is supposed to
    enforce then passes or fails by geography rather than by whether anything
    actually changed. That is worse than no pin: it trains whoever meets it to
    re-pin the value rather than investigate it.

    Normalising here rather than at checkout is deliberate. A ``.gitattributes``
    ``-text`` rule fixes the bytes a *future* checkout produces, but it does
    nothing to an existing working tree, nothing to a hash already recorded
    against the other representation, and nothing on a machine that clones
    before the rule lands. This function makes the comparison itself
    representation-independent, which is the property the pins need.

    This also makes explicit the convention the repository was already
    following by accident: every hash recorded here - the 5A/5D/5E/5E.1 config
    hashes, the linking decision config quoted in four frozen governance
    packages, the calibration and benchmark manifests, and the values
    ``interface.frozen_provenance`` reports - is the LF-normalised one, because
    it was computed either from an LF working tree or through ``read_text()``,
    which applies universal newlines. The one exception is recorded, with its
    correction, in ``configs/model/frozen_numeric_pins.json``.

    The cost is stated plainly: an edit that changes *only* line endings no
    longer trips these checks. That is the intended trade. Such an edit changes
    no value, no threshold and no decision, and the alternative is a check that
    cannot survive being run on a second machine.
    """
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def pin_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "model" / "frozen_numeric_pins.json"


def load_pins() -> dict[str, Any]:
    path = pin_path()
    if not path.exists():
        return {"artifacts": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_pins(root: Path | None = None) -> dict[str, Any]:
    """Check every pinned artifact still carries the same numbers.

    An artifact that is absent is reported as ``missing`` rather than as a
    failure: `reports/` is gitignored, so a fresh clone legitimately has none
    of them, and a check that fails on a clean checkout would be turned off
    within a week. What must never happen is an artifact that is PRESENT and
    whose numbers moved.

    ``passed`` therefore answers only "did anything present move?". It is True
    over an empty set, which is exactly the vacuous reading a caller must not
    mistake for a verified one. ``pinned`` reports the full expected set so a
    caller can tell a real pass from an unarmed one; ``coverage_reason``
    spells out why the check could not be armed, or is None when it was.
    """
    root = root or Path(__file__).resolve().parents[2]
    pins = load_pins()
    artifacts = pins.get("artifacts", {})
    checked, missing, mismatched = [], [], []
    for relative, record in sorted(artifacts.items()):
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        digest, count = digest_file(path)
        if digest != record["numeric_sha256"]:
            mismatched.append(
                {
                    "artifact": relative,
                    "pinned": record["numeric_sha256"],
                    "actual": digest,
                    "pinned_leaf_count": record["numeric_leaf_count"],
                    "actual_leaf_count": count,
                }
            )
        else:
            checked.append(relative)
    result = {
        "pinned": sorted(artifacts),
        "checked": checked,
        "missing": missing,
        "mismatched": mismatched,
        "passed": not mismatched,
        "root": str(root),
    }
    result["coverage_reason"] = coverage_reason(result)
    return result


def coverage_reason(result: dict[str, Any]) -> str | None:
    """Why this checkout cannot arm the numeric-pin tripwire, or None if it can.

    An absent artifact must never read as a pass. ``verify_pins`` compares
    digests over whatever is present, so on a checkout holding none of them it
    returns ``passed=True`` having compared nothing at all - a vacuous result
    that looks identical to a verified one. It must not read as an unexplained
    failure either: `assert result["checked"]` blowing up with an empty list
    tells whoever meets it nothing about why, and the honest answer is usually
    that the files were never there to check.

    Callers use this to report the check as explicitly NOT RUN, naming what is
    absent and why it legitimately can be. A mismatch is a separate matter and
    is always a hard failure - this function never explains one away.
    """
    missing = result["missing"]
    if not missing:
        return None
    return (
        f"{len(missing)} of {len(result['pinned'])} pinned artifacts are absent "
        f"from this checkout ({result['root']}), so the numeric-digest tripwire "
        "cannot be armed here and MUST NOT be reported as having passed. These "
        "live under `reports/`, which .gitignore excludes by content, so they "
        "exist only in a working tree where the milestone was actually run - "
        "not in a fresh clone, a CI checkout, or a linked git worktree, all of "
        "which are legitimate places to run the suite. To arm it, run the suite "
        "in the working tree that holds these artifacts, or regenerate them "
        "with the milestone-6 scripts that produced them. Absent: " + ", ".join(missing)
    )
