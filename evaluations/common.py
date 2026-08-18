"""Shared evaluation harness for the real-data reproductions.

Scores a user-prepared response matrix with the frozen paper model and
computes the agreement metrics the paper reports (Pearson and midrank
Spearman with a named comparator, pooled and per group). The user prepares
the response matrix from the source dataset following the dataset README and
item mapping; participant-level data never enters this repository.

Input format (both files): CSV with an optional leading ``group`` column
(country, cohort), then one column per item, integer 0-based category codes,
empty cells = unobserved. Comparator files carry one ``score`` column per
factor instead of item columns. Row order must match between files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def read_matrix(path: Path, has_group: bool) -> tuple[np.ndarray, list[str] | None]:
    rows = [line.rstrip("\n").split(",") for line in path.read_text(encoding="utf-8").splitlines()]
    header, body = rows[0], rows[1:]
    start = 1 if has_group else 0
    groups = [r[0] for r in body] if has_group else None
    matrix = np.array(
        [[float(cell) if cell != "" else np.nan for cell in row[start:]] for row in body]
    )
    return matrix, groups


def score_file(
    responses_csv: Path,
    *,
    factor_map: list[int] | None,
    has_group: bool,
    metadata: dict | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    from meter_reference import load_model

    matrix, groups = read_matrix(responses_csv, has_group)
    model = load_model("meter-multidim-paper" if factor_map else "meter-1d-paper")
    result = model.score(matrix, factor_map=factor_map, metadata=metadata or {})
    if result.refused:
        raise SystemExit(f"frozen model refused this request: {result.refusal_reason}")
    return np.asarray(result.scores), groups


def agreement(meter: np.ndarray, comparator: np.ndarray) -> dict:
    meter = np.asarray(meter, dtype=float).ravel()
    comparator = np.asarray(comparator, dtype=float).ravel()
    live = np.isfinite(meter) & np.isfinite(comparator)
    meter, comparator = meter[live], comparator[live]

    def midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(x)
        sorted_x = x[order]
        i = 0
        position = np.arange(len(x), dtype=float)
        while i < len(x):
            j = i
            while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            ranks[order[i : j + 1]] = position[i : j + 1].mean()
            i = j + 1
        return ranks

    return {
        "n": int(live.sum()),
        "pearson_agreement": float(np.corrcoef(meter, comparator)[0, 1]),
        "midrank_spearman": float(np.corrcoef(midrank(meter), midrank(comparator))[0, 1]),
    }


def evaluate(
    meter_scores: np.ndarray,
    comparator_scores: np.ndarray,
    groups: list[str] | None,
) -> dict:
    report: dict = {"pooled": agreement(meter_scores, comparator_scores)}
    if groups is not None:
        per_group = {}
        labels = np.asarray(groups)
        for label in sorted(set(groups)):
            mask = labels == label
            if mask.sum() >= 2:
                per_group[label] = agreement(meter_scores[mask], comparator_scores[mask])
        values = sorted(entry["pearson_agreement"] for entry in per_group.values())
        report["per_group"] = per_group
        if values:
            report["per_group_summary"] = {
                "n_groups": len(values),
                "min": values[0],
                "median": float(np.median(values)),
                "max": values[-1],
            }
    return report


def main(factor_map: list[int] | None = None, metadata: dict | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, default=None,
                        help="CSV of fitted comparator scores (same row order)")
    parser.add_argument("--group-column", action="store_true",
                        help="first column is a group label (country/cohort)")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    scores, groups = score_file(
        arguments.responses,
        factor_map=factor_map,
        has_group=arguments.group_column,
        metadata=metadata,
    )
    payload: dict = {"n_scored": int(np.asarray(scores).shape[0])}
    if arguments.comparator is not None:
        comparator, _ = read_matrix(arguments.comparator, arguments.group_column)
        if scores.ndim == 1 or scores.shape[1] == 1:
            payload["agreement"] = evaluate(scores.ravel(), comparator.ravel(), groups)
        else:
            payload["agreement_per_factor"] = {
                f"factor_{k}": evaluate(scores[:, k], comparator[:, k], groups)
                for k in range(scores.shape[1])
            }
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload.get("agreement", payload), indent=2)[:2000])
    print("written:", arguments.output)
