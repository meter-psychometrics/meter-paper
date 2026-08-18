"""Paper-reproducibility trim of ``mapped/mapping.py``.

A mapping is simple structure: every item belongs to exactly one proposed
factor. The research module additionally implements the deliberate mapping
corruptions used by the synthetic evaluation tasks (built on the battery world
generator, which is not released). ``SuppliedMapping`` and
``mirt_model_string`` below are verbatim — the mapping container is THE model
input for supplied-structure scoring, and the mirt string is the exact
confirmatory-comparator specification recorded in every comparator output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The backbone width. K above this is out of scope for the milestone.
K_MAX = 5


@dataclass(frozen=True)
class SuppliedMapping:
    """An externally supplied item-to-factor hypothesis. THE model input.

    ``item_factor[j]`` is the 0-based proposed factor of item j; ``k`` is the
    number of proposed factors; ``labels`` are analyst-facing names carried
    through to outputs so results are reported in the analyst's vocabulary.
    ``provenance`` records how the mapping was built - for synthetic tasks the
    corruption family, for real data the instrument scoring key.
    """

    item_factor: tuple[int, ...]
    k: int
    labels: tuple[str, ...]
    provenance: str = "unspecified"
    #: Items the corruption touched, for evaluation only. The model never
    #: receives this - it is not part of the Q-matrix.
    corrupted_items: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        if not 1 <= self.k <= K_MAX:
            raise ValueError(f"k={self.k} outside 1..{K_MAX}")
        if len(self.labels) != self.k:
            raise ValueError(f"{len(self.labels)} labels for k={self.k}")
        used = set(self.item_factor)
        if not used <= set(range(self.k)):
            raise ValueError(f"item_factor references factors outside 0..{self.k - 1}")
        if used != set(range(self.k)):
            raise ValueError(
                f"every proposed factor must claim at least one item; missing "
                f"{sorted(set(range(self.k)) - used)}"
            )

    @property
    def n_items(self) -> int:
        return len(self.item_factor)

    def q_matrix(self) -> np.ndarray:
        """(J, k) one-hot simple-structure Q. What the model and mirt consume."""
        q = np.zeros((self.n_items, self.k), dtype=np.float64)
        q[np.arange(self.n_items), np.asarray(self.item_factor)] = 1.0
        return q

    def items_of(self, factor: int) -> tuple[int, ...]:
        return tuple(int(j) for j, f in enumerate(self.item_factor) if f == factor)

    def relabelled(self, permutation: tuple[int, ...]) -> SuppliedMapping:
        """The same hypothesis under a permutation of factor labels.

        ``permutation[old] = new``. Outputs of a correct model must permute
        identically and nothing else may change - the equivariance the tests
        pin.
        """
        if sorted(permutation) != list(range(self.k)):
            raise ValueError(f"not a permutation of 0..{self.k - 1}: {permutation}")
        inverse = [0] * self.k
        for old, new in enumerate(permutation):
            inverse[new] = old
        return SuppliedMapping(
            item_factor=tuple(permutation[f] for f in self.item_factor),
            k=self.k,
            labels=tuple(self.labels[inverse[new]] for new in range(self.k)),
            provenance=f"{self.provenance}|relabelled",
            corrupted_items=self.corrupted_items,
        )


# --------------------------------------------------------------------------
# the confirmatory mirt model string (Stage 1 section 3, field 1)
# --------------------------------------------------------------------------
def mirt_model_string(mapping: SuppliedMapping) -> str:
    """The EXACT `mirt.model` specification for the confirmatory comparator.

    One line per factor with 1-BASED item indices (mirt's convention), then a
    COV line freeing every pairwise factor covariance. No F*F terms, so factor
    variances stay fixed at 1 and the freed covariance block IS the
    correlation matrix. Recorded verbatim in every comparator output.

    For k = 1 there is no covariance to free and the COV line is omitted.
    """
    lines = [
        f"F{f + 1} = " + ",".join(str(j + 1) for j in mapping.items_of(f))
        for f in range(mapping.k)
    ]
    if mapping.k >= 2:
        pairs = [
            f"F{a + 1}*F{b + 1}"
            for a in range(mapping.k)
            for b in range(a + 1, mapping.k)
        ]
        lines.append("COV = " + ", ".join(pairs))
    return "\n".join(lines)


__all__ = [
    "K_MAX",
    "SuppliedMapping",
    "mirt_model_string",
]
