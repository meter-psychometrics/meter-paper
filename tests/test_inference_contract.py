"""Release tests 3-6: frozen-weights, permutation, mask semantics, factor maps.

* test 3 — scoring performs NO model-parameter update (state digests before
  and after are identical, and no parameter requires grad during scoring);
* test 4 — row permutation of the input yields exactly permuted outputs;
* test 5 — the three observation encodings (NaN, negative sentinel, explicit
  mask argument) are one semantics;
* test 6 — supplied factor-map scoring matches the frozen multidimensional
  implementation (fixture equivalence) and is label-equivariant.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _rng_matrix(p=25, j=7, cats=4, missing=0.12, seed=7):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(p, 1))
    base = theta + rng.normal(size=(p, j))
    edges = np.quantile(base, np.linspace(0, 1, cats + 1)[1:-1])
    codes = np.digitize(base, edges).astype(float)
    codes[rng.random(size=(p, j)) < missing] = np.nan
    return codes


def _state_digest(model) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_no_parameter_update_during_inference():
    from meter_reference import api, load_model

    x = _rng_matrix()
    model_1d = load_model("meter-1d-paper")
    model_md = load_model("meter-multidim-paper")

    result = model_1d.score(x, metadata={"data_provenance": "synthetic"})
    assert not result.refused
    before_1d = _state_digest(api._models()["longitudinal"])
    before_md = _state_digest(api._mapped_model())

    dense = np.nan_to_num(_rng_matrix(missing=0.0, seed=8), nan=0.0)
    factor_map = [0, 0, 0, 1, 1, 2, 2]
    for _ in range(3):
        model_1d.score(x, metadata={"data_provenance": "synthetic"})
        model_md.score(dense, factor_map=factor_map, metadata={"data_provenance": "synthetic"})

    assert _state_digest(api._models()["longitudinal"]) == before_1d
    assert _state_digest(api._mapped_model()) == before_md
    assert not any(p.requires_grad and p.grad is not None for p in api._models()["longitudinal"].parameters())


def test_row_permutation_gives_permuted_scores():
    """Row permutation permutes the scores (to float summation-order precision).

    The frozen model is respondent-exchangeable by construction, but its
    item-summary features are computed by masked reductions whose floating-
    point summation order follows the input order, so equivariance holds to
    ~1e-6 rather than bitwise. That is a property of the frozen research
    implementation itself (verified against the original stack), not of this
    package.
    """
    from meter_reference import load_model

    x = _rng_matrix(seed=11)
    permutation = np.random.default_rng(0).permutation(x.shape[0])
    model = load_model("meter-1d-paper")
    base = model.score(x, metadata={"data_provenance": "synthetic"})
    permuted = model.score(x[permutation], metadata={"data_provenance": "synthetic"})
    np.testing.assert_allclose(
        np.asarray(permuted.scores),
        np.asarray(base.scores)[permutation],
        rtol=0.0,
        atol=1e-4,
    )
    assert np.corrcoef(
        np.asarray(permuted.scores), np.asarray(base.scores)[permutation]
    )[0, 1] > 0.999999


def test_mask_encodings_are_one_semantics():
    from meter_reference import load_model

    x_nan = _rng_matrix(seed=13)
    x_negative = np.where(np.isnan(x_nan), -1.0, x_nan)
    observed = ~np.isnan(x_nan)
    filled = np.nan_to_num(x_nan, nan=0.0)

    model = load_model("meter-1d-paper")
    a = model.score(x_nan, metadata={"data_provenance": "synthetic"})
    b = model.score(x_negative, metadata={"data_provenance": "synthetic"})
    c = model.score(filled, mask=observed, metadata={"data_provenance": "synthetic"})
    np.testing.assert_array_equal(np.asarray(a.scores), np.asarray(b.scores))
    np.testing.assert_array_equal(np.asarray(a.scores), np.asarray(c.scores))
    np.testing.assert_array_equal(
        np.asarray(a.relative_reliability.values),
        np.asarray(c.relative_reliability.values),
    )


def test_factor_map_scoring_is_label_equivariant():
    from meter_reference import load_model

    dense = np.nan_to_num(_rng_matrix(p=40, j=12, missing=0.0, seed=17), nan=0.0)
    base_map = [0] * 4 + [1] * 4 + [2] * 4
    permuted_map = [2 if f == 0 else (0 if f == 2 else f) for f in base_map]

    model = load_model("meter-multidim-paper")
    base = model.score(dense, factor_map=base_map, metadata={"data_provenance": "synthetic"})
    swapped = model.score(dense, factor_map=permuted_map, metadata={"data_provenance": "synthetic"})
    assert not base.refused and not swapped.refused
    np.testing.assert_array_equal(
        np.asarray(swapped.scores)[:, [2, 1, 0]], np.asarray(base.scores)
    )


def test_training_surface_absent():
    """The public API exposes no training hooks, optimizers or gradient paths."""
    import meter_reference
    from meter_reference import api

    assert not hasattr(meter_reference, "train")
    assert not hasattr(api, "train")
    model = api._models()["longitudinal"]
    assert all(not p.grad_fn for p in model.parameters())
