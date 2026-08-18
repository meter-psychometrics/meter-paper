# Canonical model lineage

The lineage below is the accepted, audited record of which frozen model
produced each result in the paper. All weights were trained only on simulated
measurement worlds and frozen before the corresponding real dataset was
opened; state digests are SHA-256 over sorted state-dictionary keys and
tensor bytes, asserted at every load. No target dataset contributed to the
optimization of any model.

## The three frozen states

| Canonical name | Package id | Architecture | Parameters | State SHA-256 | Training-world family | Frozen |
|---|---|---|---|---|---|---|
| **M-ref** (one-dimensional reference core) | `meter-1d-synthetic-m4` | `AmortizedGrm` | 31,051 | `0d571638a8eb29f64fc0202a67d72620faf2b7ec1229670bc977ab643bac36a4` | one-dimensional ordinal worlds; 200 pretraining / 40 validation / 40 unseen evaluation worlds, disjoint seed regions | 2026-08-02 |
| **M-score** (single-construct scoring module) | `meter-1d-paper` | `LongitudinalAmortizedGrm`, applied in single-wave mode through the frozen routing interface | 42,899 | `224a701b8f6981e1e3eeeb64426a9bc8ab0122d7780bb74bb606a0a630397fc8` | longitudinal-family worlds; 150-300 respondents, 6-14 items, 2-6 waves, 4/5/7 response categories; 150 pretraining / 24 validation worlds | 2026-08-03 |
| **M-struct** (structure-conditioned multidimensional module) | `meter-multidim-paper` | `KnownStructureGrm` | 30,986 | `8d42752009e49694def7c5dfbd77b328c28e337e85c67274058ec111dd2cf7b4` | battery worlds; 400-2,000 respondents, K = 3-5 supplied factors, 2/4/6 categories; reproduced deterministically from frozen seeds (torch seed 8100) and hash-asserted on every use | 2026-08-06 |

## Analysis → model

| Reported analysis | Model |
|---|---|
| Synthetic 40-world known-truth benchmark (median r = 0.9388) | M-ref |
| Uncertainty coverage correction (0.8288 → 0.9479) | M-ref |
| NHANES 2015-16 PHQ-9, development (0.9330; Extended Data) | M-score |
| NHANES 2017-18 PHQ-9, locked (0.9371) | M-score |
| ESS round 7 CES-D-8, 21 countries (0.9696 pooled) | M-score |
| SHARE wave 9 EURO-D, 28 countries (0.9851 pooled) | M-score |
| SHARE wave 8 EURO-D, 27 countries (0.9847 pooled) | M-score |
| SAPA five scales, 12.4% observed (0.9672-0.9842) | M-score (scale-wise) |
| MIDUS 1 Big Five, development (0.9048-0.9807; Extended Data) | M-struct |
| MIDUS Refresher Big Five (0.9241-0.9747; both Phi gates failed) | M-struct |
| IPIP-50 Big Five (0.9596-0.9839, median 0.9791) | M-struct |
| ICAR-16, four cognitive factors (0.8610-0.9439, median 0.9023; failed) | M-struct |
| P4B capability-router verification (Fig. 4) | routing layer over the frozen states (contract v3.5; no weights of its own) |

Two distinct one-dimensional modules therefore appear in the paper: M-ref
provides the known-truth synthetic benchmark and the uncertainty-calibration
analysis; M-score produced every real-data unidimensional result. M-score is
drawn from the longitudinal module family but is used in this Article
exclusively for cross-sectional scoring with frozen weights; longitudinal
trait-state and multimodal analyses are outside the paper.
