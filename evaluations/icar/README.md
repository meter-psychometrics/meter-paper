# ICAR-16 — Table 1 row 8 (prespecified FAILURE, reported as such)

**Paper result:** four supplied cognitive factors (n = 4,574): factor-wise
**0.8610-0.9439**, median 0.9023 — below the prospectively frozen gates
(0.90 per-factor floor, 0.93 median), so cross-domain cognitive transfer is
NOT established. The comparator converged and was admissible: this is a
scientific boundary, not a technical escape. Reported numbers:
`derived_results/icar/` including the permanent closure record.

**Source data:** International Cognitive Ability Resource, ICAR-16 Sample
Test (Condon & Revelle 2014, Intelligence). The instrument is public-domain;
the respondent dataset's exact licence must be honoured separately (see the
charter's provenance block).

**Expected input:** 16 items in the frozen order LN.07, LN.33, LN.34, LN.58,
MR.45, MR.46, MR.47, MR.55, VR.04, VR.16, VR.17, VR.19, R3D.03, R3D.04,
R3D.06, R3D.08; binary correct/incorrect codes 0/1; missing = empty.
`factor_map.json` carries the frozen four-factor assignment.

**Frozen procedure:** research script `d9d_transfer_run.py`
(sha256 04ab60eaa6906779…), reviewer-only; populated charter in this
directory. The released model attaches a construct-transfer warning whenever
metadata declares a cognition-like construct — that warning IS the D9D
boundary, enforced at the interface.

**Reproduce:**

```
python run_meter.py --responses icar16.csv \
    --comparator mirt_factor_scores.csv --output result.json
```
