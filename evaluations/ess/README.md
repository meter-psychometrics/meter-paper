# ESS round 7 CES-D-8 — Table 1 row 2

**Paper result:** 21 countries, n = 40,185; pooled METER vs country-fitted
MML-GRM r = **0.9696**, per-country 0.9545-0.9899. Reported numbers:
`derived_results/ess/phase6b2_ess7_cesd8.json` (which also carries the
country-level uncertainty-agreement association, r = -0.80).

**Source data:** European Social Survey round 7 (ESS7-2014, edition 2.2),
free registration at europeansocialsurvey.org; CC BY-NC-SA. Not
redistributable here.

**Expected input:** rows = respondents; first column = country code (use
`--group-column`); the eight CES-D-8 items (fltdpr, flteeff, slprl, wrhpp,
fltlnl, enjlf, fltsd, cldgng), 4-point codes recoded 0-3, reverse-keyed items
handled per `item_mapping.yaml`; refusals/don't-know to missing.

**Frozen procedure:** research script `phase6b2_run.py`
(sha256 69e3eac93ce88728…), reviewer-only. Comparator fitted per country with
`comparators/fit_irt_baseline.R --dimensions 1`.

**Reproduce:**

```
python run_meter.py --responses ess7_cesd8.csv --group-column \
    --comparator comparator_scores.csv --output result.json
```

Per-country agreements and the pooled value appear in `result.json`.
