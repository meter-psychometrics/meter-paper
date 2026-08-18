# NHANES PHQ-9 (DPQ) — Table 1 row 1

**Paper result:** locked 2017-2018 cycle (DPQ_J, n = 5,533): METER vs fitted
MML-GRM r = **0.9371**. The 2015-16 cycle (DPQ_I, n = 5,735; r = 0.9330) was
pipeline development and appears in Extended Data. Reported numbers:
`derived_results/nhanes/`.

**Source data:** NHANES public-use files, National Center for Health
Statistics — DPQ_I (2015-16) and DPQ_J (2017-18) questionnaire files, freely
downloadable from the CDC NHANES site. No AI-specific restriction was
identified in the public-use terms; processing was local. Data may not be
redistributed here.

**Expected input:** one row per respondent with the nine DPQ items
(DPQ010-DPQ090), 0-3 category codes, 7/9 ("refused"/"don't know") recoded to
missing (empty cell). Item mapping: `item_mapping.yaml` (frozen research
mapping) and the two governance records in this directory.

**Frozen procedure:** research script `phase6b1_run.py`
(sha256 b8c2ac83bba811f3…), reviewer-only; it scores through the same
`meter.score`/`integrated.infer` path this package ships. Comparator: MML-GRM
EAP via `comparators/fit_irt_baseline.R --dimensions 1`.

**Reproduce:** build `dpq_responses.csv` per the schema above, fit the
comparator, then:

```
python run_meter.py --responses dpq_responses.csv \
    --comparator comparator_scores.csv --output result.json
```
