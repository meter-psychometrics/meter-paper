# SAPA planned-missingness personality — Table 1 row 7

**Paper result:** 23,679 x 696 response matrix with 12.4% of cells observed by
design; across five SPI-derived scales METER agreed with full-information
MML-GRM at **0.9672-0.9842**. Reported numbers:
`derived_results/sapa/phase6b3_sapa_v2.json` (per-scale task records with
task_ids identifying the five scales).

**Source data:** SAPA-Project 2013-2014 archive (Condon & Revelle, Journal of
Open Psychology Data 2015), CC0/public-domain. The licence permits
redistribution, but the archive is large; obtain it from the source
(dataverse) and see the governance record `sapa_personality_2013_2014.yaml`.

**Expected input:** one CSV per scale (15 items each; the frozen artifact's
task records identify the scales and item counts): rows = respondents, item
columns in the archive's order, 6-point codes 0-5, unobserved = empty. The
scoring is scale-wise unidimensional with the observation mask carrying the
planned missingness — no imputation anywhere.

**Frozen procedure:** research script `phase6b3_sapa_run_v2.py`
(sha256 42f113da68879d7a…), reviewer-only.

**Reproduce:** per scale:

```
python run_meter.py --responses sapa_scale.csv \
    --comparator comparator_scores.csv --output result.json
```
