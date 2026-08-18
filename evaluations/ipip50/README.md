# IPIP-50 Big Five Factor Markers — Table 1 row 6

**Paper result:** a DIFFERENT 50-item instrument scored with the same frozen
multidimensional weights: factor-wise **0.9596-0.9839**, median 0.9791, zero
sign flips (n = 20,000, frozen seeded selection from 603,322 eligible rows —
`d9e_sample_selection.json`). Reported numbers:
`derived_results/ipip50/d9e_transfer.json`.

**Source data:** Open-Source Psychometrics Project, IPIP Big Five Factor
Markers raw archive (public download). The instrument items are Goldberg's
public-domain lexical markers.

**Expected input:** 50 item columns in the frozen block order EXT1..EXT10,
EST1..EST10, AGR1..AGR10, CSN1..CSN10, OPN1..OPN10; valid responses 1-5
recoded to 0-4; the archive's 0 skip code to missing; reverse-keying per the
published Goldberg key (recorded in `d9e_charter_populated_ipip50.json`).
`factor_map.json` carries the frozen assignment.

**Frozen procedure:** research scripts `d9e_prepare_sample.py`
(7acdfea4244fbe9c…) and `d9e_transfer_run.py` (d5ae7904a6e8d9c4…),
reviewer-only; populated charter in this directory.

**Reproduce:**

```
python run_meter.py --responses ipip50.csv \
    --comparator mirt_factor_scores.csv --output result.json
```
