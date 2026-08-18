# SHARE EURO-D, waves 9 and 8 — Table 1 rows 3-4

**Paper result:** wave 9 (n = 66,812, 28 countries): pooled r = **0.9851**
(participant bootstrap 0.9849-0.9854), per-country 0.9674-0.9899, every
country above the prospectively fixed 0.95 bar. Wave 8 temporal replication
(n = 51,732, 27 countries): pooled **0.9847**. Reported numbers:
`derived_results/share/share_a_w{9,8}_result.json`, including the frozen gate
record (GATE_SA) and the bootstrap blocks.

**Source data:** SHARE Release 9.0.0 (DOI: 10.6103/SHARE.w1.900 through
.w9.900), individual registered access via SHARE-ERIC. SHARE's Conditions of
Use permit local scientific AI processing but PROHIBIT entering SHARE
microdata into applications that are not fully self-administered (hosted AI
tools included). The frozen analysis honoured this by strictly local
execution; any reproduction must do the same. Participant-level data never
enters this repository.

**Expected input:** rows = respondents; first column = country (use
`--group-column`); the 12 EURO-D items taken from the official generated
variables (gv_health euro1..euro12), binary coding per the official
derivation; complete cases per the frozen eligibility rule (all 12 items
observed).

**Item mapping:** `item_mapping.yaml` (frozen). The full pre-registration —
wave selection, eligibility, comparator, bootstrap and pass criterion,
committed together with hashes of all analysis code and the eight source
archives BEFORE any response was scored — is preserved in the research
repository (`share_master_validation_charter.json`, `share_freeze_pins.json`;
reviewer-only, as the governance record references internal quarantine paths).

**Frozen procedure:** research scripts `share_a_extract.py`
(sha256 060953371f3d2c83…) and `share_a_run.py` (5f29f3388f1add28…),
reviewer-only. Comparator: country-wise MML-GRM EAP (R mirt 1.46.1).

**Reproduce:**

```
python run_meter.py --responses share_eurod.csv --group-column \
    --comparator comparator_scores.csv --output result.json
```
