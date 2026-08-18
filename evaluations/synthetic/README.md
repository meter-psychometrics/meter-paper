# Synthetic truth recovery — Results §1

**Paper result:** on 40 previously unseen one-dimensional worlds the frozen
31,051-parameter Phase 3 core recovered the known person latent at median
Pearson r = **0.9388** (IQR 0.9249-0.9465; worst world 0.8638) against a
per-world MML-GRM specialist at **0.9438**; the checkpoint-specific
multiplicative uncertainty correction improved nominal 95% coverage from
**0.8288 to 0.9479** on untouched validation worlds without changing point
estimates.

**Reported numbers in this package:** `derived_results/synthetic/phase3c.json`
(recovery and coverage) and `phase3-stage2-arm1_beta05.json` (the frozen
benchmark record whose `mml_reference` block is the specialist comparator).
Regenerate the panel with `python figures/reproduce_figures.py`.

**What reproduces here:** `verify_synthetic_core.py` loads the frozen Phase 3
checkpoint (hash-asserted against `0d571638…`), runs it on the packaged
synthetic example matrix, and confirms the structural decoder properties the
Methods describe: positive discriminations and strictly ordered thresholds
from the graded-response heads.

**What requires the research repository (reviewer-only):** regenerating the
40 unseen evaluation worlds requires the synthetic measurement-world
generator, which is deliberately not part of this package. The generator, the
world seeds and the evaluation harness are preserved frozen in the research
repository and can be made available for editorial review.
