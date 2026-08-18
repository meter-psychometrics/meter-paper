# RELEASE_AUDIT — meter-nmi-paper candidate

Date: 2026-08-18. Source: research branch `claude/nmi-validation-programme` @ `9ba3ea1`.
Status: **candidate only — not pushed anywhere, no remote exists for this
directory, nothing is public.** Classification per file: what it is, why it is
necessary, and its provenance class:

- **V** — verbatim copy of the frozen research implementation (byte-identical
  except path-free relocation);
- **T** — trimmed copy: deletions only, no behavioural edits, equivalence
  asserted by the fixture tests;
- **E** — edited copy: loader/provenance internals replaced (documented inline);
  bit-identical outputs asserted against the original frozen stack;
- **N** — newly written for this package;
- **A** — frozen evidence artifact, copied unmodified (aggregates only).

Automated checks passed before this audit was written: 23/23 tests, including
the nine required release tests, the P4B replay (40 executable decisions), the
restricted-content scan, and the unpublished-import guard. Fixture outputs are
byte-identical to the outputs of the original frozen stack.

## Root

| File | Class | Why necessary |
|---|---|---|
| README.md | N | package purpose, API, model identity, scope boundary |
| MODEL_LINEAGE.md | N | the accepted canonical lineage: three frozen states, training-world families, freeze dates, analysis→model map |
| RELEASE_AUDIT.md | N | this audit |
| CITATION.cff | N | citation metadata (author list pending) |
| requirements.txt, environment.yml | N | frozen environment pins (from the research freeze records) |
| paper_manifest.json | N | model identifiers, state digests, contract version/hash, source commit, env versions, per-model paper-result map, weights-release gate |
| .gitignore | N | blocks accidental commit of user-prepared real-data inputs |

## checkpoints/ — the frozen paper models

| File | Class | Why necessary |
|---|---|---|
| phase5e-arm5_combined.pt | E (state-only re-serialization; state digest `224a701b…` unchanged; training history stripped) | the model behind every real-data 1-D result in Table 1 |
| phase3-stage2-arm1_beta05.pt | E (same treatment; `0d571638…` unchanged) | the synthetic-recovery claim (r = 0.9388) and the uncertainty scalar |
| mapped_known_structure_seed8100.pt | E (materialised once via the frozen seed-8100 procedure, asserted to the charter pin `8d427520…`) | the supplied-structure results (MIDUS/IPIP/ICAR) |
| CHECKSUMS.sha256 | N | file-level integrity, complementing the in-loader state-digest assertions |

**Weights-release gate:** DATA_GOVERNANCE rule 9 (research repo) bars weight
release until a memorisation/privacy-leakage evaluation is recorded; the
manuscript's availability statement promises the same gating. All three states
were trained exclusively on synthetic worlds and frozen before any real
dataset was opened, so the evaluation is discharge-able by a recorded
determination — an owner decision. Until then this directory is
**reviewer-only**; the hashes are public regardless.

## src/human_measurement/ — minimal frozen inference subset

| File | Class | Notes |
|---|---|---|
| model/schemas.py, model/features.py, model/network.py, model/multidim.py, model/longitudinal.py | V | architectures + GRM decoders for the three paper models |
| interface.py | E | frozen routing interface; `frozen_provenance()` reads the pin constants from `frozen_pins.py` instead of importing the unreleased capability modules (values verbatim; identity asserted by test) |
| integrated.py | V | the frozen integrated entry point (`infer`) — out-of-paper branches import lazily and are unreachable from the released API |
| capability_router.py | V | contract-3.5 runtime predicates; **is Fig. 4** |
| longitudinal_design.py | V | design descriptors + routing consumed by the router |
| workbench/protocols.py | V | the frozen capability contract v3.5 record (`REAL_WORLD_CAPABILITY`) with its content hash |
| frozen_artifacts.py | V | content-hash helper used by provenance |
| frozen_pins.py | N | the four provenance pin dicts, verbatim values (documented extraction) |
| model/multimodal.py | T | ONLY `psychometric_message` (the 1-D read-out); the fusion pathway is not released |
| ingestion/canonical.py | T | `CanonicalDataset` container only; the polars mapping surface is not released |
| ingestion/routing.py | T | `to_world_tensors` only; the ingestion workflow is not released |
| mapped/model.py | V | `KnownStructureGrm` + `factor_adequacy` |
| mapped/mapping.py | T | `SuppliedMapping` + `mirt_model_string`; the synthetic corruption constructors (generator-coupled) are not released |
| battery/model.py | T | `spectral_item_features` only; the M7 discovery model is not released |
| `__init__` files | T/N | trimmed exports; no generator or trainer imports |

## meter_reference/ — the public API

| File | Class | Notes |
|---|---|---|
| api.py | E | the release `meter.score` (refusal-first) with two edits: `_models()` is a paper-specific loader replicating `frozen_longitudinal()` exactly (incl. `gate_floor = 0.0`, no `eval()`), and `_mapped_model()` is load-only (the trainer-based reproduction stays in the research repo). `weights_only=True` loads. |
| inference.py | N | `load_model("meter-1d-paper" / "meter-multidim-paper")` facade + `load_phase3_reference()`; no training surface |
| preprocessing.py | N (re-exports) | the frozen preprocessing path, importable in isolation |
| grm_decoder.py | N (re-exports) | the structured GRM decoding step, documented |
| `__init__.py` | N | public surface |

## fixtures/ + tools/

| File | Class | Notes |
|---|---|---|
| fixtures/equivalence_fixtures.npz, equivalence_manifest.json | N | inputs + outputs + SHA-256 digests produced by the ORIGINAL frozen stack; the bridge for release test 2 |
| tools/generate_fixtures_from_frozen.py | N | regenerates the fixtures; runs only inside the research repository by design |

## derived_results/ — frozen aggregate evidence (all class A)

Every artifact was scanned: aggregates only, longest numeric sequence 13
(item-level); no participant-level values anywhere.

| Directory | Files | Paper claim |
|---|---|---|
| synthetic/ | phase3c.json, phase3c_items.json, phase3-stage2-arm1_beta05.json | r = 0.9388 vs 0.9438; coverage 0.8288 → 0.9479 |
| nhanes/ | DPQ_J locked + DPQ_I development results | Table 1 row 1 (+ Extended Data) |
| ess/ | phase6b2_ess7_cesd8.json | Table 1 row 2; Fig. 2b |
| share/ | share_a_w9/w8_result.json | Table 1 rows 3-4; Fig. 2c; the pre-registered gate |
| midus/ | d9a_transfer, d9b_replication + frozen gates | Table 1 row 5; the Phi-gate FAILURE |
| ipip50/ | d9e_transfer.json | Table 1 row 6 |
| sapa/ | phase6b3_sapa_v2.json | Table 1 row 7; Fig. 3d |
| icar/ | d9d_transfer + permanent closure | Table 1 row 8; the D9D FAIL |
| m8b/ | phase8b_confirmatory, phase8_closeout | Methods: synthetic multidim support |
| p4b/ | closeout, fresh challenge, original-P4 rerun, non-regression, frozen protocol | Fig. 4 (FA 0/63, FR 0/41, 9 bit-identical fixtures, 16 exact fallbacks) |

## evaluations/ — real-data reproduction harnesses

Per dataset: README (source access, licence + AI-policy restrictions preserved
verbatim in intent, expected schema, frozen-procedure hashes), frozen
governance/charter/mapping records (class A), `run_meter.py` + `evaluate.py`
(class N thin harnesses over the released API), and for the multidimensional
batteries the frozen `factor_map.json` (N, values transcribed from the frozen
charters). `common.py` (N) is the shared scoring/agreement harness. The
internal extraction pipelines (which touch licensed raw file formats and
internal paths) are reviewer-only; their SHA-256 hashes are cited in each
README so the reviewer-only bundle is verifiable.

## comparators/, examples/, figures/, tests/

| File | Class | Notes |
|---|---|---|
| comparators/fit_irt_baseline.R | V | the frozen MML-GRM EAP comparator |
| comparators/README.md | N | incl. the confirmatory-MIRT specification via `mirt_model_string` |
| examples/synthetic_example.py, example_response_matrix.csv | N | synthetic-only demonstration (loads, masks, scores, frozen weights, refusal) |
| figures/reproduce_figures.py | N | regenerates Table 1 + Fig. 2-4 numerical panels from derived_results; asserts every headline value against the manuscript |
| tests/ (7 files) | N | the nine required release tests + P4B replay + hygiene guards |

## Explicitly NOT included (and why)

- Synthetic measurement-world generators, all trainers, optimizer code —
  commercially sensitive beyond what the paper's claims require; frozen
  inference is fully reproducible without them (fixture-proven).
- DIF, linking, change-detection, multimodal-fusion implementations —
  capabilities outside the focused paper; their provenance pins ship as data.
- The M7 structure-discovery model — closed capability; the paper reports
  only its failure, which ships as the frozen refusal string + contract entry.
- SHARE governance record and all extraction pipelines — reference internal
  quarantine paths; reviewer-only.
- All real data, dataset archives, participant-level anything — prohibited.
- Longitudinal (UKHLS/ELSA/SHARE-B), multimodal (M10A/B), P1/P2/P3/P5/P7
  planned-analysis evidence, prior-art/patent material, internal milestone
  documentation, roadmaps — outside the paper.

## Residual risks and open items for the owner

1. **Weights staging vs rule 9** — see the checkpoints section; owner must
   either record the memorisation determination or hold `checkpoints/` back
   at publication time (the package degrades gracefully: loaders name the
   missing file and the hash expectations).
2. **Architecture disclosure** — shipping `network/multidim/longitudinal.py`
   disclosively documents the model family (already described in Methods).
   The alternative (opaque TorchScript) was rejected because it breaks the
   frozen state-digest identity and NMI code review expects source.
3. **`workbench/protocols.py`** carries capability statuses for out-of-paper
   modules (longitudinal, multimodal entries). The focused manuscript names
   these programmes as existing-but-out-of-scope, and the contract hash is
   pinned in the paper evidence, so the record ships whole; trimming it would
   change the pinned content hash.
4. **Training-code classification (per the release brief):** class B applies
   in part — the manuscript's availability statement promises "synthetic
   generators, model code" deposition. Recommended resolution: the public
   package stays as built; a REVIEWER-ONLY bundle adds the generator +
   training modules for the two paper models (`simulation/`, `model/build.py`,
   `model/phase3.py`, `mapped/train.py`, `battery/worlds.py`, the phase-3
   pretraining script) without any later-phase capability code. Do not build a
   generalized `train_meter.py`.
