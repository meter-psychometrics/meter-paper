"""METER paper-reproducibility subset of the frozen ``human_measurement`` stack.

This package is a MINIMAL, verbatim subset of the frozen research
implementation behind the METER Nature Machine Intelligence paper. It contains
exactly the modules required to load the frozen paper checkpoints and perform
frozen inference: model architectures, the graded-response decoders, the
canonical input containers, the routing interface and the capability router.

It contains NO training code, NO synthetic world generators, and NO modules
for capabilities outside the paper. Files listed in ``RELEASE_AUDIT.md`` as
"verbatim" are byte-identical to the frozen research repository at the commit
recorded in ``paper_manifest.json``; files listed as "trimmed" carry only
deletions (plus this notice), never behavioural edits, and their equivalence
to the frozen implementation is asserted by the fixture tests in ``tests/``.
"""

SCHEMA_VERSION = "1.0.0"

__all__ = ["SCHEMA_VERSION"]
