"""Release test 7: figure/table reproduction regenerates the recorded values.

Runs ``figures/reproduce_figures.py`` in-process; the script itself asserts
every headline number against the manuscript-printed value and returns
non-zero on any mismatch.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _run():
    spec = importlib.util.spec_from_file_location(
        "reproduce_figures", PACKAGE_ROOT / "figures" / "reproduce_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._failures.clear()
    return module


def test_reproduction_matches_manuscript():
    module = _run()
    assert module.main() == 0, module._failures


def test_panel_files_written_and_consistent():
    module = _run()
    assert module.main() == 0
    output = PACKAGE_ROOT / "figures" / "output"
    fig2 = json.loads((output / "fig2_panels.json").read_text())
    assert len(fig2["b_ess_per_country"]) == 21
    assert len(fig2["c_share_per_country"]["w9"]) == 28
    assert len(fig2["c_share_per_country"]["w8"]) == 27
    fig3 = json.loads((output / "fig3_panels.json").read_text())
    assert len(fig3["a_midus_refresher"]) == 5
    assert len(fig3["b_ipip50"]) == 5
    assert len(fig3["c_icar16"]) == 4
    assert len(fig3["d_sapa"]) == 5
    fig4 = json.loads((output / "fig4_panels.json").read_text())
    assert fig4["b_fresh_challenge"]["false_accepts"]["count"] == 0
    assert fig4["b_fresh_challenge"]["false_refusals"]["count"] == 0
    assert len(fig4["c_nonregression"]["supported_fixtures_bit_identical"]) == 9
