"""Score the ICAR-16 (4 cognitive factors) battery with the frozen supplied-structure model.

``factor_map.json`` in this directory carries the item-to-factor assignment
(0-based, one entry per item column, in README column order). Then:

    python run_meter.py --responses responses.csv         --comparator mirt_factor_scores.csv --output result.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import main

if __name__ == "__main__":
    factor_map = json.loads((Path(__file__).parent / "factor_map.json").read_text())
    main(factor_map=factor_map, metadata={"data_provenance": "real", "construct": "cognitive ability"})
