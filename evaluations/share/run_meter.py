"""Score the SHARE EURO-D response matrix with the frozen paper model.

Prepare the response CSV per README.md (item mapping in this directory), then:

    python run_meter.py --responses share_eurod.csv --group-column         --comparator comparator_scores.csv --output result.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import main

if __name__ == "__main__":
    main(factor_map=None, metadata={"data_provenance": "real", "n_categories": 2})
