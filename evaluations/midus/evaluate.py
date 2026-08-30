"""Agreement metrics between existing METER and comparator score files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json

import numpy as np

from common import evaluate, read_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--meter", type=Path, required=True)
parser.add_argument("--comparator", type=Path, required=True)
parser.add_argument("--group-column", action="store_true")
args = parser.parse_args()
meter, groups = read_matrix(args.meter, args.group_column)
comparator, _ = read_matrix(args.comparator, args.group_column)
print(json.dumps(evaluate(meter.ravel() if meter.shape[1] == 1 else meter,
                          comparator, groups), indent=2))
