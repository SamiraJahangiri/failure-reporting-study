"""Verify unchanged FTA inputs, recovering missing pinned originals if requested."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset import recover_missing, verify_inputs
from utils.io import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fetch-missing", action="store_true", help="Fetch only missing original upstream files at v0.3.0")
    parser.add_argument("--source-bundle", type=Path, help="Existing bundle directory/ZIP, for missing prebuilt inputs or policies")
    parser.add_argument("--output", type=Path, help="Optional verification JSON; otherwise print only")
    args = parser.parse_args()
    recovered = recover_missing(args.root, args.source_bundle, args.fetch_missing)
    verification = verify_inputs(args.root)
    verification["recovered_missing_files"] = recovered
    if args.output:
        atomic_write_json(args.output, verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
