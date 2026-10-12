#!/usr/bin/env python3
"""Export every prior pilot score without changing labels or source files.

Usage:
 python export_pilot_scores.py --source-dir /path/to/originals --output-dir /path/to/new/exports

Retains all original CSV columns, the complete provisional scoring payload,
exact evidence spans, uncertainty, and manual corrections. No confidence values
are invented. The earlier scoring file did not contain full prompt/report
bodies or activation arrays, so no scoring payload fields need to be dropped.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

SOURCE_NAME = "fta_diagnostic_pilot_scores.csv"
OUTPUT_NAME = "pilot_corrected_scores.jsonl.gz"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean(value):
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return None
    raise ValueError("Unknown Boolean representation: " + repr(value))


def transform(row):
    output = {key: value for key, value in row.items() if key != "payload"}
    output["scoring_payload"] = json.loads(row["payload"])
    reviewed = boolean(row["manual_reviewed"])
    output["manual_corrections"] = {
        "reviewed": reviewed,
        "wrong_completion_header": boolean(row["manual_wrong_header"]),
        "made_up_facts": boolean(row["manual_made_up_facts"]),
        "honest_failure_admission": boolean(row["manual_honest_failure"]),
        "note": row["manual_note"],
    }
    return output


def run(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / SOURCE_NAME
    source_sha = sha(source)
    source_size = source.stat().st_size
    old_manifest = output_dir / "export_manifest.json"
    old_exports = json.loads(old_manifest.read_text()) if old_manifest.exists() else None
    old_manifest_sha = sha(old_manifest) if old_manifest.exists() else None
    historical = source_dir / "fta_exact_replay_20261012_170305" / "preserved_previous_runs.json"
    prior = json.loads(historical.read_text()) if historical.exists() else {}
    for name, spec in prior.items():
        assert sha(source_dir / name) == spec["sha256"], "Changed prior artifact: " + name

    counts = collections.Counter()
    model_counts = collections.Counter()
    correction_counts = collections.Counter()
    fingerprints = collections.Counter()
    versions = collections.Counter()
    output = output_dir / OUTPUT_NAME
    raw = output.open("xb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    stream = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    keys = set()
    with source.open(encoding="utf-8", newline="") as original:
        reader = csv.DictReader(original)
        header = reader.fieldnames
        for row in reader:
            record = transform(row)
            assert record["pilot_key"] not in keys
            keys.add(record["pilot_key"])
            counts["reports"] += 1
            counts["manually_reviewed"] += record["manual_corrections"]["reviewed"] is True
            counts["manual_review_status_unrecorded"] += record["manual_corrections"]["reviewed"] is None
            counts["provisional_payloads"] += record["scoring_payload"].get("scoring_provisional") is True
            counts["unresolved_provisional_payloads"] += record["scoring_payload"].get("unresolved") is True
            model_counts[record["model"]] += 1
            versions[record["scoring_payload"].get("scorer_version", "unrecorded")] += 1
            for field, source_field in [
                    ("wrong_completion_header", "wrong_completion_header"),
                    ("made_up_facts", "explicit_fabricated_requested_facts"),
                    ("honest_failure_admission", "honest_failure_admission")]:
                if record["manual_corrections"]["reviewed"] is True:
                    value = record["manual_corrections"][field]
                    if value is not None:
                        correction_counts[field + "_reviewed"] += 1
                        correction_counts[field + "_positive"] += value is True
                        correction_counts[field + "_differs_from_provisional"] += value != boolean(row[source_field])
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    stream.close()
    raw.close()
    assert counts["reports"] == 13680
    assert model_counts == {"Qwen/Qwen3-8B": 6840, "google/gemma-2-9b-it": 6840}

    # Compare every exported row against a separately reread source row.
    recovered = {}
    with gzip.open(output, "rt", encoding="utf-8") as exported:
        for text in exported:
            record = json.loads(text)
            recovered[record["pilot_key"]] = record
    assert len(recovered) == 13680
    with source.open(encoding="utf-8", newline="") as original:
        for row in csv.DictReader(original):
            assert recovered[row["pilot_key"]] == transform(row)
    assert sha(source) == source_sha and source.stat().st_size == source_size
    for name, spec in prior.items():
        assert sha(source_dir / name) == spec["sha256"], "Changed prior artifact: " + name
    if old_exports is not None:
        assert sha(old_manifest) == old_manifest_sha
        for name, spec in old_exports["exports"].items():
            assert sha(output_dir / name) == spec["sha256"], "Previous export changed: " + name

    assert output.stat().st_size < 3_000_000
    script = Path(__file__)
    manifest = {
        "schema_version": "1.0",
        "source": {"name": SOURCE_NAME, "bytes": source_size, "sha256": source_sha,
                   "csv_columns": header},
        "files": {
            OUTPUT_NAME: {"bytes": output.stat().st_size, "sha256": sha(output)},
            script.name: {"bytes": script.stat().st_size, "sha256": sha(script)}
        },
        "counts": dict(counts),
        "counts_by_model": dict(model_counts),
        "manual_correction_counts": dict(correction_counts),
        "scorer_versions": dict(versions),
        "schema": {
            "original_columns": "Every original non-payload CSV field preserved verbatim as a string.",
            "scoring_payload": "Entire original JSON score payload, including provisional labels, header interpretation, matching text/spans, evidence eligibility, ambiguity, uncertainty and limitations.",
            "manual_corrections": "Convenience typed view of existing manual_reviewed, manual_wrong_header, manual_made_up_facts, manual_honest_failure and manual_note columns; empty original values become null.",
            "confidence": "No confidence ratings were recorded in this source; none are invented."
        },
        "dropped_fields": [],
        "transformations": [
            "Original CSV payload parsed into scoring_payload; every other column retained verbatim.",
            "Manual corrections preserved separately from provisional lexical labels.",
            "Only explicitly reviewed axes have a manual correction; unsupported-completion and other unreviewed axes remain provisional.",
            "No new scoring, answer edits or adjudication performed.",
            "No full prompt/report bodies or activation arrays occurred in this scoring payload.",
            "Deterministic UTF-8 JSONL gzip uses mtime zero and no embedded filename."
        ],
        "validation": {
            "reports": 13680, "unique_keys": 13680,
            "every_exported_record_equals_transformed_source": True,
            "source_unchanged": True, "previous_artifacts_checked": len(prior),
            "previous_artifacts_unchanged": True,
            "previous_public_exports_unchanged": old_exports is not None,
            "compressed_output_under_3_MB": True,
        }
    }
    with (output_dir / "pilot_scores_export_manifest.json").open("x", encoding="utf-8") as exported:
        json.dump(manifest, exported, sort_keys=True, ensure_ascii=False, indent=2)
        exported.write("\n")
    print(json.dumps({"output_bytes": output.stat().st_size, "counts": dict(counts),
                      "manual_corrections": dict(correction_counts)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
