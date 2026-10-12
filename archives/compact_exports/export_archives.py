#!/usr/bin/env python3
"""Lossless report exports of earlier exploratory FTA archives.

Usage:
  python export_archives.py --source-dir /path/to/originals --output-dir /path/to/new/exports

The source files are read-only. Outputs are created exclusively. Model weights,
activation arrays, repeated prompt bodies, native prefills, and old automated
score payloads are not copied into report rows. Prompt bodies and scalar probe
predictions are retained in deterministic gzip files.
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

SCHEMA_VERSION = "1.0"
PILOT_NAME = "fta_results.csv"
DIAGNOSTIC_NAME = "fta_diagnostic_results.csv"
SCORE_NAME = "fta_diagnostic_pilot_scores.csv"
METADATA = ["key", "model", "revision", "base_task_id", "family_id", "split",
            "state", "condition", "policy", "intervention"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def line(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def gzip_text_new(path):
    raw = Path(path).open("xb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, text


def write_jsonl(path, records):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        for record in records:
            stream.write(line(record))


def write_gzip_jsonl(path, records):
    raw, stream = gzip_text_new(path)
    try:
        for record in records:
            stream.write(line(record))
        stream.close()
    finally:
        raw.close()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(row) for row in stream if row.strip()]


def run(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_names = [PILOT_NAME, DIAGNOSTIC_NAME]
    if (source_dir / SCORE_NAME).exists():
        source_names.append(SCORE_NAME)
    before = {name: {"bytes": (source_dir / name).stat().st_size,
                     "sha256": sha(source_dir / name)} for name in source_names}

    # Preserve every previously recorded artifact, including files not exported.
    historical = source_dir / "fta_exact_replay_20261012_170305" / "preserved_previous_runs.json"
    prior = json.loads(historical.read_text()) if historical.exists() else {}
    for name, spec in prior.items():
        assert sha(source_dir / name) == spec["sha256"], "Changed prior artifact: " + name

    pilot_reports = collections.defaultdict(list)
    pilot_fingerprints = collections.defaultdict(set)
    pilot_prompts, scenarios, originals, summaries = {}, [], [], []
    pilot_types, tracks = collections.Counter(), collections.Counter()
    probe_fields = ["fingerprint"] + METADATA + ["predicted_supported", "probability_supported"]
    probe_path = output_dir / "pilot_probe_predictions.csv.gz"
    probe_raw, probe_stream = gzip_text_new(probe_path)
    probe_writer = csv.DictWriter(probe_stream, fieldnames=probe_fields, lineterminator="\n")
    probe_writer.writeheader()
    probe_rows = 0
    with (source_dir / PILOT_NAME).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            payload = json.loads(row["payload"])
            kind = row["record_type"]
            pilot_types[kind] += 1
            if kind == "prediction":
                prompt = payload["prompt"]
                prompt_hash = text_sha(prompt)
                assert prompt_hash not in pilot_prompts or pilot_prompts[prompt_hash] == prompt
                pilot_prompts[prompt_hash] = prompt
                report = {key: row[key] for key in METADATA}
                pilot_fingerprints[row["model"]].add(row["fingerprint"])
                report["prompt_sha256"] = prompt_hash
                report.update({key: value for key, value in payload.items()
                               if key not in ("prompt", "score")})
                # The old CSV did not record an EOS/finish_reason field.
                # Preserve its actual truncation flag rather than inventing EOS.
                tracks[(row["model"], row["intervention"], payload["status_only"])] += 1
                pilot_reports[row["model"]].append(report)
            elif kind == "probe_prediction":
                record = {key: row[key] for key in ["fingerprint"] + METADATA}
                record.update(payload)
                assert set(record) == set(probe_fields), "Unexpected probe payload fields"
                probe_writer.writerow(record)
                probe_rows += 1
            elif kind == "scenario":
                scenarios.append({"source_key": row["key"],
                                  "source_fingerprint": row["fingerprint"], **payload})
            elif kind == "original_scenario":
                originals.append({"source_key": row["key"],
                                  "source_fingerprint": row["fingerprint"], **payload})
            elif kind == "summary":
                summaries.append({"source_key": row["key"],
                                  "source_fingerprint": row["fingerprint"], **payload})
            else:
                raise ValueError("Unrecognized pilot record type: " + kind)
    probe_stream.close()
    probe_raw.close()
    assert pilot_types == {"prediction": 13680, "probe_prediction": 46800,
                           "scenario": 1500, "original_scenario": 100, "summary": 4}
    filenames = {}
    for model, reports in sorted(pilot_reports.items()):
        suffix = model.split("/")[-1].lower().replace("-", "_")
        name = "pilot_reports_" + suffix + ".jsonl"
        write_jsonl(output_dir / name, reports)
        filenames[model] = name
    write_gzip_jsonl(output_dir / "pilot_prompts.jsonl.gz",
                     [{"prompt_sha256": key, "prompt": value}
                      for key, value in sorted(pilot_prompts.items())])
    write_gzip_jsonl(output_dir / "pilot_scenarios.jsonl.gz", scenarios)
    write_jsonl(output_dir / "pilot_original_scenarios.jsonl", originals)
    write_jsonl(output_dir / "pilot_summaries.jsonl", summaries)

    diagnostic_reports, diagnostic_prompts, contexts = [], {}, {}
    diagnostic_stages = collections.Counter()
    with (source_dir / DIAGNOSTIC_NAME).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            payload = json.loads(row["payload"])
            context, old_report = payload["context"], payload["report"]
            prompt = context["prompt"]
            prompt_hash = text_sha(prompt)
            assert prompt_hash == context["prompt_sha256"]
            assert prompt_hash not in diagnostic_prompts or diagnostic_prompts[prompt_hash] == prompt
            diagnostic_prompts[prompt_hash] = prompt
            clean_context = {key: value for key, value in context.items() if key != "prompt"}
            if context["key"] in contexts:
                assert contexts[context["key"]] == clean_context
            contexts[context["key"]] = clean_context
            record = {key: value for key, value in row.items() if key != "payload"}
            record["revision"] = payload["revision"]
            record["protocol_sha256"] = payload["protocol_sha256"]
            record["source_payload_sha256"] = text_sha(row["payload"])
            record["context_key"] = context["key"]
            record["prompt_sha256"] = prompt_hash
            record["prefill_text_sha256"] = text_sha(old_report["prefill_text"])
            record["report"] = {key: value for key, value in old_report.items()
                                if key != "prefill_text"}
            record["response_sha256"] = text_sha(old_report["response"])
            record["raw_generated_text_sha256"] = text_sha(old_report["raw_generated_text"])
            diagnostic_reports.append(record)
            diagnostic_stages[row["stage"]] += 1
    assert len(diagnostic_reports) == 768
    assert diagnostic_stages == {"main": 480, "confirmation": 288}
    write_jsonl(output_dir / "diagnostic_reports.jsonl", diagnostic_reports)
    write_gzip_jsonl(output_dir / "diagnostic_prompts.jsonl.gz",
                     [{"prompt_sha256": key, "prompt": value}
                      for key, value in sorted(diagnostic_prompts.items())])
    write_jsonl(output_dir / "diagnostic_contexts.jsonl",
                [contexts[key] for key in sorted(contexts)])

    # Round-trip validation independently reads the newly published files.
    read_pilot = [record for name in filenames.values()
                  for record in read_jsonl(output_dir / name)]
    read_diag = read_jsonl(output_dir / "diagnostic_reports.jsonl")
    assert len(read_pilot) == 13680
    assert len({row["key"] for row in read_pilot}) == 13680
    assert len({row["key"] for row in read_diag}) == 768
    for prefix, reports, original_prompts in [
            ("pilot", read_pilot, pilot_prompts),
            ("diagnostic", read_diag, diagnostic_prompts)]:
        recovered = {row["prompt_sha256"]: row["prompt"]
                     for row in read_jsonl(output_dir / (prefix + "_prompts.jsonl.gz"))}
        assert recovered == original_prompts
        for prompt_hash, prompt in recovered.items():
            assert text_sha(prompt) == prompt_hash
        for record in reports:
            assert record["prompt_sha256"] in recovered
            body = record if prefix == "pilot" else record["report"]
            original = next(row for row in pilot_reports[record["model"]] if row["key"] == record["key"]) if prefix == "pilot" else next(row for row in diagnostic_reports if row["key"] == record["key"])
            original_body = original if prefix == "pilot" else original["report"]
            assert text_sha(body["response"]) == text_sha(original_body["response"])
            assert text_sha(body["raw_generated_text"]) == text_sha(original_body["raw_generated_text"])
            assert record == original
    with gzip.open(probe_path, "rt", encoding="utf-8", newline="") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 46800
    for name, spec in before.items():
        assert (source_dir / name).stat().st_size == spec["bytes"]
        assert sha(source_dir / name) == spec["sha256"], "Source changed: " + name
    for name, spec in prior.items():
        assert sha(source_dir / name) == spec["sha256"], "Changed prior artifact: " + name

    output_files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            size = path.stat().st_size
            assert size < 10_000_000, "Publication file exceeds 10 MB: " + path.name
            output_files[path.name] = {"bytes": size, "sha256": sha(path)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "archive_status": "Exploratory pilot and diagnostic; distinct from exact task replay.",
        "source_files": before,
        "source_record_counts": {
            PILOT_NAME: dict(pilot_types),
            DIAGNOSTIC_NAME: dict(diagnostic_stages)},
        "exports": output_files,
        "counts": {"pilot_reports": len(read_pilot), "pilot_unique_prompts": len(pilot_prompts),
                   "pilot_scenarios": len(scenarios), "pilot_original_scenarios": len(originals),
                   "pilot_summaries": len(summaries), "pilot_probe_predictions": probe_rows,
                   "diagnostic_reports": len(read_diag),
                   "diagnostic_unique_prompts": len(diagnostic_prompts),
                   "diagnostic_contexts": len(contexts)},
        "pilot_reports_by_model": {model: {"file": filenames[model], "count": len(rows), "source_fingerprints": sorted(pilot_fingerprints[model])}
                                   for model, rows in sorted(pilot_reports.items())},
        "pilot_tracks": [{"model": model, "intervention": intervention,
                          "status_only": status_only, "count": count}
                         for (model, intervention, status_only), count in sorted(tracks.items())],
        "schemas": {
            "pilot_reports": "CSV scalar metadata + generation payload excluding prompt and old score; per-model source fingerprint is stored once in pilot_reports_by_model. Donor is a small provenance mapping. Both response and raw_generated_text are unchanged. No recorded EOS reason is synthesized.",
            "pilot_prompts": "Unique prompt_sha256 and exact prompt; UTF-8 JSONL in deterministic gzip.",
            "pilot_scenarios": "Source key/fingerprint and original generated scenario payload, in deterministic gzip JSONL.",
            "pilot_original_scenarios": "Source key/fingerprint and original author scenario payload.",
            "pilot_summaries": "Source key/fingerprint and all original summary payload fields.",
            "pilot_probe_predictions": "All 46,800 original scalar predictions, metadata and probabilities in deterministic gzip CSV; no activation arrays.",
            "diagnostic_reports": "CSV scalar metadata + revision/protocol/context hashes + report excluding repeated native prefill; both answer text fields unchanged. Old rule-based score payload omitted.",
            "diagnostic_prompts": "Unique prompt_sha256 and exact logical prompt; UTF-8 JSONL in deterministic gzip.",
            "diagnostic_contexts": "Unique original context keyed by context_key; repeated logical prompt replaced by original prompt_sha256."
        },
        "transformations": [
            "Sources are read-only; report text is neither edited nor rescored.",
            "Repeated logical prompts are deduplicated by SHA-256 and retained losslessly.",
            "Native prefill strings are omitted from diagnostic rows; exact prefill SHA-256 remains.",
            "Old provisional rule-based score payloads are omitted; corrected/manual scoring is published separately.",
            "All probe prediction rows are retained as scalar CSV; activations and checkpoints are excluded.",
            "JSON formatting and row ordering are deterministic; gzip timestamp and embedded filename are fixed."
        ],
        "excluded_originals": {
            PILOT_NAME: "Large mixed CSV replaced by these faithful exports.",
            DIAGNOSTIC_NAME: "Mixed CSV replaced by reports, deduplicated prompts and contexts; prefills and provisional score payloads omitted.",
            SCORE_NAME: "Large provisional automated scoring payload file omitted; original SHA-256 recorded when available; corrections/manual scores are separate."
        },
        "validation": {
            "pilot_report_keys_unique": True, "diagnostic_report_keys_unique": True,
            "all_prompt_hashes_resolve": True, "answer_hashes_match_round_trip": True,
            "all_probe_rows_retained": True, "source_files_unchanged": True,
            "previous_artifacts_checked": len(prior), "previous_artifacts_unchanged": True,
            "each_export_under_10_MB": True,
            "export_bytes_before_manifest": sum(value["bytes"] for value in output_files.values())
        }
    }
    with (output_dir / "export_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"counts": manifest["counts"],
                      "total_bytes": sum(path.stat().st_size for path in output_dir.iterdir()
                                         if path.is_file()),
                      "files": sorted(output_files) + ["export_manifest.json"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
