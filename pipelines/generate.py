"""One natural report per unchanged FTA task/policy and new official model."""
import argparse
import fcntl
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset import load_and_verify_inputs
from utils.io import append_jsonl, atomic_write_json, read_json, read_jsonl, sha256_file
from utils.models import digest, load_tokenizer, model_spec, native_input, verify_snapshot
from utils.runtime import ReportGenerator, utc_now


def response_id(spec, record):
    return "|".join((spec["repo_id"], record["scenario_id"], record["condition"], "0"))


def preflight(root, spec, require_weights=True):
    if spec["mode"] == "reuse_completed":
        raise ValueError("Existing Qwen/Gemma answers must be reused, never regenerated")
    inputs = load_and_verify_inputs(root)
    snapshot = verify_snapshot(spec, require_weights=require_weights)
    tokenizer = load_tokenizer(spec)
    prepared = [native_input(tokenizer, spec, record)[1] for record in inputs]
    result = {"checked_at": utc_now(), "model": spec["repo_id"], "revision": spec["revision"],
              "records": len(prepared), "distinct_native_input_hashes": len(set(row["native_input_token_ids_sha256"] for row in prepared)),
              "maximum_input_tokens": max(row["input_length"] for row in prepared),
              "snapshot": snapshot, "template_kwargs": spec.get("template_kwargs", {}),
              "native_system_user": True, "logical_messages_unchanged": True,
              "complete_weights_required": require_weights,
              "native_input_checks": [dict(scenario_id=record["scenario_id"], condition=record["condition"],
                                           **{name: prepared[index][name] for name in
                                              ("logical_input_sha256", "native_prompt_sha256", "native_input_token_ids_sha256", "input_length")})
                                      for index, record in enumerate(inputs)],
              "generation": spec["generation"], "model_inference_performed": False}
    atomic_write_json(root / "results" / ("preflight_" + spec["key"] + ".json"), result)
    blocker = root / "results" / ("blocker_" + spec["key"] + ".json")
    if blocker.exists():
        previous = read_json(blocker)
        atomic_write_json(blocker, dict(previous, status="resolved", resolved_at=utc_now(),
                                        resolution="Native input preflight passed; runtime readiness is checked before generation"))
    return inputs, result


def run(root, spec):
    inputs, checked = preflight(root, spec)
    results = root / "results"
    results.mkdir(exist_ok=True)
    answers = results / "answers.jsonl"
    with (results / ".answers.lock").open("a") as journal_lock:
        fcntl.flock(journal_lock, fcntl.LOCK_EX)
        completed = {row["response_id"]: row for row in read_jsonl(answers)} if answers.exists() else {}
    planned = {response_id(spec, record) for record in inputs}
    if planned.issubset(completed):
        print("Already saved all 300 answers for " + spec["repo_id"], flush=True)
        return
    settings = {"model": spec["repo_id"], "revision": spec["revision"], "generation": spec["generation"],
                "template_kwargs": spec.get("template_kwargs", {}),
                "inputs_sha256": sha256_file(root / "data/model_inputs.jsonl")}
    settings_hash = digest(settings)
    for key in planned.intersection(completed):
        if completed[key].get("generation_settings_sha256") != settings_hash:
            raise ValueError("Resume settings differ from saved answer; preserve the answer and investigate")
    generator = ReportGenerator(spec)
    manifest_path = results / ("runtime_" + spec["key"] + ".json")
    manifest = dict(generator.manifest(), frozen_settings=settings,
                    generation_settings_sha256=settings_hash, preflight=checked)
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["generation_settings_sha256"] != settings_hash or previous["hardware"]["packages"] != manifest["hardware"]["packages"]:
            raise ValueError("Resume runtime or frozen settings changed; the original manifest is preserved")
    else:
        atomic_write_json(manifest_path, manifest, overwrite=False)
    for index, record in enumerate(inputs):
        identity = response_id(spec, record)
        if identity in completed:
            continue
        try:
            report = generator.answer(record)
        except Exception as error:
            # A runtime exception produced no report. Preserve the event and leave this slot unsaved.
            _, provenance = native_input(generator.tokenizer, spec, record)
            event = dict(provenance, response_id=identity, model=spec["repo_id"], revision=spec["revision"],
                         scenario_id=record["scenario_id"], condition=record["condition"],
                         occurred_at=utc_now(), generation_settings_sha256=settings_hash,
                         error_type=type(error).__name__, reason=str(error), model_report_saved=False)
            with (results / ".answers.lock").open("a") as journal_lock:
                fcntl.flock(journal_lock, fcntl.LOCK_EX)
                append_jsonl(results / "generation_errors.jsonl", event)
            raise RuntimeError("Generation stopped on an infrastructure error; no answer was saved for " + identity) from error
        row = dict(report, response_id=identity, model=spec["repo_id"], revision=spec["revision"],
                   scenario_id=record["scenario_id"], condition=record["condition"], repeat=0,
                   seed=spec["generation"]["seed"], max_new_tokens=spec["generation"]["max_new_tokens"],
                   source="new_official_model_generation", generated_at=utc_now(),
                   generation_settings_sha256=settings_hash,
                   runtime_manifest_sha256=sha256_file(manifest_path), runtime_hardware=generator.hardware,
                   scheduler_job_id=generator.hardware["scheduler_job_id"])
        # Only hold the consolidated journal lock while appending; different models can run concurrently.
        with (results / ".answers.lock").open("a") as journal_lock:
            fcntl.flock(journal_lock, fcntl.LOCK_EX)
            existing = {item["response_id"] for item in read_jsonl(answers)} if answers.exists() else set()
            if identity in existing:
                raise RuntimeError("Concurrent duplicate model job detected; refusing a second saved answer")
            append_jsonl(answers, row)
        completed[identity] = row
        print(utc_now(), spec["key"], str(index + 1) + "/300", report["finish_reason"], flush=True)
    with (results / ".answers.lock").open("a") as journal_lock:
        fcntl.flock(journal_lock, fcntl.LOCK_EX)
        saved = [row for row in read_jsonl(answers) if row["response_id"] in planned]
    atomic_write_json(results / ("generation_" + spec["key"] + ".json"),
                      dict(settings, completed_at=utc_now(), reports=len(saved),
                           truncated=sum(row["truncated"] for row in saved),
                           technical_errors=sum(bool(row["technical_error"]) for row in saved),
                           repetitions=1, identical_greedy_repetitions_created=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run"))
    parser.add_argument("--model", required=True, help="Configured model key or exact official repository ID")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--templates-only", action="store_true", help="CPU preflight while official weights are downloading")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.templates_only and args.phase != "preflight":
        parser.error("--templates-only applies only to the CPU preflight phase")
    spec = model_spec(root, args.model)
    (root / "results").mkdir(exist_ok=True)
    try:
        if args.phase == "preflight":
            _, report = preflight(root, spec, require_weights=not args.templates_only)
            print({key: value for key, value in report.items() if key != "native_input_checks"})
        else:
            with (root / "results" / ("." + spec["key"] + ".generation.lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run(root, spec)
    except Exception as error:
        atomic_write_json(root / "results" / ("blocker_" + spec["key"] + ".json"),
                          {"checked_at": utc_now(), "model": spec["repo_id"], "revision": spec["revision"],
                           "phase": args.phase, "error_type": type(error).__name__, "reason": str(error),
                           "status": "blocked", "checkpoint_substitution": False})
        raise


if __name__ == "__main__":
    main()
