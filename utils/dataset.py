"""Verify and load the saved FTA v0.3.0 inputs without changing their text."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen
import zipfile

from utils.io import atomic_write_bytes, read_json, read_jsonl, sha256_file


def load_inputs(root):
    return read_jsonl(Path(root) / "data/model_inputs.jsonl")


def load_scenarios(root):
    rows = read_jsonl(Path(root) / "data/confirmatory_scenarios.jsonl")
    scenarios = {row["id"]: row for row in rows}
    if len(rows) != len(scenarios):
        raise ValueError("Duplicate original scenario IDs")
    return scenarios


def load_policies(root):
    return read_json(Path(root) / "config/policies.json")


def recover_missing(root, source_bundle=None, fetch_missing=False):
    """Recover missing files only; never replace a present, possibly altered file."""
    root = Path(root)
    source = read_json(root / "config/source.json")
    recovered = []
    for relative, expected in source["files"].items():
        destination = root / relative
        if destination.exists():
            continue
        content = None
        if source_bundle is not None:
            bundle = Path(source_bundle).expanduser()
            bundled_path = expected["bundled_path"]
            if bundle.is_dir():
                candidate = bundle / bundled_path
                if candidate.is_file():
                    content = candidate.read_bytes()
            elif bundle.is_file():
                with zipfile.ZipFile(bundle) as archive:
                    candidates = [name for name in archive.namelist()
                                  if name == bundled_path or name.endswith("/" + bundled_path)]
                    if len(candidates) == 1:
                        content = archive.read(candidates[0])
        if content is None and fetch_missing and expected.get("url"):
            url = expected["url"]
            prefix = "https://raw.githubusercontent.com/junru-zhu/failure-transparent-agents/v0.3.0/"
            if not url.startswith(prefix):
                raise ValueError(f"Refusing an unpinned source URL: {url}")
            with urlopen(Request(url, headers={"User-Agent": "failure-reporting-study"}), timeout=60) as response:
                content = response.read()
        if content is None:
            hint = " Supply --source-bundle for the saved prebuilt inputs/policies." if not expected.get("url") else " Use --fetch-missing to fetch the pinned original."
            raise FileNotFoundError(f"Missing {destination}.{hint}")
        if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"Recovered source does not match its frozen hash: {relative}")
        atomic_write_bytes(destination, content, overwrite=False)
        recovered.append(relative)
    return recovered


def verify_inputs(root):
    """Check original bytes and author-formatted messages; do not rebuild inputs."""
    root = Path(root)
    source = read_json(root / "config/source.json")
    if source["upstream_release"] != "v0.3.0":
        raise ValueError("This phase is pinned to FTA v0.3.0")
    verified = {}
    for relative, expected in source["files"].items():
        path = root / relative
        digest = sha256_file(path)
        if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"Frozen source checksum mismatch: {relative}")
        verified[relative] = digest

    scenarios = load_scenarios(root)
    policies = load_policies(root)
    inputs = load_inputs(root)
    names = source["policies"]
    if len(scenarios) != 100 or len(inputs) != 300 or list(policies) != names:
        raise ValueError("Expected 100 original tasks and three policies (300 inputs)")

    # Read the pinned author policy literals, rather than trusting a rewritten policy.
    condition_tree = ast.parse((root / "data/upstream_conditions.py").read_text())
    prompt_node = next(node.value for node in condition_tree.body
                       if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                       and node.target.id == "PROMPTS")
    author_policies = {key.attr.lower(): ast.literal_eval(value)
                       for key, value in zip(prompt_node.keys, prompt_node.values)}
    if policies != author_policies:
        raise ValueError("Saved policy text differs from the original source")

    # Invoke only the original, checksum-verified formatter for equality checks.
    # Generation consumes the frozen strings below, never this check's output.
    formatter_tree = ast.parse((root / "data/upstream_prompting.py").read_text())
    formatter = next(node for node in formatter_tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "build_user_message")
    namespace = {"json": json, "GenerationRequest": object}
    exec(compile(ast.Module(body=[formatter], type_ignores=[]), "original FTA formatter", "exec"), namespace)
    expected_keys = [(scenario_id, policy) for scenario_id in scenarios for policy in names]
    actual_keys = []
    for row in inputs:
        if set(row) != {"scenario_id", "condition", "system", "user"}:
            raise ValueError("Frozen input contains extra model-visible/evaluation fields")
        scenario = scenarios[row["scenario_id"]]
        if row["system"] != policies[row["condition"]]:
            raise ValueError(f"Policy mismatch for {row['scenario_id']}/{row['condition']}")
        request = SimpleNamespace(scenario=SimpleNamespace(
            user_request=scenario["user_request"],
            tool_observation=SimpleNamespace(**scenario["tool_observation"])))
        if row["user"] != namespace["build_user_message"](request):
            raise ValueError(f"Author message mismatch for {row['scenario_id']}/{row['condition']}")
        actual_keys.append((row["scenario_id"], row["condition"]))
    if actual_keys != expected_keys:
        raise ValueError("Frozen input order, coverage, or uniqueness differs from the original bundle")
    for field, expected_name in (("category", "category_counts"), ("pressure_type", "pressure_counts")):
        if dict(Counter(row[field] for row in scenarios.values())) != source[expected_name]:
            raise ValueError(f"Original {field} coverage changed")
    if dict(Counter(row["tool_observation"]["status"] for row in scenarios.values())) != source["tool_status_counts"]:
        raise ValueError("Original failed-tool status coverage changed")
    return {"upstream_release": source["upstream_release"], "tasks": 100,
            "policies": names, "logical_inputs": 300, "file_sha256": verified,
            "author_messages_verified": True, "inputs_rebuilt": False,
            "model_execution_performed": False, "scoring_performed": False}


def load_and_verify_inputs(root):
    verify_inputs(root)
    return load_inputs(root)
