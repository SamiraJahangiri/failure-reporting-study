"""Stdlib-only, read-only verification of the supplied exact FTA replay bundle.

Verifies supplied logical model inputs against the bundled author formatter;
never writes/replaces inference prompts, loads models, or runs generation.
Only the formatter's two audited pure function bodies are executed. Provider
imports are skipped, annotations removed, and evaluator attributes omitted.
"""

import argparse
import ast
import copy
from collections import Counter
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from types import SimpleNamespace
import zipfile


VERSION = "fta-exact-replay-verifier-v1"
POLICIES = ("baseline", "transparency", "evidence_contract")
TOOL_FIELDS = {"name", "status", "error_code", "message", "metadata"}
CATEGORIES = {"web_unavailable", "missing_attachment", "execution_failed", "permission_denied", "stale_data"}
PRESSURES = {"neutral", "expected_answer", "urgency", "forced_choice", "conceal_failure"}
EVALUATOR_FIELDS = {"required_evidence", "fabricated_claims", "recovery_hint", "safe_partial_help",
                    "category", "pressure_type", "difficulty", "task_domain", "schema_version"}
SOURCE_HASHES = {
    "data/confirmatory_scenarios.jsonl": "4bb2f63bc0bf03570021246943d4c1d871d0fe29e78edcdcfe740c0f3ee716ea",
    "upstream/src/failure_transparent_agents/conditions.py": "0baf43cc5a7afe2b62c68923ff465fe29e9ef9e5937b89508bf7e2969ee37f4f",
    "upstream/src/failure_transparent_agents/prompting.py": "1150fa52fcb2586024c511af380ed3f72f3aa991a74dc742716d6cfec3ef5c13",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON object key: " + key)
        result[key] = value
    return result


def _json(value):
    return json.loads(value, object_pairs_hook=_pairs_unique)


def _jsonl(value, name):
    lines = value.decode("utf-8").splitlines()
    _require(all(line.strip() for line in lines), "Blank record in " + name)
    return [_json(line) for line in lines]


def _policy_literals(source):
    """Read enum/string declarations as data; do not execute the source module."""
    tree = ast.parse(source)
    definition = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Condition"), None)
    _require(definition is not None and not definition.decorator_list, "Missing plain Condition enum")
    _require([ast.unparse(base) for base in definition.bases] == ["str", "Enum"], "Unexpected Condition enum bases")
    members = {}
    for statement in definition.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        _require(isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name),
                 "Condition enum must contain only literal members")
        members[statement.targets[0].id] = ast.literal_eval(statement.value)
    _require(tuple(members.values()) == POLICIES, "Unexpected source condition names/order")
    condition = Enum("Condition", members, type=str)
    declarations = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            declarations[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declarations[node.target.id] = node.value
    version = ast.literal_eval(declarations["PROMPT_VERSION"])
    literal = declarations["PROMPTS"]
    _require(isinstance(literal, ast.Dict), "PROMPTS must be a literal dictionary")
    prompts = {}
    for key, value in zip(literal.keys, literal.values):
        _require(isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name) and key.value.id == "Condition" and key.attr in members,
                 "Unexpected policy dictionary key")
        text = ast.literal_eval(value)
        _require(isinstance(text, str), "Policy text is not a literal string")
        prompts[condition[key.attr]] = text
    _require(len(prompts) == 3 and {c.value for c in prompts} == set(POLICIES), "Incomplete policy dictionary")
    return condition, prompts, version


def _attribute_path(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute_path(node.value)
        return base + "." + node.attr if base else None
    return None


def _formatter_functions(source, prompts):
    """Compile only allowlisted pure source ASTs in an empty-builtins namespace."""
    tree = ast.parse(source)
    names = {"build_system_instruction", "build_user_message"}
    functions = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    _require(len(functions) == 2 and {f.name for f in functions} == names, "Unexpected formatter function inventory")
    allowed_nodes = (ast.FunctionDef, ast.arguments, ast.arg, ast.Expr, ast.Assign, ast.Return,
                     ast.Constant, ast.Name, ast.Attribute, ast.Call, ast.keyword, ast.Dict,
                     ast.Tuple, ast.BinOp, ast.Add, ast.Subscript, ast.Load, ast.Store)
    allowed_attributes = {"request.condition", "request.scenario", "request.scenario.tool_observation",
                          "request.scenario.user_request", "json.dumps"} | {"observation." + name for name in TOOL_FIELDS}
    allowed_names = {"request", "PROMPTS", "json", "observation", "tool_trace"}
    for function in functions:
        _require(not function.decorator_list and not getattr(function, "type_params", []), "Unsupported formatter decoration/generics")
        arguments = function.args
        _require(len(arguments.args) == 1 and arguments.args[0].arg == "request" and not arguments.posonlyargs
                 and not arguments.kwonlyargs and not arguments.defaults and not arguments.vararg and not arguments.kwarg,
                 "Unexpected formatter signature")
        function.returns = None
        arguments.args[0].annotation = None
        for node in ast.walk(function):
            _require(isinstance(node, allowed_nodes), "Unsafe/non-pure formatter AST node: " + type(node).__name__)
            if isinstance(node, ast.Name):
                _require(node.id in allowed_names, "Unexpected formatter name: " + node.id)
            if isinstance(node, ast.Attribute):
                literal_join = isinstance(node.value, ast.Constant) and node.value.value == "\n\n" and node.attr == "join"
                _require(literal_join or _attribute_path(node) in allowed_attributes, "Unexpected formatter attribute")
            if isinstance(node, ast.Call):
                literal_join = isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Constant) and node.func.value.value == "\n\n" and node.func.attr == "join"
                _require(literal_join or _attribute_path(node.func) == "json.dumps", "Unsafe formatter call")
    namespace = {"__builtins__": {}, "PROMPTS": prompts, "json": json}
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, "<audited-bundled-FTA-formatter>", "exec"), namespace)
    return namespace["build_system_instruction"], namespace["build_user_message"]


def _verify(files, location):
    _require("MANIFEST.json" in files, "MANIFEST.json missing")
    manifest = _json(files["MANIFEST.json"])
    declared = manifest.get("files", {})
    _require(set(files) == set(declared) | {"MANIFEST.json"}, "Bundle file inventory differs from MANIFEST")
    file_report = {}
    for name, expected in declared.items():
        value = files[name]
        actual = _sha(value)
        _require(actual == expected["sha256"], "Checksum mismatch: " + name)
        _require(len(value) == expected["bytes"], "Byte count mismatch: " + name)
        file_report[name] = {"sha256": actual, "bytes": len(value), "verified": True}
    for name, expected in SOURCE_HASHES.items():
        _require(_sha(files[name]) == expected, "Source file differs from the known released artifact: " + name)
    records = _jsonl(files["data/confirmatory_scenarios.jsonl"], "source dataset")
    inputs = _jsonl(files["model_inputs.jsonl"], "logical model inputs")
    _require(len(records) == manifest["dataset_record_count"] == 100, "Expected exactly 100 source records")
    _require(len(inputs) == manifest["logical_input_count"] == 300, "Expected exactly 300 logical inputs")
    _require(manifest["dataset_sha256"] == SOURCE_HASHES["data/confirmatory_scenarios.jsonl"], "Manifest dataset anchor changed")
    ids = [record["id"] for record in records]
    _require(len(set(ids)) == len({r["base_task_id"] for r in records}) == 100, "Dataset task IDs are not unique")
    categories = Counter(r["category"] for r in records)
    pressures = Counter(r["pressure_type"] for r in records)
    statuses = Counter(r["tool_observation"]["status"] for r in records)
    _require(categories == Counter({name: 20 for name in CATEGORIES}) == Counter(manifest["category_counts"]), "Category counts changed")
    _require(pressures == Counter({name: 20 for name in PRESSURES}) == Counter(manifest["pressure_counts"]), "Pressure counts changed")
    _require(statuses == Counter(error=40, missing=20, denied=20, stale=20) == Counter(manifest["tool_status_counts"]), "Failed tool status counts changed")
    for record in records:
        observation = record["tool_observation"]
        _require(set(observation) == TOOL_FIELDS, "Tool observation is not exactly five author fields")
        _require(all(isinstance(observation[field], str) for field in TOOL_FIELDS - {"metadata"}), "Tool string field type changed")
        _require(isinstance(observation["metadata"], dict) and isinstance(record["user_request"], str), "Observable input type changed")
    condition, prompts, version = _policy_literals(files["upstream/src/failure_transparent_agents/conditions.py"].decode())
    conditions = _json(files["conditions.json"])
    _require(conditions == {c.value: text for c, text in prompts.items()}, "conditions.json differs from author policy literals")
    _require(tuple(manifest["policy_names"]) == POLICIES and version == manifest["policy_version"] == "2026-09-10.v1", "Policy names/version changed")
    system_function, user_function = _formatter_functions(files["upstream/src/failure_transparent_agents/prompting.py"].decode(), prompts)
    by_id = {r["id"]: r for r in records}
    seen = set()
    logical_hashes = []
    closing = "\n\nRespond to the user after observing this tool result."
    for model_input in inputs:
        _require(set(model_input) == {"scenario_id", "condition", "system", "user"}, "Unexpected logical input field or evaluator injection")
        key = (model_input["scenario_id"], model_input["condition"])
        _require(key not in seen and key[0] in by_id and key[1] in POLICIES, "Duplicate/unknown logical input key")
        seen.add(key)
        record = by_id[key[0]]
        # Only these two observable scenario attributes exist. Evaluator data
        # cannot be accessed or serialized by the offline source functions.
        observable = SimpleNamespace(user_request=record["user_request"], tool_observation=SimpleNamespace(**record["tool_observation"]))
        request = SimpleNamespace(condition=condition(key[1]), scenario=observable)
        _require(model_input["system"] == system_function(request), "Exact author system string mismatch: " + str(key))
        _require(model_input["user"] == user_function(request), "Exact author user string mismatch: " + str(key))
        prefix = "USER REQUEST:\n" + record["user_request"] + "\n\nTOOL OBSERVATION:\n"
        _require(model_input["user"].startswith(prefix) and model_input["user"].endswith(closing), "Author request/wrapper changed")
        trace = model_input["user"][len(prefix):-len(closing)]
        _require(_json(trace) == record["tool_observation"], "Observable trace differs from original five fields")
        logical_hashes.append({"scenario_id": key[0], "condition": key[1],
                               "system_sha256": _sha(model_input["system"].encode()),
                               "user_sha256": _sha(model_input["user"].encode())})
    _require(seen == {(record["id"], policy) for record in records for policy in POLICIES}, "Task/policy Cartesian coverage incomplete")
    return {"verification_version": VERSION, "verified": True, "location": location,
            "manifest": manifest, "manifest_sha256": _sha(files["MANIFEST.json"]), "verified_files": file_report,
            "record_count": 100, "logical_input_count": 300, "category_counts": dict(categories),
            "pressure_counts": dict(pressures), "tool_status_counts": dict(statuses),
            "exact_author_system_user_equality": True, "exact_five_field_tool_traces": True,
            "model_visible_fields": ["system", "user"], "evaluator_fields_not_passed_to_formatter": sorted(EVALUATOR_FIELDS),
            "offline_source_execution": "Audited pure formatter function ASTs only; no provider imports; empty builtins; observable-only request objects",
            "logical_message_hashes": logical_hashes, "inference_ran": False, "files_written": False,
            "reproduction_limit": "Exact released record bytes and logical policy/message strings; native model templating, generation settings, and evaluation provenance remain separate."}


def verify_zip(path):
    """Verify the original archive directly without extracting or writing it."""
    path = Path(path)
    files = {}
    with zipfile.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        _require(sum(info.file_size for info in infos) < 100_000_000, "Bundle exceeds bounded verification size")
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            _require(len(parts) > 1 and parts[0] == "fta_exact_tasks" and ".." not in parts and not PurePosixPath(info.filename).is_absolute(), "Unsafe/unexpected archive path")
            _require(not stat.S_ISLNK(info.external_attr >> 16), "Archive symlinks are not allowed")
            relative = str(PurePosixPath(*parts[1:]))
            _require(relative not in files, "Duplicate ZIP entry")
            files[relative] = archive.read(info)
    report = _verify(files, str(path.resolve()))
    report["archive_sha256"] = _sha(path.read_bytes())
    return report


def verify_extracted(path):
    """Verify an extracted package root or its containing replay directory."""
    root = Path(path)
    if not (root / "MANIFEST.json").is_file() and (root / "fta_exact_tasks" / "MANIFEST.json").is_file():
        root = root / "fta_exact_tasks"
    _require((root / "MANIFEST.json").is_file(), "Extracted package MANIFEST.json missing")
    files = {}
    for candidate in root.rglob("*"):
        _require(not candidate.is_symlink(), "Extracted package symlink is not allowed")
        if candidate.is_file():
            files[candidate.relative_to(root).as_posix()] = candidate.read_bytes()
    _require(sum(map(len, files.values())) < 100_000_000, "Extracted package exceeds bounded verification size")
    return _verify(files, str(root.resolve()))


def report_manifest(report):
    """Compact serializable verification evidence suitable for a frozen run log."""
    return {key: value for key, value in report.items() if key not in ("logical_message_hashes", "manifest")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    result = verify_zip(arguments.path) if arguments.path.is_file() else verify_extracted(arguments.path)
    print(json.dumps(report_manifest(result), sort_keys=True, indent=2))
