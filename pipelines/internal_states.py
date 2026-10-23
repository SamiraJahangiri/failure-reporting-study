"""Reconstruct internal states from saved exact FTA inputs and answers.

Prompt-only extraction has no answer tokens or generation. The separate small
 during-answer study teacher-forces saved tokens and generates one new tail per
 supported shared prefix. Heavy vectors stay outside the project; no interventions.
The PCA command only visualizes existing original-run prompt-state caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
from contextlib import nullcontext
import datetime
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset import load_and_verify_inputs
from utils.io import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file
from utils.models import cached_snapshot, digest, load_tokenizer, model_spec, native_input, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_VERSION = "fta-prompt-block-final-token-v1"
EXTRACTION_MODELS = ("qwen3_8b", "gemma2_9b", "ministral3_8b")


def extraction_packages():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "accelerate", "safetensors", "tokenizers", "numpy")}


def extraction_reference(root, spec):
    """Read saved interface metadata, never put answers or labels in model inputs."""
    if spec["key"] == "ministral3_8b":
        runtime_path = root / "results/runtime_ministral3_8b.json"
        preflight_path = root / "results/preflight_ministral3_8b.json"
        runtime, preflight = read_json(runtime_path), read_json(preflight_path)
        # Read only the original model's prompt metadata from the answer journal.
        import fcntl
        lock_path = root / "results/.answers.lock"
        with (lock_path.open("r") if lock_path.exists() else nullcontext()) as lock:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_SH)
            answers = [row for row in read_jsonl(root / "results/answers.jsonl")
                       if row["model"] == spec["repo_id"] and row["repeat"] == 0]
        fields = ("scenario_id", "condition", "revision", "original_messages", "effective_messages",
                  "native_prompt", "native_prompt_sha256", "native_input_token_ids_sha256", "input_length")
        native = sorted(({name: row[name] for name in fields} for row in answers),
                        key=lambda row: (row["scenario_id"], row["condition"]))
        checks = {(row["scenario_id"], row["condition"]): row
                  for row in preflight["native_input_checks"]}
        if runtime["revision"] != spec["revision"] or preflight["revision"] != spec["revision"]:
            raise ValueError("Ministral revision differs from the completed run")
        return {"runtime": runtime, "native": native, "preflight_checks": checks,
                "sources": {"runtime": str(runtime_path.relative_to(root)),
                            "preflight": str(preflight_path.relative_to(root)),
                            "native": "results/answers.jsonl (only selected prompt metadata fields)"},
                "source_hashes": {"runtime": sha256_file(runtime_path), "preflight": sha256_file(preflight_path),
                                  "selected_native_metadata": digest(native)}}
    directory = root / "results/preserved_exact"
    runtime_path = directory / ("runtime_Qwen3-8B.json" if spec["key"] == "qwen3_8b"
                                else "runtime_gemma-2-9b-it.json")
    runtime, manifest = read_json(runtime_path), read_json(directory / "run_manifest.json")
    native = [row for row in read_jsonl(directory / "native_inputs.jsonl") if row["model"] == spec["repo_id"]]
    lengths = {(row["scenario_id"], row["condition"]): row["input_length"]
               for row in read_jsonl(directory / "raw_responses.jsonl")
               if row["model"] == spec["repo_id"] and row["repeat"] == 0}
    if manifest["models"][spec["repo_id"]] != spec["revision"] or runtime["checkpoint_revision"] != spec["revision"]:
        raise ValueError("Cached model revision differs from the preserved exact replay")
    if runtime["precision"] != "torch.float16" or runtime["attention_implementation"] != "eager":
        raise ValueError("Unexpected original Qwen/Gemma numerical settings")
    if sha256_file(directory / "native_inputs.jsonl") != manifest["native_inputs_sha256"]:
        raise ValueError("Preserved native-input file differs from its original manifest")
    return {"runtime": runtime, "native": native, "lengths": lengths,
            "sources": {"runtime": str(runtime_path.relative_to(root)),
                        "native": "results/preserved_exact/native_inputs.jsonl",
                        "lengths": "results/preserved_exact/raw_responses.jsonl (only input_length)"},
            "source_hashes": {"runtime": sha256_file(runtime_path),
                              "manifest": sha256_file(directory / "run_manifest.json"),
                              "native": sha256_file(directory / "native_inputs.jsonl"),
                              "raw": sha256_file(directory / "raw_responses.jsonl")}}


def verify_extraction_inputs(root, spec, records, snapshot, reference, tokenizer):
    """Verify all native prompt bytes/tokens before the first model forward."""
    runtime = reference["runtime"]
    if type(tokenizer).__name__ != runtime["tokenizer_class"]:
        raise ValueError("Tokenizer class differs from the saved generation interface")
    metadata = snapshot["metadata_sha256"]
    if spec["key"] == "ministral3_8b":
        for name, expected in runtime["snapshot"]["metadata_sha256"].items():
            if metadata.get(name) != expected:
                raise ValueError("Ministral checkpoint metadata changed: " + name)
    else:
        if metadata["config.json"] != runtime["config_sha256"] or metadata["tokenizer_config.json"] != runtime["tokenizer_config_sha256"]:
            raise ValueError("Original model/tokenizer configuration checksum differs")
        # Historical template metadata hashed a JSON string; rendered prompts
        # were hashed as raw UTF-8. Preserve those two distinct conventions.
        template_hash = digest(json.dumps(tokenizer.chat_template, sort_keys=True,
                                          ensure_ascii=False, separators=(",", ":")))
        if tokenizer.chat_template != runtime["chat_template"] or template_hash != runtime["chat_template_sha256"]:
            raise ValueError("Original native chat template differs")
    saved = {(row["scenario_id"], row["condition"]): row for row in reference["native"]}
    keys = {(row["scenario_id"], row["condition"]) for row in records}
    if len(reference["native"]) != 300 or set(saved) != keys:
        raise ValueError("Saved native prompts do not cover all 300 distinct inputs")
    encoded, checks = [], []
    for index, record in enumerate(records):
        key = record["scenario_id"], record["condition"]
        row = saved[key]
        messages = [{"role": "system", "content": record["system"]}, {"role": "user", "content": record["user"]}]
        if row["revision"] != spec["revision"] or row["original_messages"] != messages:
            raise ValueError("Saved original message content/revision differs: " + str(key))
        if spec["key"] == "ministral3_8b":
            tokenized, native = native_input(tokenizer, spec, record)
            expected = reference["preflight_checks"][key]
            for name in ("native_prompt_sha256", "native_input_token_ids_sha256", "input_length"):
                if native[name] != row[name] or native[name] != expected[name]:
                    raise ValueError("Ministral saved answer/preflight prompt check differs: " + str(key))
            if native["native_prompt"] != row["native_prompt"] or native["effective_messages"] != row["effective_messages"]:
                raise ValueError("Ministral native rendering/messages differ: " + str(key))
            prompt, effective, mode = native["native_prompt"], messages, "native_system_user"
        else:
            effective = ([{"role": "user", "content": record["system"] + "\n\n" + record["user"]}]
                         if spec["key"] == "gemma2_9b" else messages)
            kwargs = runtime["chat_template_kwargs"]
            prompt = tokenizer.apply_chat_template(effective, tokenize=False, **kwargs)
            if row["effective_messages"] != effective or row["rendered_prompt"] != prompt or row["rendered_prompt_sha256"] != digest(prompt):
                raise ValueError("Preserved native rendering/adapter differs: " + str(key))
            # Match the original runner's encoder exactly; independently check native IDs.
            tokenized = tokenizer(prompt, add_special_tokens=False, return_attention_mask=True)
            direct_ids = tokenizer.apply_chat_template(effective, tokenize=True, return_dict=False, **kwargs)
            if tokenized["input_ids"] != direct_ids or len(direct_ids) != reference["lengths"][key]:
                raise ValueError("Original/native token encodings or saved lengths disagree: " + str(key))
            mode = row["template_mode"]
        ids, mask = tokenized["input_ids"], tokenized["attention_mask"]
        if ids and isinstance(ids[0], list):
            ids, mask = ids[0], mask[0]
        if not ids or len(ids) != len(mask) or any(value != 1 for value in mask):
            raise ValueError("Extraction requires one complete unpadded input")
        encoded.append({"input_ids": ids, "attention_mask": mask})
        checks.append({"index": index, "scenario_id": key[0], "condition": key[1],
                       "revision": spec["revision"], "original_messages": messages,
                       "effective_messages": effective, "native_template_mode": mode,
                       "rendered_prompt": prompt, "rendered_prompt_sha256": digest(prompt),
                       "input_token_ids_sha256": digest(ids), "input_length": len(ids),
                       "final_input_token_index": len(ids) - 1, "final_input_token_id": ids[-1],
                       "final_input_token_text": tokenizer.decode([ids[-1]], clean_up_tokenization_spaces=False),
                       "saved_rendering_verified": True, "saved_token_length_verified": True,
                       "original_encoder_matches_direct_native_ids": spec["key"] != "ministral3_8b",
                       "saved_native_token_hash_verified": spec["key"] == "ministral3_8b"})
    return encoded, checks


def extraction_gpu(torch):
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")) or not torch.cuda.is_available():
        raise RuntimeError("Extraction requires an allocated CUDA GPU; no login-node forward passes")
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory < 30 * 1024**3:
        raise RuntimeError("Extraction requires one GPU with at least 30 GiB memory")
    torch.cuda.set_device(0)
    readiness = torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    del readiness
    return {"scheduler_job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("PBS_JOBID")),
            "gpu": gpu.name, "gpu_memory_bytes": gpu.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "startup_free_memory_bytes": free_bytes, "startup_total_memory_bytes": total_bytes,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
            "cuda_runtime": torch.version.cuda, "hostname": socket.gethostname(),
            "python": platform.python_version(), "packages": extraction_packages(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def verify_complete_cache(paths, signature, records, checks):
    """Reuse only a complete, immutable, hash-verified extraction cache."""
    import numpy as np
    present = [path.exists() for path in paths.values()]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("Incomplete external cache is preserved; inspect it before a fresh extraction")
    manifest = read_json(paths["manifest"])
    if manifest.get("status") != "complete" or manifest["frozen_signature_sha256"] != signature:
        raise ValueError("Cached extraction signature differs; existing vectors will not be overwritten")
    for key in ("states", "native_checks", "log"):
        if sha256_file(paths[key]) != manifest["files"][key]["sha256"]:
            raise ValueError("External extraction cache checksum mismatch: " + key)
    with np.load(paths["states"], allow_pickle=False) as arrays:
        states = arrays["states"]
        if states.dtype != np.float32 or list(states.shape) != manifest["states_shape"] or states.shape[0] != 300 or not np.isfinite(states).all():
            raise ValueError("Cached state array shape/precision/finiteness differs")
        if arrays["scenario_ids"].tolist() != [r["scenario_id"] for r in records] or arrays["conditions"].tolist() != [r["condition"] for r in records]:
            raise ValueError("Cached state ordering differs from the frozen inputs")
        if arrays["input_lengths"].tolist() != [r["input_length"] for r in checks] or arrays["final_token_ids"].tolist() != [r["final_input_token_id"] for r in checks]:
            raise ValueError("Cached final-token metadata differs")
    if read_jsonl(paths["native_checks"]) != checks:
        raise ValueError("Cached native-input verification differs")
    return manifest


def extract(root, model_name, cache_dir):
    """One prompt-only forward per input, with decoder-block output hooks."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    root, cache_dir = Path(root).resolve(), Path(cache_dir).expanduser().resolve()
    if cache_dir == root or root in cache_dir.parents:
        raise ValueError("Heavy hidden-state vectors must be stored outside the project")
    spec = model_spec(root, model_name)
    if spec["key"] not in EXTRACTION_MODELS:
        raise ValueError("Extraction is restricted to saved Qwen, Gemma, and Ministral reports")
    records = load_and_verify_inputs(root)
    snapshot = verify_snapshot(spec, require_weights=True)
    reference = extraction_reference(root, spec)
    tokenizer = load_tokenizer(spec)
    hardware = extraction_gpu(torch)
    expected_packages = (reference["runtime"]["hardware"]["packages"] if spec["key"] == "ministral3_8b"
                         else reference["runtime"]["packages"])
    for name in ("torch", "transformers", "accelerate", "safetensors", "tokenizers"):
        if hardware["packages"][name] != expected_packages[name]:
            raise ValueError("Model runtime package differs from the saved run: " + name)
    if spec["key"] == "ministral3_8b":
        hardware["packages"]["mistral-common"] = importlib.metadata.version("mistral-common")
        if hardware["packages"]["mistral-common"] != expected_packages["mistral-common"]:
            raise ValueError("Original Mistral tokenizer backend version differs")
    encoded, checks = verify_extraction_inputs(root, spec, records, snapshot, reference, tokenizer)
    print(json.dumps({"status": "native_inputs_verified", "model": spec["key"], "inputs": len(checks),
                      "hardware": hardware}), flush=True)
    method = {"version": EXTRACTION_VERSION, "model": spec["repo_id"], "revision": spec["revision"],
              "logical_inputs_sha256": sha256_file(root / "data/model_inputs.jsonl"),
              "checkpoint_metadata_sha256": snapshot["metadata_sha256"],
              "reference_hashes": reference["source_hashes"], "verified_native_checks_sha256": digest(checks),
              "precision": "torch.bfloat16" if spec["key"] == "ministral3_8b" else "torch.float16",
              "attention_implementation": "eager", "batch_size": 1, "padding": False, "use_cache": False,
              "capture": "Every decoder block output at final full native input token, before final decoder norm",
              "output_hidden_states": False, "lm_head_called": False, "answer_tokens_in_input": False,
              "original_fp8_dequantized": spec["key"] == "ministral3_8b",
              "runtime_packages": hardware["packages"],
              "helper_sha256": {str(p.relative_to(root)): sha256_file(p) for p in (root / "utils").glob("*.py")},
              "extraction_implementation_sha256": digest("\n".join(inspect.getsource(function) for function in
                   (extraction_reference, verify_extraction_inputs, extraction_gpu, verify_complete_cache, extract)))}
    signature = digest(method)
    paths = {"states": cache_dir / (spec["key"] + ".npz"),
             "manifest": cache_dir / (spec["key"] + ".manifest.json"),
             "native_checks": cache_dir / (spec["key"] + ".native_checks.jsonl"),
             "log": cache_dir / (spec["key"] + ".extraction.jsonl")}
    reused = verify_complete_cache(paths, signature, records, checks)
    if reused:
        print(json.dumps({"status": "verified_complete_cache_reused", "model": spec["repo_id"],
                          "states": str(paths["states"]), "new_forward_passes": 0}), flush=True)
        return reused

    kwargs = {"local_files_only": True, "trust_remote_code": False,
              "torch_dtype": torch.bfloat16 if spec["key"] == "ministral3_8b" else torch.float16,
              "attn_implementation": "eager"}
    if spec["key"] == "ministral3_8b":
        from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        kwargs["device_map"] = "auto"
        model_class = Mistral3ForConditionalGeneration
    else:
        kwargs["low_cpu_mem_usage"] = True
        model_class = AutoModelForCausalLM
    print(json.dumps({"status": "loading_original_checkpoint", "model": spec["key"]}), flush=True)
    model = model_class.from_pretrained(str(cached_snapshot(spec)), **kwargs)
    if spec["key"] != "ministral3_8b":
        model.to("cuda:0")
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise ValueError("Original single-GPU loading requires all model parameters on CUDA; no offload")
    if type(model).__name__ != reference["runtime"]["model_class"]:
        raise ValueError("Loaded model class differs from the completed run")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    backbone_path = "model.language_model" if spec["key"] == "ministral3_8b" else "model"
    backbone = model.get_submodule(backbone_path)
    blocks, configuration = backbone.layers, backbone.config
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != configuration.num_hidden_layers:
        raise ValueError("Unexpected decoder block ordering/count")
    if spec["key"] != "ministral3_8b" and (len(blocks) != reference["runtime"]["decoder_blocks"] or configuration.hidden_size != reference["runtime"]["hidden_dimension"]):
        raise ValueError("Original decoder dimensions differ")
    states = np.empty((300, len(blocks), configuration.hidden_size), dtype=np.float32)
    active, handles, log = {}, [], []

    def hook_for(layer):
        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if tuple(value.shape) != (1, active["input_length"], configuration.hidden_size):
                raise ValueError("Unexpected decoder output shape")
            vector = value[0, -1, :].detach().to(device="cpu", dtype=torch.float32).numpy()
            if not np.isfinite(vector).all() or active["counts"][layer]:
                raise ValueError("Nonfinite or repeated block output")
            states[active["index"], layer] = vector
            active["counts"][layer] += 1
        return capture

    started = time.monotonic()
    try:
        handles = [block.register_forward_hook(hook_for(index)) for index, block in enumerate(blocks)]
        input_device = backbone.get_input_embeddings().weight.device
        for index, tokens in enumerate(encoded):
            active.update(index=index, input_length=len(tokens["input_ids"]), counts=[0] * len(blocks))
            tensors = {key: torch.tensor([value], dtype=torch.long, device=input_device) for key, value in tokens.items()}
            tick = time.monotonic()
            with torch.inference_mode():
                output = backbone(**tensors, use_cache=False, output_hidden_states=False, output_attentions=False)
            del output
            if active["counts"] != [1] * len(blocks):
                raise ValueError("Not every decoder block produced exactly one captured output")
            log.append({"index": index, "scenario_id": records[index]["scenario_id"],
                        "condition": records[index]["condition"], "input_length": active["input_length"],
                        "hook_counts": active["counts"], "all_finite": True,
                        "forward_seconds": round(time.monotonic() - tick, 6)})
            if (index + 1) % 25 == 0:
                print(json.dumps({"model": spec["key"], "prompt_forwards_completed": index + 1, "total": 300}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if not np.isfinite(states).all():
        raise ValueError("State array contains nonfinite values")
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, states=states,
                     scenario_ids=np.asarray([r["scenario_id"] for r in records]),
                     conditions=np.asarray([r["condition"] for r in records]),
                     input_lengths=np.asarray([r["input_length"] for r in checks], dtype=np.int32),
                     final_token_ids=np.asarray([r["final_input_token_id"] for r in checks], dtype=np.int64))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, paths["states"])
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    atomic_write_jsonl(paths["native_checks"], checks, overwrite=False)
    atomic_write_jsonl(paths["log"], log, overwrite=False)
    source_file = Path(inspect.getfile(type(backbone)))
    original_gpu = (reference["runtime"]["hardware"]["gpus"][0]["name"] if spec["key"] == "ministral3_8b"
                    else reference["runtime"]["gpu"])
    manifest = {"status": "complete", "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model_key": spec["key"], "model": spec["repo_id"], "revision": spec["revision"],
                "frozen_method": method, "frozen_signature_sha256": signature, "hardware": hardware,
                "original_generation_gpu": original_gpu, "hardware_differs_from_generation": hardware["gpu"] != original_gpu,
                "numerical_reproduction_limit": "Prompt tokens and weight precision match the saved run. Extraction uses batch 1 without padding, versus old Qwen/Gemma generation batch 4 with left padding. These are new prompt-state measurements, not byte-identical historical activation captures.",
                "states_shape": list(states.shape), "states_dtype": "float32", "all_finite": True,
                "block_definition": "B0 is the first transformer decoder block output; all vectors precede the backbone final norm. No final-normalized hidden-state replacement.",
                "layer_names": [backbone_path + ".layers." + str(i) for i in range(len(blocks))],
                "block_classes": [type(block).__name__ for block in blocks],
                "backbone_class": type(backbone).__name__, "backbone_module_path": backbone_path,
                "backbone_source_sha256": sha256_file(source_file), "backbone_source_filename": str(source_file),
                "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
                "capture_position": "Final token after the full native assistant prefix; no prefix trimming or answer tokens.",
                "native_input_count": 300, "native_checks": {"renderings_verified": 300, "token_sequences_verified": 300,
                    "qwen_gemma_original_token_ids_not_saved": spec["key"] != "ministral3_8b",
                    "token_verification_note": "Qwen/Gemma: saved rendering, pinned tokenizer, original no-special-token encoding, direct native-ID equality and saved input length. Ministral: direct native IDs match all saved answer and preflight hashes."},
                "reference_sources": reference["sources"], "snapshot": snapshot,
                "inference_mode": True, "eval_mode": not model.training, "frozen_parameters": True,
                "prompt_forward_passes": 300, "generation_calls": 0, "logits_computed": False,
                "semantic_labels_changed": False, "total_seconds": round(time.monotonic() - started, 6),
                "files": {key: {"filename": str(paths[key]), "bytes": paths[key].stat().st_size,
                                  "sha256": sha256_file(paths[key])} for key in ("states", "native_checks", "log")}}
    atomic_write_json(paths["manifest"], manifest, overwrite=False)
    print(json.dumps({"status": "complete", "model": spec["repo_id"], "states": str(paths["states"]),
                      "shape": list(states.shape), "prompt_forward_passes": 300, "generation_calls": 0}), flush=True)
    return manifest



from collections import defaultdict
from pathlib import Path
import json
import sys

import numpy as np

STATE_MODELS = ("qwen3_8b", "gemma2_9b", "ministral3_8b")
STATE_PAIRS = (("baseline", "evidence_contract"), ("baseline", "transparency"),
               ("transparency", "evidence_contract"))
STATE_METRICS = ("false_success", "fabricated_details", "either")
STATE_SEED, STATE_BOOTSTRAPS = 20261013, 2000
# Frozen from saved labels before states were inspected: eligible, disappeared,
# metric-clean, persistent, newly, matched cases, unique controls, missing cases.
STATE_EXPECTED_MAIN = {
    "qwen3_8b": ((97,25,72,0,0,2,6,23), (99,29,70,0,0,5,15,24), (97,27,70,0,0,3,9,24)),
    "gemma2_9b": ((94,8,86,0,0,5,7,3), (95,9,85,0,1,6,10,3), (94,9,84,0,1,6,10,3)),
    "ministral3_8b": ((100,31,69,0,0,16,14,15), (98,48,46,1,3,28,24,20), (98,48,46,1,3,28,24,20)),
}


def _state_valid(answer):
    return (bool(answer["response"].strip()) and not answer.get("truncated")
            and not answer.get("technical_error")
            and answer.get("finish_reason") == "eos")


def _state_known(label, name):
    return (name in label.get("rubric_labels", {}) and
            label.get("confidence", {}).get(name) in (2, 3) and
            name not in label.get("unresolved_labels", []))


def _state_joint_clear(label):
    return all(_state_known(label, k) and not label["rubric_labels"][k]
               for k in STATE_METRICS[:2])


def _state_unit(array):
    lengths = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, lengths, out=np.zeros_like(array), where=lengths > 1e-12)


def _state_pairwise(units, weights=None):
    """Mean pairwise cosine, excluding self-pairs and zero-length directions."""
    weights = np.ones(len(units)) if weights is None else np.asarray(weights)
    active = np.linalg.norm(units, axis=-1) > 1e-12
    mass = (weights[:, None] * active).sum(axis=0)
    diagonal = ((weights[:, None] ** 2) * active).sum(axis=0)
    summed = (units * weights[:, None, None]).sum(axis=0)
    denominator = mass ** 2 - diagonal
    return np.divide((summed ** 2).sum(axis=-1) - diagonal, denominator,
        out=np.full(denominator.shape, np.nan), where=denominator > 1e-12)


def _state_matching(cases, controls, strata, distances, deltas, rng):
    """Weight each control cell by its number of matched case tasks.

    Bootstrap task indices jointly across layers, independently within each
    matched category×pressure cell. This is a descriptive resampling interval.
    """
    case_cells, control_cells = defaultdict(list), defaultdict(list)
    for i in cases: case_cells[strata[i]].append(i)
    for i in controls: control_cells[strata[i]].append(i)
    matched_cells = {s: ids for s, ids in case_cells.items() if control_cells[s]}
    matched = [i for ids in matched_cells.values() for i in ids]
    unique_controls = sorted({i for s in matched_cells for i in control_cells[s]})
    missing = [i for s, ids in case_cells.items() if s not in matched_cells for i in ids]
    result = {"matched_cases": matched, "matched_controls": unique_controls,
              "missing_cases": missing, "matched_strata": len(matched_cells),
              "singleton_case_cells": sum(len(ids) == 1 for ids in matched_cells.values()),
              "singleton_control_cells": sum(len(control_cells[s]) == 1 for s in matched_cells)}
    if not matched:
        return result
    n = len(matched)
    control_distance = np.zeros(distances.shape[1], dtype=np.float64)
    control_delta = np.zeros_like(deltas[0], dtype=np.float64)
    control_unit_delta = np.zeros_like(control_delta)
    control_relative_l2 = np.zeros_like(control_distance)
    boot = np.zeros((STATE_BOOTSTRAPS, distances.shape[1]), dtype=np.float64)
    residuals, control_residuals = [], []
    for s, ids in matched_cells.items():
        cids, weight = control_cells[s], len(ids) / n
        centroid = deltas[cids].mean(axis=0)
        control_distance += weight * distances[cids].mean(axis=0)
        control_delta += weight * centroid
        control_unit_delta += weight * _state_unit(deltas[cids]).mean(axis=0)
        control_relative_l2 += weight * np.linalg.norm(deltas[cids], axis=-1).mean(axis=0)
        residuals.extend(deltas[i] - centroid for i in ids)
        # Same-cell clean controls require leave-one-out residual directions.
        if len(cids) > 1:
            total = deltas[cids].sum(axis=0)
            for i in cids:
                other_mean = (total - deltas[i]) / (len(cids) - 1)
                control_residuals.append((weight / len(cids), deltas[i] - other_mean))
        case_draw = rng.choice(ids, size=(STATE_BOOTSTRAPS, len(ids)), replace=True)
        control_draw = rng.choice(cids, size=(STATE_BOOTSTRAPS, len(cids)), replace=True)
        boot += weight * (distances[case_draw].mean(axis=1) - distances[control_draw].mean(axis=1))
    mean_distance = distances[matched].mean(axis=0)
    mean_delta = deltas[matched].mean(axis=0)
    case_relative_l2 = np.linalg.norm(deltas[matched], axis=-1).mean(axis=0)
    case_direction = _state_unit(deltas[matched]).mean(axis=0)
    denominator = np.linalg.norm(case_direction, axis=-1) * np.linalg.norm(control_unit_delta, axis=-1)
    alignment = np.divide((case_direction * control_unit_delta).sum(axis=-1), denominator,
                          out=np.full(denominator.shape, np.nan), where=denominator > 1e-12)
    loo_weight = sum(w for w, _ in control_residuals)
    loo = sum(w * _state_unit(d) for w, d in control_residuals) / loo_weight if loo_weight else None
    control_weights = [len(matched_cells[strata[i]]) / (n * len(control_cells[strata[i]])) for i in unique_controls]
    result.update(case_distance=mean_distance, control_distance=control_distance,
        contrast=mean_distance - control_distance, ci_low=np.quantile(boot, .025, axis=0),
        ci_high=np.quantile(boot, .975, axis=0),
        relative_delta_mean_contrast_norm=np.linalg.norm(mean_delta - control_delta, axis=-1),
        case_relative_l2=case_relative_l2, control_relative_l2=control_relative_l2,
        relative_l2_contrast=case_relative_l2-control_relative_l2,
        case_directional_coherence=np.linalg.norm(case_direction, axis=-1),
        control_directional_coherence=np.linalg.norm(control_unit_delta, axis=-1),
        case_control_direction_alignment=alignment,
        case_pairwise_direction_cosine=_state_pairwise(_state_unit(deltas[matched])),
        control_pairwise_direction_cosine=_state_pairwise(_state_unit(deltas[unique_controls]), control_weights),
        case_residual_pairwise_direction_cosine=_state_pairwise(_state_unit(np.asarray(residuals))),
        control_loo_residual_pairwise_direction_cosine=_state_pairwise(np.asarray([_state_unit(d) for _, d in control_residuals]), [w for w, _ in control_residuals]) if control_residuals else None,
        case_residual_directional_coherence=np.linalg.norm(_state_unit(np.asarray(residuals)).mean(axis=0), axis=-1),
        control_loo_residual_directional_coherence=np.linalg.norm(loo, axis=-1) if loo is not None else None,
        control_loo_weight_covered=loo_weight,
        case_nonzero_residual_n=(np.linalg.norm(np.asarray(residuals), axis=-1) > 1e-12).sum(axis=0),
        control_nonzero_loo_residual_n=(np.linalg.norm(np.asarray([d for _, d in control_residuals]), axis=-1) > 1e-12).sum(axis=0) if control_residuals else np.zeros(distances.shape[1]),
        zero_case_deltas=(np.linalg.norm(deltas[matched], axis=-1) <= 1e-12).sum(axis=0))
    return result


def _state_plot(profiles, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#3d6c93", "#cf8436")
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 7.1))
    for row, model in enumerate(STATE_MODELS):
        for col, pair in enumerate(STATE_PAIRS):
            ax = axes[row, col]
            sizes = []
            for metric, color in zip(STATE_METRICS[:2], colors):
                item = profiles[(model, pair, metric)]
                result = item["matching"]
                sizes.append(f"{'Success' if metric == 'false_success' else 'Details'}: {len(result['matched_cases'])} cases / {len(result['matched_controls'])} controls; {len(result['missing_cases'])} unmatched")
                if "contrast" not in result: continue
                x = np.arange(1, len(result["contrast"]) + 1)
                ax.plot(x, result["contrast"], color=color, label=metric.replace("_", " "))
                ax.fill_between(x, result["ci_low"], result["ci_high"], color=color, alpha=.12)
            ax.axhline(0, color="#8996a0", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.text(.02, .98, "\n".join(sizes), transform=ax.transAxes, va="top", fontsize=6,
                    bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none", "pad":1})
            if row == 0: ax.set_title(" → ".join(p.replace("evidence_contract", "contract") for p in pair), fontsize=9)
            if col == 0: ax.set_ylabel(model.replace("_", " ") + "\nmatched cosine-distance contrast", fontsize=8)
            if row == 2: ax.set_xlabel("Decoder block (before final norm)", fontsize=8)
            if not any("contrast" in profiles[(model, pair, m)]["matching"] for m in STATE_METRICS[:2]):
                ax.text(.5, .5, "No eligible matched cases", transform=ax.transAxes, ha="center", fontsize=8)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=c) for c in colors],
        ["false-success label disappeared", "fabricated-details label disappeared"],
        loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(.5, .955))
    fig.suptitle("Policy-associated movement at the final input token", fontsize=13, fontweight="bold", y=.995)
    fig.text(.5, .012, "Positive: larger movement than strict joint-clean controls in the same category × pressure cells.\n"
        "Shading: descriptive within-cell bootstrap; small cells and exploratory layers do not establish a mechanism or causal effect.",
        ha="center", fontsize=7, color="#53616c")
    fig.subplots_adjust(top=.88, bottom=.13, hspace=.42, wspace=.27, left=.095, right=.99)
    fig.savefig(path, dpi=175, facecolor="white"); plt.close(fig)


def _state_example(lines, model, scenario, pair, answers, labels, inputs, native, native_source):
    lines.append(f"\nTask {scenario} — {model}\n")
    for policy in pair:
        answer, answer_line = answers[(model, scenario, policy)]
        label, label_line = labels[answer["response_id"]]
        item, input_line = inputs[(scenario, policy)]
        lines.extend([f"\nPolicy: {policy}; response ID: {answer['response_id']}\n",
            f"Sources: results/answers.jsonl:{answer_line}; results/labels.jsonl:{label_line}; data/model_inputs.jsonl:{input_line}\n",
            "--- EXACT SYSTEM ---\n", item["system"], "\n--- END SYSTEM ---\n",
            "--- EXACT USER INCLUDING TOOL OBSERVATION ---\n", item["user"], "\n--- END USER ---\n",
            "--- COMPLETE SAVED ANSWER ---\n", answer["response"], "\n--- END ANSWER ---\n",
            "Saved rubric labels: " + json.dumps(label["rubric_labels"], sort_keys=True) + "\n",
            "Saved relevant confidence: " + json.dumps({k: label["confidence"][k] for k in STATE_METRICS[:2]}) + "\n"])
        check = native.get((scenario, policy))
        if check:
            lines.append(f"Native source: {native_source}:{check['index']+1}\n")
            lines.append("Native input provenance: " + json.dumps({k: v for k, v in check.items()
                if k not in ("original_messages", "effective_messages", "native_prompt", "rendered_prompt", "input_ids")}, ensure_ascii=False, sort_keys=True) + "\n")
            prompt = check.get("native_prompt", check.get("rendered_prompt", ""))
            messages = check.get("effective_messages", [])
            if prompt and messages and all(m["content"] in prompt for m in messages):
                prefix = prompt.split(messages[0]["content"], 1)[0]
                suffix = prompt.rsplit(messages[-1]["content"], 1)[1]
                lines.append("Actual native prefix: " + repr(prefix) + "\nActual native assistant-start suffix: " + repr(suffix) + "\n")
        else: lines.append("Native source details missing from supplied cache checks.\n")


def _state_validation_digest(value):
    import hashlib
    import json
    payload = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_validation_sha(content):
    import hashlib
    return hashlib.sha256(content).hexdigest()


def _state_validation_rows(content, source):
    import json
    rows = []
    for number, line in enumerate(content.splitlines(keepends=True), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL record: {source}:{number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object: {source}:{number}")
        rows.append((row, number, line))
    return rows


def state_read_frozen_sources(root, model_keys, policy_pairs, metrics, seed, bootstrap_resamples):
    """Read one locked journal snapshot and verify the frozen three-model data.

    Returns analysis-ready answer/label mappings, input rows, configuration,
    protocol, and JSON-safe provenance. Only EXTRA rows for configured Nemotron
    may be ignored. Selected raw lines retain original bytes/order, so appending
    Nemotron does not invalidate the original three-model snapshot hash.
    """
    import fcntl
    import json
    import re
    from pathlib import Path

    root = Path(root).resolve()
    protocol_path = root / "config/internal_states.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    config_bytes = (root / "config/models.json").read_bytes()
    configuration = json.loads(config_bytes)
    if protocol.get("phase") != "exploratory_prompt_only_internal_state_comparison":
        raise ValueError("Unexpected frozen internal-state phase")
    if tuple(protocol["models"]) != tuple(model_keys):
        raise ValueError("Analysis model list differs from frozen protocol")
    declared_pairs = tuple((p["source"], p["target"]) for p in protocol["policy_pairs"])
    if declared_pairs != tuple(tuple(p) for p in policy_pairs):
        raise ValueError("Analysis policy pairs differ from frozen protocol")
    if [p["priority"] for p in protocol["policy_pairs"]] != ["main", "secondary", "secondary"]:
        raise ValueError("Frozen policy-pair priorities differ")
    if tuple(protocol["labels"]) != tuple(metrics[:2]) or tuple(metrics) != ("false_success", "fabricated_details", "either"):
        raise ValueError("Analysis metrics differ from frozen labels/composite")
    declared_bootstraps = re.match(r"\s*(\d+)\b", protocol["uncertainty"])
    if protocol["seed"] != seed or not declared_bootstraps or int(declared_bootstraps[1]) != bootstrap_resamples:
        raise ValueError("Analysis resampling settings differ from frozen protocol")
    if "EOS" not in protocol["eligibility"]:
        raise ValueError("Frozen EOS eligibility rule is absent")

    selected_models = {configuration["models"][key]["repo_id"] for key in model_keys}
    if len(selected_models) != 3:
        raise ValueError("Expected three distinct selected model repositories")
    nemotron = configuration["models"]["nemotron3_super120b"]["repo_id"]
    if nemotron in selected_models:
        raise ValueError("Nemotron must remain outside this analysis")
    expected_hashes = protocol["preserved_source_snapshot_sha256"]
    static_bytes, verified_hashes = {}, {}
    for relative, expected in expected_hashes.items():
        if relative in ("results/answers.jsonl", "results/labels.jsonl"):
            continue
        content = (root / relative).read_bytes()
        actual = _state_validation_sha(content)
        if actual != expected:
            raise ValueError("Frozen source checksum mismatch: " + relative)
        static_bytes[relative], verified_hashes[relative] = content, actual

    lock_path = root / "results/.answers.lock"
    # Public/local clones omit this ignored lock. Create only the empty lock
    # when needed, using the exact path that the existing journal writer owns.
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        journal_bytes = {name: (root / name).read_bytes()
                         for name in ("results/answers.jsonl", "results/labels.jsonl")}
    selected, source_info = {}, {}
    for relative, content in journal_bytes.items():
        kept, excluded = [], 0
        for row, line_number, raw_line in _state_validation_rows(content, relative):
            if row.get("model") in selected_models:
                kept.append((row, line_number, raw_line))
            elif row.get("model") == nemotron:
                excluded += 1
            else:
                raise ValueError(f"Unexpected extra model in frozen journal: {relative}:{line_number}")
        selected_content = b"".join(line for _, _, line in kept)
        selected_hash = _state_validation_sha(selected_content)
        if selected_hash != expected_hashes[relative]:
            raise ValueError("Frozen selected-model raw-line checksum mismatch: " + relative)
        selected[relative] = kept
        verified_hashes[relative] = selected_hash
        source_info[relative] = {"full_snapshot_sha256": _state_validation_sha(content),
                                 "full_snapshot_bytes": len(content),
                                 "selected_raw_lines_sha256": selected_hash,
                                 "selected_records": len(kept),
                                 "ignored_nemotron_records": excluded,
                                 "allowed_extra_model": nemotron}

    raw_inputs = [row for row, _, _ in _state_validation_rows(static_bytes["data/model_inputs.jsonl"], "data/model_inputs.jsonl")]
    input_keys = [(r["scenario_id"], r["condition"]) for r in raw_inputs]
    if len(raw_inputs) != 300 or len(set(input_keys)) != 300:
        raise ValueError("Frozen input matrix must contain 300 unique task/policy keys")
    expected_keys = {(model, scenario, policy) for model in selected_models for scenario, policy in input_keys}
    answers, response_ids = {}, {}
    for row, number, _ in selected["results/answers.jsonl"]:
        key = (row["model"], row["scenario_id"], row["condition"])
        if row["repeat"] != 0 or key not in expected_keys or key in answers:
            raise ValueError("Invalid/duplicate selected answer key: " + repr(key))
        model_key = next(k for k in model_keys if configuration["models"][k]["repo_id"] == row["model"])
        if row["revision"] != configuration["models"][model_key]["revision"]:
            raise ValueError("Saved answer revision differs: " + repr(key))
        identity = row["response_id"]
        if identity in response_ids:
            raise ValueError("Duplicate selected answer response_id: " + identity)
        if not isinstance(row["response"], str):
            raise ValueError("Saved answer is not text: " + identity)
        answers[key], response_ids[identity] = (row, number), key
    if set(answers) != expected_keys:
        raise ValueError("Selected answers do not cover the frozen 900-row matrix")
    labels = {}
    for row, number, _ in selected["results/labels.jsonl"]:
        identity = row["response_id"]
        if identity in labels or identity not in response_ids:
            raise ValueError("Invalid/duplicate selected label response_id: " + identity)
        key = (row["model"], row["scenario_id"], row["condition"])
        if row["repeat"] != 0 or key != response_ids[identity]:
            raise ValueError("Label identity does not join its selected answer: " + identity)
        answer = answers[key][0]
        if row["response_sha256"] != _state_validation_sha(answer["response"].encode("utf-8")):
            raise ValueError("Saved label response hash does not match exact answer: " + identity)
        labels[identity] = (row, number)
    if set(labels) != set(response_ids):
        raise ValueError("Selected labels do not cover the frozen selected answers")
    return {"protocol": protocol, "configuration": configuration, "raw_inputs": raw_inputs,
            "answers": answers, "labels": labels,
            "provenance": {"protocol_sha256": _state_validation_sha(protocol_bytes),
                           "model_configuration_sha256": _state_validation_sha(config_bytes),
                           "verified_frozen_source_sha256": verified_hashes,
                           "journal_snapshots": source_info,
                           "unique_answer_label_joins": len(answers),
                           "response_hash_joins_verified": True,
                           "label_semantics_reviewed": False}}


def state_validate_cache(root, cache_dir, model_key, sources, extraction_functions, extraction_version):
    """Validate a complete cache and return its arrays without any model imports.

    Pass the SAME five extraction function objects used by extract() when it
    computed extraction_implementation_sha256. Returning the validated arrays
    avoids reopening an unchecked NPZ after this guard. Source/method checks
    reject an internally consistent cache from an older extractor as well.
    """
    import inspect
    import io
    import json
    from pathlib import Path
    import numpy as np

    root, cache_dir = Path(root).resolve(), Path(cache_dir).expanduser().resolve()
    if cache_dir == root or root in cache_dir.parents:
        raise ValueError("State vectors must remain outside the project")
    if model_key not in sources["protocol"]["models"]:
        raise ValueError("Cache model is outside frozen protocol")
    spec = sources["configuration"]["models"][model_key]
    paths = {"states": cache_dir / (model_key + ".npz"),
             "manifest": cache_dir / (model_key + ".manifest.json"),
             "native_checks": cache_dir / (model_key + ".native_checks.jsonl"),
             "log": cache_dir / (model_key + ".extraction.jsonl")}
    if not all(p.is_file() for p in paths.values()):
        raise ValueError("Analysis requires every completed cache file: " + model_key)
    manifest_bytes = paths["manifest"].read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "complete" or (manifest.get("model_key"), manifest.get("model"), manifest.get("revision")) != (model_key, spec["repo_id"], spec["revision"]):
        raise ValueError("Cache completion/model/revision differs: " + model_key)
    method = manifest["frozen_method"]
    if _state_validation_digest(method) != manifest["frozen_signature_sha256"]:
        raise ValueError("Cache frozen-method signature is invalid: " + model_key)
    required_method = {"version": extraction_version, "model": spec["repo_id"], "revision": spec["revision"],
                       "logical_inputs_sha256": sources["provenance"]["verified_frozen_source_sha256"]["data/model_inputs.jsonl"],
                       "precision": "torch.bfloat16" if model_key == "ministral3_8b" else "torch.float16",
                       "attention_implementation": "eager", "batch_size": 1, "padding": False,
                       "use_cache": False, "output_hidden_states": False, "lm_head_called": False,
                       "answer_tokens_in_input": False, "original_fp8_dequantized": model_key == "ministral3_8b",
                       "capture": "Every decoder block output at final full native input token, before final decoder norm"}
    for name, expected in required_method.items():
        if method.get(name) != expected:
            raise ValueError(f"Cache method differs for {model_key}: {name}")
    expected_names = ("extraction_reference", "verify_extraction_inputs", "extraction_gpu", "verify_complete_cache", "extract")
    if tuple(f.__name__ for f in extraction_functions) != expected_names:
        raise ValueError("Expected the original ordered five extraction functions")
    current_extraction_hash = _state_validation_digest("\n".join(inspect.getsource(f) for f in extraction_functions))
    if method["extraction_implementation_sha256"] != current_extraction_hash:
        raise ValueError("Cache was produced by a different extraction implementation: " + model_key)
    helper_hashes = {str(p.relative_to(root)): _state_validation_sha(p.read_bytes()) for p in (root / "utils").glob("*.py")}
    if method["helper_sha256"] != helper_hashes:
        raise ValueError("Cache helper sources differ from the frozen current helpers")
    if manifest["model_class"] != spec["architecture"]:
        raise ValueError("Cache architecture differs from pinned model")
    snapshot = manifest["snapshot"]
    if snapshot["repo_id"] != spec["repo_id"] or snapshot["revision"] != spec["revision"] or snapshot["metadata_sha256"] != method["checkpoint_metadata_sha256"]:
        raise ValueError("Cache checkpoint snapshot provenance is inconsistent")
    backbone_path = "model.language_model" if model_key == "ministral3_8b" else "model"
    expected_backbone = {"qwen3_8b": "Qwen3Model", "gemma2_9b": "Gemma2Model", "ministral3_8b": "Ministral3Model"}[model_key]
    if manifest["backbone_module_path"] != backbone_path or manifest["backbone_class"] != expected_backbone:
        raise ValueError("Cache decoder backbone differs")

    file_bytes, file_info = {}, {}
    for name in ("states", "native_checks", "log"):
        content = paths[name].read_bytes()
        declared = manifest["files"][name]
        actual = _state_validation_sha(content)
        if actual != declared["sha256"] or len(content) != declared["bytes"] or Path(declared["filename"]).name != paths[name].name:
            raise ValueError("Declared cache file hash/size/name differs: " + name)
        file_bytes[name] = content
        file_info[name] = {"path": str(paths[name]), "sha256": actual, "bytes": len(content)}
    checks = [row for row, _, _ in _state_validation_rows(file_bytes["native_checks"], str(paths["native_checks"]))]
    log = [row for row, _, _ in _state_validation_rows(file_bytes["log"], str(paths["log"]))]
    if _state_validation_digest(checks) != method["verified_native_checks_sha256"]:
        raise ValueError("Native verification checks differ from frozen extraction method")
    records = sources["raw_inputs"]
    record_keys = [(r["scenario_id"], r["condition"]) for r in records]
    if len(checks) != 300 or [(r["scenario_id"], r["condition"]) for r in checks] != record_keys:
        raise ValueError("Native-check order differs from frozen input matrix")
    selected_answers = [a for a, _ in sources["answers"].values() if a["model"] == spec["repo_id"]]
    answer_map = {(r["scenario_id"], r["condition"]): r for r in selected_answers}

    directory = root / "results/preserved_exact"
    if model_key == "ministral3_8b":
        runtime_path, preflight_path = root / "results/runtime_ministral3_8b.json", root / "results/preflight_ministral3_8b.json"
        runtime = json.loads(runtime_path.read_bytes())
        preflight_bytes = preflight_path.read_bytes()
        preflight = json.loads(preflight_bytes)
        fields = ("scenario_id", "condition", "revision", "original_messages", "effective_messages",
                  "native_prompt", "native_prompt_sha256", "native_input_token_ids_sha256", "input_length")
        selected_native = sorted(({name: r[name] for name in fields} for r in selected_answers), key=lambda r: (r["scenario_id"], r["condition"]))
        reference_hashes = {"runtime": _state_validation_sha(runtime_path.read_bytes()),
                            "preflight": _state_validation_sha(preflight_bytes),
                            "selected_native_metadata": _state_validation_digest(selected_native)}
        saved_native = {(r["scenario_id"], r["condition"]): r for r in selected_native}
        preflight_checks = {(r["scenario_id"], r["condition"]): r for r in preflight["native_input_checks"]}
        if len(preflight["native_input_checks"]) != 300 or set(preflight_checks) != set(record_keys) or runtime["revision"] != spec["revision"] or preflight["revision"] != spec["revision"]:
            raise ValueError("Original Ministral references differ")
        expected_packages = runtime["hardware"]["packages"]
        if any(method["checkpoint_metadata_sha256"].get(name) != expected for name, expected in runtime["snapshot"]["metadata_sha256"].items()):
            raise ValueError("Ministral checkpoint metadata differs from original runtime")
    else:
        runtime_path = directory / ("runtime_Qwen3-8B.json" if model_key == "qwen3_8b" else "runtime_gemma-2-9b-it.json")
        runtime = json.loads(runtime_path.read_bytes())
        original_manifest_bytes = (directory / "run_manifest.json").read_bytes()
        original_manifest = json.loads(original_manifest_bytes)
        native_bytes = (directory / "native_inputs.jsonl").read_bytes()
        raw_bytes = (directory / "raw_responses.jsonl").read_bytes()
        reference_hashes = {"runtime": _state_validation_sha(runtime_path.read_bytes()),
                            "manifest": _state_validation_sha(original_manifest_bytes),
                            "native": _state_validation_sha(native_bytes), "raw": _state_validation_sha(raw_bytes)}
        native = [r for r, _, _ in _state_validation_rows(native_bytes, "preserved native inputs") if r["model"] == spec["repo_id"]]
        saved_native = {(r["scenario_id"], r["condition"]): r for r in native}
        if len(native) != 300 or set(saved_native) != set(record_keys) or reference_hashes["native"] != original_manifest["native_inputs_sha256"]:
            raise ValueError("Preserved native input coverage/hash differs")
        if original_manifest["models"][spec["repo_id"]] != spec["revision"] or runtime["checkpoint_revision"] != spec["revision"]:
            raise ValueError("Original Qwen/Gemma revision differs")
        expected_packages = runtime["packages"]
        if method["checkpoint_metadata_sha256"]["config.json"] != runtime["config_sha256"] or method["checkpoint_metadata_sha256"]["tokenizer_config.json"] != runtime["tokenizer_config_sha256"]:
            raise ValueError("Original model/tokenizer metadata differs")
    if method["reference_hashes"] != reference_hashes:
        raise ValueError("Cache source references differ from current frozen originals")
    if manifest["tokenizer_class"] != runtime["tokenizer_class"] or method["runtime_packages"] != manifest["hardware"]["packages"]:
        raise ValueError("Cache tokenizer/runtime package provenance is inconsistent")
    package_names = ("torch", "transformers", "accelerate", "safetensors", "tokenizers") + (("mistral-common",) if model_key == "ministral3_8b" else ())
    if any(method["runtime_packages"].get(name) != expected_packages.get(name) for name in package_names):
        raise ValueError("Cache numerical/tokenizer package versions differ from saved generation")
    gpu_family = "H100" if model_key == "ministral3_8b" else "V100"
    if gpu_family not in manifest["hardware"]["gpu"]:
        raise ValueError("Cache hardware family differs from frozen precision protocol")

    for index, (record, check) in enumerate(zip(records, checks)):
        key = record_keys[index]
        saved = saved_native[key]
        answer = answer_map[key]
        messages = [{"role": "system", "content": record["system"]}, {"role": "user", "content": record["user"]}]
        prompt_name, prompt_hash_name = (("native_prompt", "native_prompt_sha256") if model_key == "ministral3_8b" else ("rendered_prompt", "rendered_prompt_sha256"))
        if check["index"] != index or check["revision"] != spec["revision"] or check["original_messages"] != messages or saved["original_messages"] != messages:
            raise ValueError("Cache original-message identity differs: " + repr(key))
        if check["effective_messages"] != saved["effective_messages"] or check["rendered_prompt"] != saved[prompt_name] or check["rendered_prompt_sha256"] != saved[prompt_hash_name] or _state_validation_digest(check["rendered_prompt"]) != saved[prompt_hash_name]:
            raise ValueError("Cache native prompt/adapter differs: " + repr(key))
        length = check["input_length"]
        if type(length) is not int or length <= 0 or length != answer["input_length"] or check["final_input_token_index"] != length - 1 or type(check["final_input_token_id"]) is not int or check["final_input_token_id"] < 0:
            raise ValueError("Cache final-token position/length differs: " + repr(key))
        if not check["saved_rendering_verified"] or not check["saved_token_length_verified"]:
            raise ValueError("Cache native verification is incomplete")
        if model_key == "ministral3_8b":
            if check["input_token_ids_sha256"] != saved["native_input_token_ids_sha256"] or check["input_token_ids_sha256"] != preflight_checks[key]["native_input_token_ids_sha256"] or not check["saved_native_token_hash_verified"]:
                raise ValueError("Ministral saved native token hash differs")
        elif check["native_template_mode"] != saved["template_mode"] or not check["original_encoder_matches_direct_native_ids"]:
            raise ValueError("Original/native token encoding provenance differs")

    with np.load(io.BytesIO(file_bytes["states"]), allow_pickle=False) as cache:
        states = cache["states"]
        ids, conditions = cache["scenario_ids"], cache["conditions"]
        lengths, final_ids = cache["input_lengths"], cache["final_token_ids"]
        if states.dtype != np.float32 or states.ndim != 3 or states.shape[0] != 300 or list(states.shape) != manifest["states_shape"] or manifest["states_dtype"] != "float32" or not np.isfinite(states).all():
            raise ValueError("Cache state shape/precision/finiteness differs")
        if np.any(np.linalg.norm(states, axis=-1) <= 1e-12):
            raise ValueError("Zero block-state vector cannot support cosine analysis")
        if ids.shape != (300,) or conditions.shape != (300,) or list(zip(ids.tolist(), conditions.tolist())) != record_keys:
            raise ValueError("NPZ identity ordering differs from frozen native inputs")
        if lengths.shape != (300,) or final_ids.shape != (300,) or lengths.tolist() != [r["input_length"] for r in checks] or final_ids.tolist() != [r["final_input_token_id"] for r in checks]:
            raise ValueError("NPZ final-token metadata differs from native checks")
    layers, width = states.shape[1:]
    if len(manifest["layer_names"]) != layers or manifest["layer_names"] != [backbone_path + ".layers." + str(i) for i in range(layers)] or len(manifest["block_classes"]) != layers:
        raise ValueError("Cache block ordering/classes differ")
    if model_key != "ministral3_8b" and (layers != runtime["decoder_blocks"] or width != runtime["hidden_dimension"]):
        raise ValueError("Cache dimensions differ from preserved model")
    if len(log) != 300 or any(r["index"] != i or (r["scenario_id"], r["condition"]) != record_keys[i] or r["input_length"] != checks[i]["input_length"] or r["hook_counts"] != [1] * layers or r["all_finite"] is not True for i, r in enumerate(log)):
        raise ValueError("Extraction log does not verify every block once for each input")
    required_manifest = {"native_input_count": 300, "prompt_forward_passes": 300, "generation_calls": 0,
                         "logits_computed": False, "semantic_labels_changed": False,
                         "inference_mode": True, "eval_mode": True, "frozen_parameters": True, "all_finite": True}
    if any(manifest.get(k) != v for k, v in required_manifest.items()):
        raise ValueError("Cache extraction-completion claims differ from frozen protocol")
    backbone_source = Path(manifest["backbone_source_filename"])
    source_rechecked = backbone_source.is_file()
    if source_rechecked and _state_validation_sha(backbone_source.read_bytes()) != manifest["backbone_source_sha256"]:
        raise ValueError("Recorded backbone source changed after extraction")
    return {"states": states, "scenario_ids": ids, "conditions": conditions,
            "input_lengths": lengths, "final_token_ids": final_ids,
            "native_checks": checks, "manifest": manifest,
            "provenance": {"manifest_path": str(paths["manifest"]),
                           "manifest_sha256": _state_validation_sha(manifest_bytes),
                           "frozen_signature_sha256": manifest["frozen_signature_sha256"],
                           "extraction_implementation_sha256": current_extraction_hash,
                           "validated_files": file_info, "helper_sha256": helper_hashes,
                           "reference_hashes": reference_hashes,
                           "backbone_source_rechecked_here": source_rechecked,
                           "protocol_sha256": sources["provenance"]["protocol_sha256"],
                           "model_forward_passes_in_validation": 0, "numpy_analysis_version": np.__version__}}


def state_analysis_implementation_hash(functions, constants):
    """Fingerprint only analysis/validation code and its executable constants."""
    import inspect
    return _state_validation_digest({"functions": [{"name": f.__name__, "source": inspect.getsource(f)} for f in functions],
                                     "constants": constants})


def _state_findings(metadata, table):
    lines = ["Completed: 900 prompt-only forwards across Qwen, Gemma and Ministral; zero new answers and zero new rubric judgments. Nemotron's submitted job and frozen dependencies were untouched.\n\n",
        "Changing policy moves states even for reports already free of both saved unsupported-claim flags. Comparing with those controls gives different patterns across models and labels; it does not identify a mechanism we can repair.\n\n",
        "Main comparison (baseline to evidence contract). Candidate blocks rank absolute matched cosine-distance contrast; positive means more angular movement than controls, negative means less. Block numbering starts at 1 and excludes final normalization.\n"]
    for model in metadata:
        lines.append("\n" + model["model"] + "\n")
        for metric, title in zip(STATE_METRICS[:2], ("Unsupported success", "Invented details")):
            item = next(p for p in model["pairs"] if p["source_policy"] == "baseline" and p["target_policy"] == "evidence_contract" and p["metric"] == metric)
            rows = [r for r in table if r["model"] == model["model"] and r["metric"] == metric and r["source_policy"] == "baseline" and r["target_policy"] == "evidence_contract"]
            if not item.get("exploratory_peak_block"):
                lines.append(title + ": no matched candidate.\n"); continue
            peak = next(r for r in rows if r["decoder_block"] == item["exploratory_peak_block"])
            lines.append(f"{title}: candidates {item['exploratory_largest_absolute_contrast_blocks']}; peak cosine contrast {peak['contrast']:+.5f}, relative-L2 contrast {peak['relative_l2_contrast']:+.5f}. "
                f"Matched {peak['matched_case_n']} of {peak['disappeared_n']} label-removed cases to {peak['matched_unique_control_n']} unique controls; {peak['unmatched_case_n']} cases lacked controls. "
                f"Case/control movement-direction alignment {peak['case_control_direction_alignment']:.3f}; control-centered case residual pairwise cosine {peak['case_residual_pairwise_direction_cosine']:.3f}.\n")
    lines.extend(["\nQwen's success comparison has only two matched cases. Gemma and Ministral have less angular and relative-L2 movement than controls at their primary peaks. Shared movement directions remain strongly aligned, and residual directions do not show a strong common direction at these peaks. These observations are compatible with a broad policy/format effect; they do not isolate false-reporting circuitry.\n",
        "Secondary comparisons have different cohorts: baseline-to-transparency peak contrasts are positive for both flags in each model; transparency-to-contract has negative late-block troughs, with a positive final-block Qwen details peak. The full trajectories, group sizes, exclusions and token-length confounds follow below and in the CSV.\n",
        "No result establishes that a model knowingly hid failure. Tiny matched cells and descriptive bootstrap intervals do not support significance claims or confirmation of the selected layers.\n\n--- METHODS, COUNTS AND COMPLETE SAVED EXAMPLES ---\n\n"])
    return "".join(lines)


def analyze(root, cache_dir, output_prefix):
    """Write one CSV, PNG, text report, and compact JSON; no model calls."""
    import csv
    root, cache_dir, output_prefix = Path(root).resolve(), Path(cache_dir).resolve(), Path(output_prefix)
    sys.path.insert(0, str(root))
    from utils.dataset import load_and_verify_inputs, load_scenarios
    from utils.io import read_json, read_jsonl, sha256_file, atomic_write_json
    sources = state_read_frozen_sources(root, STATE_MODELS, STATE_PAIRS, STATE_METRICS, STATE_SEED, STATE_BOOTSTRAPS)
    scenarios, configuration = load_scenarios(root), sources["configuration"]
    raw_inputs = sources["raw_inputs"]
    if raw_inputs != load_and_verify_inputs(root):
        raise ValueError("Frozen input snapshot differs from verified author formatting")
    inputs = {(r["scenario_id"], r["condition"]): (r, i + 1) for i, r in enumerate(raw_inputs)}
    answers, labels = sources["answers"], sources["labels"]
    profiles, table, metadata = {}, [], []
    text = ["Saved-policy internal-state comparison\n\n",
        "This compares cached decoder-block states before any report token was generated. It uses the original saved answers and labels without rescoring.\n",
        "All models replay the 300 author inputs unchanged. Qwen preserves the historical native system/user template with thinking disabled; Gemma preserves the historical adapter joining system + two newlines + user into one native user message; Ministral uses its original native system/user interface. No instruction or assistant-start suffix is shortened.\n",
        "Qwen/Gemma use their pinned FP16/eager checkpoints on V100-32GB; Ministral uses its pinned official FP8 checkpoint dequantized to BF16/eager on H100. Extraction uses batch 1, no padding and no cache. Old Qwen/Gemma generation used batch 4 and left padding, so these are reconstructed prompt states rather than byte-identical historical activations. Core Torch/Transformers/tokenizer versions match; host NumPy versions may differ and are recorded.\n",
        "Qwen/Gemma historical native token-ID arrays were not saved: their exact rendered bytes, original encoder, independent direct native-ID equality and saved real-token counts were verified. Ministral's native IDs also match all saved answer/preflight token hashes.\n",
        "A label-disappeared case loses the named flag; this does not imply the other unsupported-claim flag disappeared. Controls require both false_success and fabricated_details to be known false under both policies.\n",
        "Controls are different tasks matched only on original failure category × pressure type. Cells contain at most four tasks, and absent controls are never borrowed from other cells.\n",
        "Angular movement uses cosine distance. Core movement vectors are v=(h_target-h_source)/||h_source||, retaining radial change; relative L2 and angular contrasts are reported separately. Matched control centroids of v are subtracted before residual directions are normalized. Clean residual coherence uses leave-one-out centroids.\n",
        "Mean pairwise direction cosine excludes self-pairs and is unavailable with fewer than two nonzero directions. Resultant-length coherence is supplementary and depends on group size; it is not directly comparable across unequal sample sizes. No directions are compared across models.\n",
        "Clean leave-one-out residuals are constrained by centering: two controls give antipodal directions by construction. Their coherence cannot be treated as a neutral reference for independently centered case residuals; the directional columns are exploratory diagnostics. Singleton control cells contribute no clean residual directions.\n",
        "Bootstrap task indices are shared across layers within each exact cell, holding cell weights fixed. Intervals describe these tiny saved groups, not population effects or label uncertainty; no p-values or cross-policy-pair inference are claimed. Layers are explored, not confirmed.\n"]
    for model_key in STATE_MODELS:
        spec, path = configuration["models"][model_key], cache_dir / (model_key + ".npz")
        validated = state_validate_cache(root, cache_dir, model_key, sources,
            (extraction_reference, verify_extraction_inputs, extraction_gpu, verify_complete_cache, extract), EXTRACTION_VERSION)
        states, ids, conditions = validated["states"], validated["scenario_ids"].astype(str), validated["conditions"].astype(str)
        input_lengths = validated["input_lengths"]
        positions = {(s, p): i for i, (s, p) in enumerate(zip(ids, conditions))}
        unit_states = _state_unit(states)
        task_ids = sorted(scenarios)
        strata = [(scenarios[s]["category"], scenarios[s]["pressure_type"]) for s in task_ids]
        native_path = cache_dir / (model_key + ".native_checks.jsonl")
        native = {(r["scenario_id"], r["condition"]): r for r in validated["native_checks"]}
        model_meta = {"model": spec["repo_id"], "cache": str(path),
                      "cache_sha256": validated["provenance"]["validated_files"]["states"]["sha256"],
                      "shape": list(states.shape), "pairs": [], "validated_cache": validated["provenance"],
                      "extraction_manifest": validated["manifest"]}
        text.append(f"\n{spec['repo_id']} ({states.shape[1]} decoder blocks)\n")
        for pair in STATE_PAIRS:
            src, tgt = pair
            src_positions, tgt_positions = [[positions[(s, p)] for s in task_ids] for p in pair]
            source_states, target_states = states[src_positions], states[tgt_positions]
            distances = 1 - np.clip((unit_states[src_positions] * unit_states[tgt_positions]).sum(axis=-1), -1, 1)
            deltas = (target_states-source_states) / np.linalg.norm(source_states, axis=-1, keepdims=True)
            source_lengths, target_lengths = input_lengths[src_positions], input_lengths[tgt_positions]
            for metric in STATE_METRICS:
                groups = {k: [] for k in ("disappeared", "metric_clean", "persistent", "newly")}
                controls, exclusions, target_clear, other_positive, other_unknown = [], [], [], [], []
                relevant = STATE_METRICS[:2] if metric == "either" else (metric,)
                for i, scenario in enumerate(task_ids):
                    endpoint = [answers.get((spec["repo_id"], scenario, p)) for p in pair]
                    judged = [labels.get(a[0]["response_id"]) if a else None for a in endpoint]
                    if not all(a and _state_valid(a[0]) for a in endpoint):
                        exclusions.append({"task_id": scenario, "reason": "missing_or_invalid_answer"}); continue
                    if not all(l and all(_state_known(l[0], k) for k in relevant) for l in judged):
                        exclusions.append({"task_id": scenario, "reason": "relevant_label_unknown"}); continue
                    left, right = [any(l[0]["rubric_labels"][k] for k in relevant) for l in judged]
                    group = {(True, False): "disappeared", (False, False): "metric_clean", (True, True): "persistent", (False, True): "newly"}[(left, right)]
                    groups[group].append(i)
                    if all(_state_joint_clear(l[0]) for l in judged): controls.append(i)
                    if group == "disappeared":
                        if _state_joint_clear(judged[1][0]): target_clear.append(i)
                        else:
                            other = "fabricated_details" if metric == "false_success" else "false_success"
                            (other_positive if _state_known(judged[1][0], other) else other_unknown).append(i)
                rng = np.random.default_rng(STATE_SEED + STATE_MODELS.index(model_key) * 100 + STATE_PAIRS.index(pair) * 10 + STATE_METRICS.index(metric))
                matched = _state_matching(groups["disappeared"], controls, strata, distances, deltas, rng)
                length_stats = {"source_length_case_mean": None, "source_length_control_mean": None,
                    "source_length_matched_contrast": None, "policy_length_increment_case_mean": None,
                    "policy_length_increment_control_mean": None, "policy_length_increment_matched_contrast": None}
                if matched["matched_cases"]:
                    mc, nc = matched["matched_cases"], len(matched["matched_cases"])
                    weights = [sum(strata[j] == strata[i] for j in mc) / (nc * sum(strata[j] == strata[i] for j in matched["matched_controls"])) for i in matched["matched_controls"]]
                    cl = matched["matched_controls"]
                    increment = target_lengths-source_lengths
                    length_stats.update(source_length_case_mean=float(source_lengths[mc].mean()),
                        source_length_control_mean=float(np.dot(weights, source_lengths[cl])),
                        policy_length_increment_case_mean=float(increment[mc].mean()),
                        policy_length_increment_control_mean=float(np.dot(weights, increment[cl])))
                    length_stats["source_length_matched_contrast"] = length_stats["source_length_case_mean"]-length_stats["source_length_control_mean"]
                    length_stats["policy_length_increment_matched_contrast"] = length_stats["policy_length_increment_case_mean"]-length_stats["policy_length_increment_control_mean"]
                profiles[(model_key, pair, metric)] = {"groups": groups, "matching": matched}
                stats = {"source_policy": src, "target_policy": tgt, "metric": metric,
                    "groups": {k: [task_ids[i] for i in ids] for k, ids in groups.items()},
                    "persistent_or_new_task_ids": [task_ids[i] for i in groups["persistent"] + groups["newly"]],
                    "strict_joint_clean_task_ids": [task_ids[i] for i in controls], "excluded": exclusions,
                    "target_joint_clear_task_ids": [task_ids[i] for i in target_clear],
                    "target_other_flag_positive_task_ids": [task_ids[i] for i in other_positive],
                    "target_other_flag_unknown_task_ids": [task_ids[i] for i in other_unknown],
                    "matched_case_task_ids": [task_ids[i] for i in matched["matched_cases"]],
                    "matched_control_task_ids": [task_ids[i] for i in matched["matched_controls"]],
                    "unmatched_case_task_ids": [task_ids[i] for i in matched["missing_cases"]],
                    "input_length_increment_values": sorted(set((target_lengths-source_lengths).tolist())),
                    "all_source_input_length_range": [int(source_lengths.min()), int(source_lengths.max())],
                    **length_stats,
                    "singleton_case_cells": matched["singleton_case_cells"], "singleton_control_cells": matched["singleton_control_cells"]}
                if pair == STATE_PAIRS[0]:
                    observed = (100-len(exclusions), *(len(groups[k]) for k in ("disappeared","metric_clean","persistent","newly")), len(matched["matched_cases"]), len(matched["matched_controls"]), len(matched["missing_cases"]))
                    expected = STATE_EXPECTED_MAIN[model_key][STATE_METRICS.index(metric)]
                    if observed != expected: raise ValueError(f"Frozen label-group sanity check failed: {model_key}/{metric}: {observed} != {expected}")
                    if stats["input_length_increment_values"] != [{"qwen3_8b":40,"gemma2_9b":39,"ministral3_8b":42}[model_key]]:
                        raise ValueError("Verified B/E native token-length increment changed")
                    stats["frozen_label_group_check"] = {"expected": list(expected), "observed": list(observed), "matches": True}
                model_meta["pairs"].append(stats)
                text.append(f"{src} → {tgt}, {metric}: disappeared {len(groups['disappeared'])}; metric-clean {len(groups['metric_clean'])}; persistent {len(groups['persistent'])}; newly flagged {len(groups['newly'])}; excluded {len(exclusions)}. Strict joint-clean controls {len(controls)}; matched cases {len(matched['matched_cases'])}, unique controls {len(matched['matched_controls'])}, unmatched cases {len(matched['missing_cases'])}; target jointly clear {len(target_clear)}, other flag remains {len(other_positive)}, other flag unknown {len(other_unknown)}.\n")
                if "contrast" in matched and pair == STATE_PAIRS[0]:
                    ranked = np.argsort(-np.abs(matched["contrast"]))[:3]
                    peak = int(ranked[0])
                    stats["exploratory_peak_block"] = peak + 1
                    stats["exploratory_largest_absolute_contrast_blocks"] = [int(i)+1 for i in ranked]
                    text.append(f"Exploratory largest absolute contrast: block {peak+1}, {matched['contrast'][peak]:+.5f}, descriptive 95% interval [{matched['ci_low'][peak]:+.5f}, {matched['ci_high'][peak]:+.5f}]. Singleton cells: {matched['singleton_case_cells']} case / {matched['singleton_control_cells']} control.\n")
                text.append("Native policy token-length increments: " + repr(stats["input_length_increment_values"]) + "; matched source-length case-minus-control: " + repr(length_stats["source_length_matched_contrast"]) + ". Equal policy increments do not remove source-length or task-content confounding.\n")
                if model_key == "qwen3_8b" and pair == STATE_PAIRS[0] and metric == "false_success":
                    text.append("Only two Qwen false-success-label-disappeared cases have strict same-cell controls; 23 do not. Their singleton-cell bootstrap is fragile and does not support a mechanism or solid layer conclusion.\n")
                for layer in range(states.shape[1]):
                    row = {"model": spec["repo_id"], "metric": metric, "source_policy": src, "target_policy": tgt,
                        "layer_index": layer, "decoder_block": layer + 1, **{k + "_n": len(v) for k, v in groups.items()},
                        "strict_joint_clean_n": len(controls), "excluded_n": len(exclusions), "matched_case_n": len(matched["matched_cases"]),
                        "matched_unique_control_n": len(matched["matched_controls"]), "unmatched_case_n": len(matched["missing_cases"]),
                        "matched_strata_n": matched["matched_strata"], "target_joint_clear_n": len(target_clear),
                        "target_other_flag_positive_n": len(other_positive), "target_other_flag_unknown_n": len(other_unknown),
                        "singleton_case_cells": matched["singleton_case_cells"], "singleton_control_cells": matched["singleton_control_cells"]}
                    row["control_loo_weight_covered"] = matched.get("control_loo_weight_covered")
                    row.update(length_stats)
                    for key in ("case_distance", "control_distance", "contrast", "ci_low", "ci_high", "case_relative_l2", "control_relative_l2", "relative_l2_contrast", "relative_delta_mean_contrast_norm", "case_directional_coherence", "control_directional_coherence", "case_pairwise_direction_cosine", "control_pairwise_direction_cosine", "case_control_direction_alignment", "case_residual_pairwise_direction_cosine", "control_loo_residual_pairwise_direction_cosine", "case_residual_directional_coherence", "control_loo_residual_directional_coherence", "zero_case_deltas", "case_nonzero_residual_n", "control_nonzero_loo_residual_n"):
                        value = matched.get(key)
                        row[key] = float(value[layer]) if value is not None and np.isfinite(value[layer]) else None
                    table.append(row)
        primary = profiles[(model_key, STATE_PAIRS[0], "either")]
        paired = next(((i, j) for i in primary["matching"]["matched_cases"] for j in primary["matching"]["matched_controls"] if strata[i] == strata[j]), None)
        if paired:
            text.append("\nExact examples: one jointly resolved case and one already jointly clean control in the same category × pressure cell. These are different tasks.\n")
            for title, i in zip(("Jointly resolved case", "Strict joint-clean control"), paired):
                text.append(title + f"; cell={strata[i]}\n")
                _state_example(text, spec["repo_id"], task_ids[i], STATE_PAIRS[0], answers, labels, inputs, native, native_path)
        else: text.append("No eligible jointly resolved case with a same-cell strict-clean control; no example is substituted.\n")
        metadata.append(model_meta)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(table)
    _state_plot(profiles, output_prefix.with_suffix(".png"))
    output_prefix.with_suffix(".txt").write_text(_state_findings(metadata, table) + "".join(text), encoding="utf-8")
    manifest = {"models": metadata, "bootstrap_seed": STATE_SEED, "bootstrap_resamples": STATE_BOOTSTRAPS,
        "answer_source_sha256": sources["provenance"]["journal_snapshots"]["results/answers.jsonl"]["full_snapshot_sha256"], "label_source_sha256": sources["provenance"]["journal_snapshots"]["results/labels.jsonl"]["full_snapshot_sha256"],
        "validated_source_provenance": sources["provenance"],
        "analysis_implementation_sha256": state_analysis_implementation_hash(
            (_state_valid, _state_known, _state_joint_clear, _state_unit, _state_pairwise, _state_matching, _state_plot, _state_example,
             _state_validation_digest, _state_validation_sha, _state_validation_rows, state_read_frozen_sources, state_validate_cache, _state_findings, analyze),
            {"models": STATE_MODELS, "pairs": STATE_PAIRS, "metrics": STATE_METRICS, "seed": STATE_SEED, "bootstraps": STATE_BOOTSTRAPS, "expected_main": STATE_EXPECTED_MAIN}),
        "input_source_sha256": sha256_file(root / "data/model_inputs.jsonl"), "new_scoring_performed": False,
        "model_forward_passes_in_analysis": 0, "state_position": "last input token before any generated answer; decoder-block outputs before final norm",
        "controls": "Strict joint-clean at both endpoints, matched only on category×pressure; each matched case weights its cell's control mean equally.",
        "direction_note": "Clean LOO residuals are algebraically centered: two controls have antipodal directions; case and clean residual coherence are not a neutral comparison. Core v=(target-source)/source norm retains radial change. Residual directions normalize after subtracting matched control centroid; control residual coherence is leave-one-out. Pairwise direction cosine excludes self-pairs and needs at least two nonzero directions; resultant-length coherence depends on n.",
        "inference_limit": "Exploratory descriptive association; no classifier, causal inference, mechanism, or confirmatory layer claim. Bootstrap does not include uncertainty about saved labels."}
    manifest["pipeline_source_sha256"] = sha256_file(Path(__file__))
    manifest["published_artifacts"] = {output_prefix.with_suffix(ext).name:
        {"bytes": output_prefix.with_suffix(ext).stat().st_size, "sha256": sha256_file(output_prefix.with_suffix(ext))}
        for ext in (".csv", ".png", ".txt")}
    atomic_write_json(output_prefix.with_suffix(".json"), manifest)
    return manifest


# During-answer measurements are deliberately separate from the preserved FTA
# answers and from prompt-only extraction. This fragment is integrated into the
# existing internal-state pipeline; all heavy arrays live outside the project.
DURING_VERSION = "fta-during-answer-causal-replay-v1"
DURING_POLICIES = ("baseline", "transparency", "evidence_contract")


def during_answer_tokens(tokenizer, spec, response):
    """Recover saved IDs when available, otherwise retokenize exact saved text.

    Unicode offsets are checked against the full answer, never guessed by
    concatenating independently decoded pieces that can split UTF-8 characters.
    """
    import bisect
    text = response["response"]
    if spec["key"] == "ministral3_8b":
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        ids = list(response["output_token_ids"])
        backend = tokenizer.tokenizer.instruct_tokenizer.tokenizer
        pieces = [backend.id_to_byte_piece(i, special_token_policy=SpecialTokenPolicy.IGNORE) for i in ids]
        if b"".join(pieces) != text.encode("utf-8"):
            raise ValueError("Saved Ministral token bytes do not reproduce its exact visible answer")
        byte_boundaries, cursor = [0], 0
        for character in text:
            cursor += len(character.encode("utf-8"))
            byte_boundaries.append(cursor)
        offsets, byte_offsets, cursor = [], [], 0
        for piece in pieces:
            start, end = cursor, cursor + len(piece)
            cursor = end
            byte_offsets.append([start, end])
            if start == end:
                position = bisect.bisect_left(byte_boundaries, start)
                offsets.append([position, position])
            else:
                offsets.append([bisect.bisect_right(byte_boundaries, start) - 1,
                                bisect.bisect_left(byte_boundaries, end)])
        provenance = "original saved output_token_ids, including invisible terminal/control tokens"
        offset_method = "exact Tekken raw bytes, asserted equal to answer UTF-8; overlapping codepoint spans when bytes split a character"
    else:
        ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = list(ids)
        try:
            mapping = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                return_attention_mask=False)
            if list(mapping["input_ids"]) != ids:
                raise ValueError("Native offset token IDs differ from answer reconstruction")
            offsets = [list(item) for item in mapping["offset_mapping"]]
            offset_method = "native tokenizer offsets for exact saved-answer retokenization"
        except (NotImplementedError, TypeError):
            from tokenizers import Tokenizer
            fast = Tokenizer.from_file(str(cached_snapshot(spec) / "tokenizer.json"))
            encoded = fast.encode(text, add_special_tokens=False)
            if encoded.ids != ids:
                raise ValueError("Rust tokenizer offsets differ from native reconstructed answer IDs")
            offsets = [list(item) for item in encoded.offsets]
            offset_method = "pinned tokenizer.json Rust offsets; IDs checked against native tokenizer"
        boundaries = [len(text[:i].encode("utf-8")) for i in range(len(text) + 1)]
        byte_offsets = [[boundaries[start], boundaries[end]] for start, end in offsets]
        provenance = "retokenized exact saved visible answer; original generated token IDs and terminal EOS are unavailable"
    if not ids or len(ids) != len(offsets):
        raise ValueError("Missing answer IDs or offsets")
    decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if decoded != text:
        raise ValueError("Answer token reconstruction does not decode to the exact saved answer")
    if any(not (0 <= start <= end <= len(text)) for start, end in offsets):
        raise ValueError("Answer offsets fall outside saved text")
    return {"ids": ids, "offsets": offsets, "byte_offsets": byte_offsets,
            "token_id_provenance": provenance, "offset_method": offset_method,
            "exact_visible_decode_verified": True, "original_ids_available": spec["key"] == "ministral3_8b"}


def during_span_tokens(answer, annotations, tokenization):
    """Map checked exact annotation substrings to overlapping answer tokens."""
    events = []
    for annotation in annotations:
        start, end = annotation["start_char"], annotation["end_char"]
        if not (0 <= start < end <= len(answer)) or answer[start:end] != annotation["quoted_text"]:
            raise ValueError("Selected annotation does not quote its exact saved response")
        positions = [i for i, (left, right) in enumerate(tokenization["offsets"])
                     if left < end and right > start]
        if not positions:
            raise ValueError("Visible annotation has no overlapping answer tokens")
        events.append(dict(annotation, answer_token_indices=positions,
                           first_answer_token_index=positions[0], last_answer_token_index=positions[-1]))
    return events


def during_realize_prefix(tokenizer, baseline, tokenization, proposal):
    """Shorten a frozen supported text prefix only to an exact token boundary."""
    if not proposal.get("eligible"):
        return dict(proposal, input_token_ids=[], token_count=0)
    end = proposal["end_char"]
    if proposal["text"] != baseline[:end] or not (0 < end <= len(baseline)):
        raise ValueError("Frozen prefix is not a nonempty actual baseline-answer beginning")
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    best = None
    for count in range(1, len(tokenization["ids"]) + 1):
        ids = tokenization["ids"][:count]
        if ids[-1] in special:
            break
        decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if len(decoded) > end:
            break
        if decoded and baseline.startswith(decoded):
            best = count, ids, decoded
    if best is None:
        return dict(proposal, eligible=False, reason="No nonempty exact token boundary within the frozen supported prefix",
                    proposed_end_char=end, proposed_text=proposal["text"], text="", end_char=0,
                    input_token_ids=[], token_count=0)
    count, ids, decoded = best
    return dict(proposal, proposed_end_char=end, proposed_text=proposal["text"], text=decoded,
                end_char=len(decoded), input_token_ids=ids, token_count=count,
                prefix_token_ids_sha256=digest(ids), exact_baseline_token_slice_verified=True,
                next_baseline_token_id=(tokenization["ids"][count] if count < len(tokenization["ids"]) else None),
                adjustment="Greatest exact nonempty baseline token-slice boundary no later than frozen text endpoint")


def during_open_cache(path, method):
    import sqlite3
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=60)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, response_id TEXT NOT NULL, "
                       "scenario_id TEXT NOT NULL, condition TEXT NOT NULL, status TEXT NOT NULL, "
                       "metadata_json TEXT NOT NULL, arrays BLOB, PRIMARY KEY(kind,response_id))")
    previous = connection.execute("SELECT value FROM metadata WHERE name='method'").fetchone()
    if previous is None:
        connection.execute("INSERT INTO metadata VALUES ('method',?)", (json.dumps(method, ensure_ascii=False, sort_keys=True),))
        connection.commit()
    elif json.loads(previous[0]) != method:
        connection.close()
        raise ValueError("External during-answer cache has a different frozen method; it is preserved")
    return connection


def during_cached(connection, kind, response_id):
    import hashlib
    row = connection.execute("SELECT status,metadata_json,arrays FROM records WHERE kind=? AND response_id=?",
                             (kind, response_id)).fetchone()
    if row is None:
        return None
    status, encoded, arrays = row
    metadata = json.loads(encoded)
    if status != "complete":
        raise RuntimeError("Preserved continuation attempt is " + status + "; no automatic duplicate generation: " + response_id)
    if arrays is not None and hashlib.sha256(arrays).hexdigest() != metadata["arrays_sha256"]:
        raise ValueError("External state-record checksum differs")
    return metadata


def during_store(connection, kind, response_id, metadata, arrays=None, pending=False):
    import hashlib
    import io
    import numpy as np
    payload = None
    metadata = dict(metadata, kind=kind, response_id=response_id)
    if arrays is not None:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        metadata["arrays_sha256"] = hashlib.sha256(payload).hexdigest()
        metadata["array_shapes"] = {name: list(value.shape) for name, value in arrays.items()}
        metadata["array_dtypes"] = {name: str(value.dtype) for name, value in arrays.items()}
    with connection:
        if pending:
            connection.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?)",
                               (kind, response_id, metadata["scenario_id"], metadata["condition"], "pending",
                                json.dumps(metadata, ensure_ascii=False), payload))
        else:
            current = connection.execute("SELECT status FROM records WHERE kind=? AND response_id=?", (kind, response_id)).fetchone()
            if current is not None and current[0] == "complete":
                raise ValueError("Completed external record is immutable")
            connection.execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?)",
                               (kind, response_id, metadata["scenario_id"], metadata["condition"], "complete",
                                json.dumps(metadata, ensure_ascii=False), payload))
    return metadata


def during_read_cache(path):
    """Stream one external record's arrays at a time for downstream analysis."""
    import hashlib
    import io
    import sqlite3
    import numpy as np
    connection = sqlite3.connect("file:" + str(Path(path).resolve()) + "?mode=ro", uri=True)
    try:
        for kind, response_id, status, encoded, payload in connection.execute(
                "SELECT kind,response_id,status,metadata_json,arrays FROM records ORDER BY scenario_id,kind,condition"):
            if status != "complete":
                raise RuntimeError("External cache contains an unfinished record: " + response_id)
            metadata = json.loads(encoded)
            arrays = None
            if payload is not None:
                if hashlib.sha256(payload).hexdigest() != metadata["arrays_sha256"]:
                    raise ValueError("External compressed-array checksum mismatch")
                with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
            yield metadata, arrays
    finally:
        connection.close()


def during_capture(model, backbone, blocks, full_ids, positions, targets=None):
    """Capture pre-final-norm block outputs and the original wrapper's logits.

    In particular, Gemma's wrapper applies final logit softcapping. All supplied
    positions are before their next target token; attention remains causal.
    """
    import numpy as np
    import torch
    hidden = backbone.config.hidden_size
    states = np.empty((len(blocks), len(positions), hidden), dtype=np.float32)
    counts = [0] * len(blocks)
    device = backbone.get_input_embeddings().weight.device
    indices = torch.tensor(positions, dtype=torch.long, device=device)
    def make_hook(layer):
        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if tuple(value.shape) != (1, len(full_ids), hidden) or counts[layer]:
                raise ValueError("Unexpected during-answer block output/count")
            states[layer] = value[0].index_select(0, indices).detach().to("cpu", dtype=torch.float32).numpy()
            counts[layer] += 1
        return capture
    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(blocks)]
    tick = time.monotonic()
    try:
        ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           logits_to_keep=indices, output_hidden_states=False, output_attentions=False)
            logits = result.logits[0]
            if logits.shape[0] != len(positions):
                raise ValueError("Model wrapper did not return exactly the requested predictor logits")
            greedy_ids = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
            log_probs, greedy_log_probs = [], []
            for start in range(0, len(positions), 16):
                stop = min(start + 16, len(positions))
                values = torch.log_softmax(logits[start:stop].float(), dim=-1)
                chosen = torch.tensor(greedy_ids[start:stop], dtype=torch.long, device=device)
                greedy_log_probs.extend(values.gather(1, chosen[:, None])[:, 0].cpu().tolist())
                if targets is not None:
                    target = torch.tensor(targets[start:stop], dtype=torch.long, device=device)
                    log_probs.extend(values.gather(1, target[:, None])[:, 0].cpu().tolist())
            del result, logits, ids
    finally:
        for handle in handles:
            handle.remove()
    if counts != [1] * len(blocks) or not np.isfinite(states).all():
        raise ValueError("Missing or nonfinite block states")
    arrays = {"states": states, "predictor_positions": np.asarray(positions, dtype=np.int64),
              "greedy_next_token_ids": greedy_ids,
              "greedy_next_token_log_probabilities": np.asarray(greedy_log_probs, dtype=np.float32)}
    if targets is not None:
        values = np.asarray(log_probs, dtype=np.float32)
        if not np.isfinite(values).all() or np.any(values > 1e-5):
            raise ValueError("Invalid next-token log probabilities")
        arrays.update(target_token_ids=np.asarray(targets, dtype=np.int64),
                      next_token_log_probabilities=values,
                      next_token_probabilities=np.exp(values).astype(np.float32))
    return arrays, {"forward_seconds": round(time.monotonic() - tick, 6), "hook_counts": counts,
                    "states_all_finite": True, "original_model_wrapper_logits": True,
                    "gemma_final_logit_softcapping_preserved": type(model).__name__ == "Gemma2ForCausalLM"}


def during_replay(root, model_name, plan_path, cache_dir):
    """Replay fixed saved answers causally, then run separate same-prefix tails."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig
    root, cache_dir, plan_path = Path(root).resolve(), Path(cache_dir).expanduser().resolve(), Path(plan_path).resolve()
    if cache_dir == root or root in cache_dir.parents:
        raise ValueError("During-answer activations must be cached outside the project")
    spec = model_spec(root, model_name)
    if spec["key"] not in EXTRACTION_MODELS:
        raise ValueError("This study is restricted to Qwen, Gemma, and Ministral")
    plan = read_json(plan_path)
    sources = state_read_frozen_sources(root, STATE_MODELS, STATE_PAIRS, STATE_METRICS, STATE_SEED, STATE_BOOTSTRAPS)
    for relative, expected_hash in plan["source_sha256"].items():
        if sources["provenance"]["verified_frozen_source_sha256"].get(relative) != expected_hash:
            raise ValueError("During-answer source differs from frozen original result: " + relative)
    selected = plan["models"][spec["key"]]["selected_tasks"]
    if not selected or len(selected) > 20 or len({r["scenario_id"] for r in selected}) != len(selected):
        raise ValueError("Frozen subset must contain 1–20 distinct tasks per model")
    records = load_and_verify_inputs(root)
    snapshot, reference = verify_snapshot(spec, require_weights=True), extraction_reference(root, spec)
    tokenizer, hardware = load_tokenizer(spec), extraction_gpu(torch)
    if spec["key"] == "ministral3_8b" and hardware["compute_capability"][0] < 8:
        raise ValueError("Original Ministral BF16 inference requires its compatible allocated GPU")
    expected = (reference["runtime"]["hardware"]["packages"] if spec["key"] == "ministral3_8b"
                else reference["runtime"]["packages"])
    for name in ("torch", "transformers", "accelerate", "safetensors", "tokenizers"):
        if hardware["packages"][name] != expected[name]:
            raise ValueError("Saved runtime package version differs: " + name)
    if spec["key"] == "ministral3_8b":
        hardware["packages"]["mistral-common"] = importlib.metadata.version("mistral-common")
        if hardware["packages"]["mistral-common"] != expected["mistral-common"]:
            raise ValueError("Saved Mistral tokenizer version differs")
    encoded, checks = verify_extraction_inputs(root, spec, records, snapshot, reference, tokenizer)
    inputs = {(r["scenario_id"], r["condition"]): (r, t, c) for r, t, c in zip(records, encoded, checks)}
    saved = {(r["scenario_id"], r["condition"]): r for r, _ in sources["answers"].values()
             if r["model"] == spec["repo_id"] and r["repeat"] == 0}
    tokenized, prefixes, requested, exclusions = {}, {}, [], []
    for task in selected:
        for policy in DURING_POLICIES:
            key, frozen = (task["scenario_id"], policy), task["responses"][policy]
            row = saved[key]
            if row["response_id"] != frozen["response_id"] or digest(row["response"]) != frozen["response_sha256"] or row["revision"] != spec["revision"]:
                raise ValueError("Frozen selected saved response differs: " + str(key))
            if not frozen.get("eligible", True):
                exclusions.append({"scenario_id": key[0], "condition": policy, "response_id": row["response_id"],
                                   "reason": frozen.get("exclusion_reason", "Frozen policy-answer exclusion")})
                continue
            tokenized[key] = during_answer_tokens(tokenizer, spec, row)
            during_span_tokens(row["response"], frozen.get("annotations", []), tokenized[key])
            requested.append(("natural", row["response_id"]))
        baseline_key = task["scenario_id"], "baseline"
        if baseline_key not in tokenized:
            prefixes[task["scenario_id"]] = dict(task["prefix"], eligible=False,
                                                reason="Baseline answer excluded from natural replay")
        else:
            prefixes[task["scenario_id"]] = during_realize_prefix(tokenizer, saved[baseline_key]["response"],
                                                                  tokenized[baseline_key], task["prefix"])
        prefix = prefixes[task["scenario_id"]]
        if prefix.get("eligible"):
            for policy in DURING_POLICIES:
                identity = "during_" + digest([DURING_VERSION, spec["repo_id"], task["scenario_id"], policy,
                                                prefix["prefix_token_ids_sha256"], sha256_file(plan_path)])[:24]
                requested.extend((("boundary", identity), ("continuation", identity)))
    generation = dict(reference["runtime"]["generation"])
    max_new = 2048 if spec["key"] == "ministral3_8b" else 160
    if generation["max_new_tokens"] != max_new or generation["do_sample"] is not False or generation["num_beams"] != 1:
        raise ValueError("Saved generation settings differ from the original greedy continuation protocol")
    selected_seeds = {saved[(task["scenario_id"], policy)]["seed"] for task in selected for policy in DURING_POLICIES}
    if len(selected_seeds) != 1:
        raise ValueError("Frozen selected repetition has inconsistent original generation seeds")
    seed = selected_seeds.pop()
    method = {"version": DURING_VERSION, "model": spec["repo_id"], "revision": spec["revision"],
              "plan_sha256": sha256_file(plan_path), "logical_inputs_sha256": sha256_file(root / "data/model_inputs.jsonl"),
              "selected_response_metadata_sha256": digest([{k: saved[(t["scenario_id"], p)].get(k) for k in
                   ("response_id", "response", "revision", "output_token_ids", "finish_reason", "truncated")}
                   for t in selected for p in DURING_POLICIES]),
              "verified_frozen_source_sha256": sources["provenance"]["verified_frozen_source_sha256"],
              "checkpoint_metadata_sha256": snapshot["metadata_sha256"], "reference_hashes": reference["source_hashes"],
              "native_checks_sha256": digest(checks), "runtime_packages": hardware["packages"],
              "precision": "torch.bfloat16" if spec["key"] == "ministral3_8b" else "torch.float16",
              "attention_implementation": "eager", "batch_size": 1, "padding": False,
              "original_generation_settings": generation, "controlled_max_new_tokens": max_new,
              "controlled_seed": seed,
              "controlled_repetitions": 1, "same_prefix_ids": True, "assistant_turn_closed": False,
              "natural_saved_answers_only": True, "capture": "block output before final norm, before target answer token j at P+j−1",
              "teacher_forcing_use_cache": False, "generation_use_cache": True,
              "next_token_probability_definition": "Original model conditional softmax including Gemma final logit softcapping, before generation processors; target is the supplied saved token",
              "implementation_sha256": digest("\n".join(inspect.getsource(f) for f in
                   (during_answer_tokens, during_span_tokens, during_realize_prefix, during_open_cache,
                    during_cached, during_store, during_capture, during_replay)))}
    db_path = cache_dir / (spec["key"] + ".during.sqlite")
    connection = during_open_cache(db_path, method)
    all_complete = all(during_cached(connection, kind, response_id) is not None for kind, response_id in requested)
    print(json.dumps({"status": "during_native_inputs_verified", "model": spec["key"], "all_inputs": len(checks),
                      "selected_tasks": len(selected), "natural_answers": len(tokenized),
                      "eligible_prefixes": sum(bool(p.get("eligible")) for p in prefixes.values()),
                      "completed_cache_reused": all_complete}), flush=True)
    model = None
    if not all_complete:
        kwargs = {"local_files_only": True, "trust_remote_code": False, "attn_implementation": "eager",
                  "torch_dtype": torch.bfloat16 if spec["key"] == "ministral3_8b" else torch.float16}
        if spec["key"] == "ministral3_8b":
            from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration
            kwargs.update(quantization_config=FineGrainedFP8Config(dequantize=True), device_map="auto")
            model_class = Mistral3ForConditionalGeneration
        else:
            kwargs["low_cpu_mem_usage"] = True
            model_class = AutoModelForCausalLM
        model = model_class.from_pretrained(str(cached_snapshot(spec)), **kwargs)
        if spec["key"] != "ministral3_8b":
            model.to("cuda:0")
        if any(p.device.type != "cuda" for p in model.parameters()) or type(model).__name__ != reference["runtime"]["model_class"]:
            raise ValueError("Original single-GPU model class/loading differs; no offload or substitute")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        backbone_path = "model.language_model" if spec["key"] == "ministral3_8b" else "model"
        backbone = model.get_submodule(backbone_path)
        blocks = backbone.layers
        if len(blocks) != backbone.config.num_hidden_layers:
            raise ValueError("Unexpected decoder block ordering")
        config = GenerationConfig(**reference["runtime"]["effective_generation_config"]) if spec["key"] == "ministral3_8b" else None
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        for task in selected:
            for policy in DURING_POLICIES:
                key = task["scenario_id"], policy
                if key not in tokenized:
                    continue
                row, tokens = saved[key], tokenized[key]
                if during_cached(connection, "natural", row["response_id"]) is not None:
                    continue
                logical, native, check = inputs[key]
                prompt = native["input_ids"]
                positions = list(range(len(prompt) - 1, len(prompt) + len(tokens["ids"]) - 1))
                arrays, timing = during_capture(model, backbone, blocks, prompt + tokens["ids"][:-1], positions, tokens["ids"])
                arrays.update(answer_offsets=np.asarray(tokens["offsets"], dtype=np.int64),
                              answer_byte_offsets=np.asarray(tokens["byte_offsets"], dtype=np.int64))
                events = during_span_tokens(row["response"], task["responses"][policy].get("annotations", []), tokens)
                for event in events:
                    event["predictor_positions"] = [positions[i] for i in event["answer_token_indices"]]
                metadata = dict(timing, scenario_id=key[0], condition=policy, model=spec["repo_id"],
                                role=task["role"], metrics=task["metrics"], category=task["category"], pressure_type=task["pressure_type"],
                                saved_response_id=row["response_id"], saved_answer_source=task["responses"][policy],
                                response=row["response"], response_sha256=digest(row["response"]), native_check=check,
                                input_length=len(prompt), answer_token_ids=tokens["ids"], answer_offsets=tokens["offsets"],
                                answer_byte_offsets=tokens["byte_offsets"], token_id_provenance=tokens["token_id_provenance"],
                                original_ids_available=tokens["original_ids_available"], offset_method=tokens["offset_method"],
                                exact_visible_decode_verified=True, saved_generation_token_count=row["token_count"],
                                replay_target_count=len(tokens["ids"]), predictor_positions=positions, events=events,
                                next_token_probabilities=arrays["next_token_probabilities"].tolist(),
                                next_token_log_probabilities=arrays["next_token_log_probabilities"].tolist(),
                                greedy_next_token_ids=arrays["greedy_next_token_ids"].tolist(),
                                answer_tokens_generated=False, states_shape=list(arrays["states"].shape))
                during_store(connection, "natural", row["response_id"], metadata, arrays)
                print(json.dumps({"model": spec["key"], "status": "natural_answer_replayed", "scenario_id": key[0],
                                  "condition": policy, "target_tokens": len(tokens["ids"])}), flush=True)
                del arrays
            prefix = prefixes[task["scenario_id"]]
            if not prefix.get("eligible"):
                continue
            for policy in DURING_POLICIES:
                key = task["scenario_id"], policy
                logical, native, check = inputs[key]
                identity = "during_" + digest([DURING_VERSION, spec["repo_id"], key[0], policy,
                                                prefix["prefix_token_ids_sha256"], sha256_file(plan_path)])[:24]
                full_ids = native["input_ids"] + prefix["input_token_ids"]
                common = {"scenario_id": key[0], "condition": policy, "model": spec["repo_id"],
                          "role": task["role"], "metrics": task["metrics"], "category": task["category"], "pressure_type": task["pressure_type"],
                          "prefix": prefix, "native_check": check, "input_length": len(native["input_ids"]),
                          "prefilled_input_length": len(full_ids), "prefilled_input_ids_sha256": digest(full_ids),
                          "baseline_source_response_id": saved[(key[0], "baseline")]["response_id"],
                          "controlled_experiment": "actual supported baseline-answer prefix supplied under each original policy",
                          "original_FTA_result": False, "assistant_turn_closed": False}
                if during_cached(connection, "boundary", identity) is None:
                    target = prefix["next_baseline_token_id"]
                    arrays, timing = during_capture(model, backbone, blocks, full_ids, [len(full_ids) - 1],
                                                    [target] if target is not None else None)
                    arrays["states"] = arrays["states"][:, 0, :]
                    metadata = dict(common, **timing, predictor_position=len(full_ids) - 1,
                                    states_shape=list(arrays["states"].shape), greedy_next_token_id=int(arrays["greedy_next_token_ids"][0]),
                                    greedy_next_token_log_probability=float(arrays["greedy_next_token_log_probabilities"][0]),
                                    baseline_next_token_id=target,
                                    baseline_next_token_probability=(float(arrays["next_token_probabilities"][0]) if target is not None else None))
                    during_store(connection, "boundary", identity, metadata, arrays)
                    del arrays
                if during_cached(connection, "continuation", identity) is not None:
                    continue
                context_limit = getattr(backbone.config, "max_position_embeddings", 262144)
                if len(full_ids) + max_new > context_limit:
                    raise ValueError("Same-prefix context and original generation allowance exceed checkpoint limit")
                # A durable pending record prevents a crash from silently
                # producing a second greedy continuation on automatic resume.
                during_store(connection, "continuation", identity,
                             dict(common, generation_started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()), pending=True)
                tensor = torch.tensor([full_ids], dtype=torch.long, device=backbone.get_input_embeddings().weight.device)
                tick = time.monotonic()
                with torch.inference_mode():
                    if config is not None:
                        output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), generation_config=config)[0]
                    else:
                        output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                                do_sample=False, num_beams=1, use_cache=True, max_new_tokens=max_new,
                                                repetition_penalty=1.0, length_penalty=1.0, temperature=None, top_p=None, top_k=None,
                                                pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=False, output_scores=False,
                                                logits_to_keep=1)[0]
                generated = output[len(full_ids):].detach().cpu().tolist()
                eos_value = config.eos_token_id if config is not None else model.generation_config.eos_token_id
                eos = set(eos_value if isinstance(eos_value, (list, tuple)) else [eos_value])
                stop = next((i for i, value in enumerate(generated) if value in eos), None)
                if stop is not None:
                    generated = generated[:stop + 1]
                # SentencePiece may drop a generated token's leading space
                # when decoded alone. Decode the complete assistant ID stream
                # so exact contextual text, including joining whitespace,
                # is what the continuation evaluator receives.
                full_text = tokenizer.decode(prefix["input_token_ids"] + generated, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
                if not full_text.startswith(prefix["text"]):
                    raise ValueError("Contextual continuation decoding changed its frozen actual prefix")
                text = full_text[len(prefix["text"]):]
                metadata = dict(common, output_token_ids=generated, token_count=len(generated), continuation=text,
                                full_assistant_text=full_text,
                                standalone_generated_text=tokenizer.decode(generated, skip_special_tokens=True,
                                                                         clean_up_tokenization_spaces=False),
                                raw_generated_text=tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                                finish_reason="eos" if stop is not None else "length", truncated=stop is None,
                                generation_seconds=round(time.monotonic() - tick, 6), max_new_tokens=max_new,
                                repetition=0, greedy=True, seed=seed, scoring_status="unscored_new_continuation",
                                prior_FTA_labels_reused_for_continuation=False, technical_error=None)
                during_store(connection, "continuation", identity, metadata)
                print(json.dumps({"model": spec["key"], "status": "controlled_continuation_saved", "scenario_id": key[0],
                                  "condition": policy, "generated_tokens": len(generated), "truncated": stop is None}), flush=True)
                del output, tensor
        rows = [json.loads(row[0]) for row in connection.execute("SELECT metadata_json FROM records WHERE status='complete' ORDER BY scenario_id,kind,condition")]
        if len(rows) != len(requested) or any(during_cached(connection, kind, identity) is None for kind, identity in requested):
            raise ValueError("External cache does not cover every frozen eligible record")
    finally:
        connection.close()
    records_path = cache_dir / (spec["key"] + ".during.records.jsonl")
    manifest_path = cache_dir / (spec["key"] + ".during.manifest.json")
    if records_path.exists():
        if read_jsonl(records_path) != rows:
            raise ValueError("Preserved during-answer metadata export differs")
    else:
        atomic_write_jsonl(records_path, rows, overwrite=False)
    manifest = {"status": "complete", "version": DURING_VERSION, "method": method,
                "method_sha256": digest(method), "hardware": hardware, "model_key": spec["key"],
                "selected_tasks": len(selected), "natural_answers": len(tokenized), "natural_policy_exclusions": exclusions,
                "prefixes": [{"scenario_id": task["scenario_id"], **prefixes[task["scenario_id"]]} for task in selected],
                "controlled_continuations": sum(bool(p.get("eligible")) for p in prefixes.values()) * 3,
                "block_definition": "Decoder block outputs before final decoder norm; token j predicted at native input length P+j−1",
                "causal_teacher_forcing": True, "original_answers_generated": False,
                "reconstruction_limit": "Ministral original IDs preserved. Qwen/Gemma visible saved answers retokenized exactly; historical generated IDs/EOS unavailable. Batch1 numerical reconstruction differs from historical Qwen/Gemma padded batch4 generation.",
                "same_prefix_limit": "Identical actual baseline-prefix IDs reduce wording-history differences but policy/format effects remain; answer prefill changes original FTA experiment.",
                "files": {"sqlite": {"path": str(db_path), "sha256": sha256_file(db_path), "bytes": db_path.stat().st_size},
                          "records": {"path": str(records_path), "sha256": sha256_file(records_path), "bytes": records_path.stat().st_size}}}
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["method"] != method or previous["files"] != manifest["files"]:
            raise ValueError("Preserved completed manifest differs")
        return previous
    atomic_write_json(manifest_path, manifest, overwrite=False)
    print(json.dumps({"status": "during_study_complete", "model": spec["key"], "natural_answers": len(tokenized),
                      "controlled_continuations": manifest["controlled_continuations"], "cache": str(db_path)}), flush=True)
    return manifest


_DA_POLICIES = ("baseline", "transparency", "evidence_contract")
_DA_PAIRS = (("baseline", "evidence_contract"), ("baseline", "transparency"),
             ("transparency", "evidence_contract"))
_DA_METRICS = ("false_success", "fabricated_details")
_DA_NAMES = {"false_success": "unsupported success", "fabricated_details": "invented details"}
_DA_BLOCKS = {"qwen3_8b": 24, "gemma2_9b": 38, "ministral3_8b": 34}


def during_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def during_file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def during_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def during_geometry(source, target):
    import numpy as np
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    source_norm, target_norm = np.linalg.norm(source, axis=-1), np.linalg.norm(target, axis=-1)
    if not np.isfinite(source).all() or not np.isfinite(target).all() or (source_norm <= 0).any() or (target_norm <= 0).any():
        raise ValueError("State geometry needs finite nonzero block vectors")
    cosine = np.clip(1 - np.sum(source * target, axis=-1) / (source_norm * target_norm), 0, 2)
    relative_l2 = np.linalg.norm(target - source, axis=-1) / source_norm
    return cosine, relative_l2


def during_cell(task):
    return task["category"], task["pressure_type"]


def during_known(label, name):
    return (type(label.get("rubric_labels", {}).get(name)) is bool
            and label.get("confidence", {}).get(name) in (2, 3)
            and name not in label.get("unresolved_labels", []))


def during_review_index(reviews_path, continuations):
    """Validate provided NEW decisions; no semantic inference or old label audit."""
    reviews = during_jsonl(reviews_path)
    result = {item["response_id"]: item for item in reviews}
    if len(result) != len(reviews) or set(result) != set(continuations):
        raise ValueError("Every new continuation requires exactly one supplied review")
    for response_id, item in result.items():
        answer = continuations[response_id]["response"]
        if item.get("scoring_passes") != 1 or item.get("response_sha256") != during_hash(answer):
            raise ValueError("New review is duplicated or refers to another continuation: " + response_id)
        for name in (*_DA_METRICS, "limitation_disclosed"):
            value, confidence = item.get("rubric_labels", {}).get(name), item.get("confidence", {}).get(name)
            if type(value) is not bool or confidence not in (1, 2, 3):
                raise ValueError("Three explicit decisions and confidence are required: " + response_id)
            spans = item.get("evidence_spans", {}).get(name, [])
            if value and not spans:
                raise ValueError("New positive decision lacks an exact response span")
            for span in spans:
                start, end = span["start"], span["end"]
                if not 0 <= start < end <= len(answer) or answer[start:end] != span["text"]:
                    raise ValueError("New-review span differs from the supplied full context")
                if name in _DA_METRICS and start < len(continuations[response_id]["prefix"]["text"]):
                    raise ValueError("Unsupported-claim evidence must start in the newly generated continuation")
            if confidence == 1 and not (item.get("uncertainty", {}).get(name) or item.get("notes")):
                raise ValueError("Unresolved new judgment needs an explanation")
    return result


def during_event_tokens(tokens, span):
    indices = [j for j, token in enumerate(tokens)
               if token["char_start"] < span["end"] and token["char_end"] > span["start"]]
    if not indices:
        raise ValueError("Saved positive annotation has no overlapping answer token")
    return indices[0], indices[-1]


def during_stage_indices(onset, last, token_count):
    return {"before": onset - 1 if onset > 0 else None,
            "onset": onset, "middle": onset + (last - onset) // 2,
            "last": last, "after": last + 1 if last + 1 < token_count else None}


def during_cache_records(cache_dir, model_key, plan_sha256, model_reference):
    """Verify complete per-record and whole-database hashes, then stream arrays."""
    import io
    import numpy as np
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / (model_key + ".during.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["method"]["plan_sha256"] != plan_sha256:
        raise ValueError("During-answer cache is incomplete or belongs to another frozen subset")
    method = manifest["method"]
    current_runtime_hash = digest("\n".join(inspect.getsource(f) for f in
        (during_answer_tokens, during_span_tokens, during_realize_prefix, during_open_cache,
         during_cached, during_store, during_capture, during_replay)))
    if method["implementation_sha256"] != current_runtime_hash:
        raise ValueError("Completed cache differs from the published runtime implementation")
    method_hash = hashlib.sha256(json.dumps(method, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if (manifest["model_key"] != model_key or method["model"] != model_reference["repo_id"]
            or method["revision"] != model_reference["revision"] or manifest["method_sha256"] != method_hash
            or method["checkpoint_metadata_sha256"] != model_reference["checkpoint_metadata_sha256"]):
        raise ValueError("Cache checkpoint/model/revision or method signature differs from frozen originals")
    db_path = cache_dir / (model_key + ".during.sqlite")
    records_path = cache_dir / (model_key + ".during.records.jsonl")
    for key, path in (("sqlite", db_path), ("records", records_path)):
        if during_file_hash(path) != manifest["files"][key]["sha256"]:
            raise ValueError("Completed cache file checksum differs: " + str(path))
    connection = sqlite3.connect("file:" + str(db_path.resolve()) + "?mode=ro", uri=True)
    try:
        method = json.loads(connection.execute("SELECT value FROM metadata WHERE name='method'").fetchone()[0])
        if method != manifest["method"]:
            raise ValueError("Database and manifest methods differ")
        exported = {(row["kind"], row["response_id"]): row for row in during_jsonl(records_path)}
        for kind, response_id, status, encoded, payload in connection.execute(
                "SELECT kind,response_id,status,metadata_json,arrays FROM records ORDER BY scenario_id,kind,condition"):
            if status != "complete":
                raise ValueError("Cache contains an unfinished or duplicate continuation attempt")
            metadata = json.loads(encoded)
            if exported.pop((kind, response_id), None) != metadata:
                raise ValueError("Metadata export differs from exact stored record")
            arrays = None
            if payload is not None:
                if hashlib.sha256(payload).hexdigest() != metadata["arrays_sha256"]:
                    raise ValueError("Compressed state record checksum differs")
                with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                if arrays["states"].dtype != np.float32 or not np.isfinite(arrays["states"]).all():
                    raise ValueError("Cached block states must be finite float32")
                if {name: list(a.shape) for name, a in arrays.items()} != metadata["array_shapes"]:
                    raise ValueError("Declared compressed state-array shapes differ")
                if {name: str(a.dtype) for name, a in arrays.items()} != metadata["array_dtypes"]:
                    raise ValueError("Declared compressed state-array precision differs")
                shape = arrays["states"].shape
                if shape[0] != model_reference["blocks"] or shape[-1] != model_reference["hidden_size"] or len(shape) != (3 if kind == "natural" else 2):
                    raise ValueError("Cached decoder block count/hidden dimensions/layout differ")
            yield manifest, metadata, arrays
        if exported:
            raise ValueError("Export contains records absent from the database")
    finally:
        connection.close()


def during_boundary_rows(model_key, selected, boundary):
    """Use fixed original main-pair memberships for every policy comparison."""
    import numpy as np
    tasks = {task["scenario_id"]: task for task in selected}
    rows, groups = [], []
    for source, target in _DA_PAIRS:
        available = {sid for sid in tasks if (sid, source) in boundary and (sid, target) in boundary}
        controls = [task for task in selected if task["role"] == "control" and task["scenario_id"] in available]
        control_cells = {}
        for task in controls:
            control_cells.setdefault(during_cell(task), []).append(task["scenario_id"])
        movement = {sid: during_geometry(boundary[sid, source][1], boundary[sid, target][1]) for sid in available}
        for metric in _DA_METRICS:
            fixed_cases = [t for t in selected if t["role"] == "case" and metric in t["metrics"]]
            eligible = [t for t in fixed_cases if t["scenario_id"] in available]
            matched = [t for t in eligible if control_cells.get(during_cell(t))]
            unmatched = [t["scenario_id"] for t in eligible if not control_cells.get(during_cell(t))]
            used_controls = sorted({sid for task in matched for sid in control_cells[during_cell(task)]})
            group = {"model_key": model_key, "source_policy": source, "target_policy": target,
                     "metric": metric, "fixed_natural_cases": len(fixed_cases),
                     "eligible_prefix_cases": len(eligible), "matched_prefix_cases": len(matched),
                     "eligible_prefix_controls": len(controls), "matched_unique_controls": len(used_controls),
                     "natural_only_case_ids": [t["scenario_id"] for t in fixed_cases if t["scenario_id"] not in available],
                     "prefix_unmatched_case_ids": unmatched,
                     "case_ids": [t["scenario_id"] for t in matched], "control_ids": used_controls,
                     "case_control_matches": {t["scenario_id"]: control_cells[during_cell(t)] for t in matched},
                     "cohort_note": "Fixed baseline-to-contract label-disappeared memberships; secondary pairs do not reselect cases.",
                     "matching_note": "Equal case weights; each case compared with mean available strict-clean controls in its exact category × pressure cell."}
            groups.append(group)
            if not matched:
                continue
            case_cos, control_cos, case_l2, control_l2 = [], [], [], []
            for task in matched:
                sid, cell = task["scenario_id"], during_cell(task)
                case_cos.append(movement[sid][0]); case_l2.append(movement[sid][1])
                control_cos.append(np.mean([movement[c][0] for c in control_cells[cell]], axis=0))
                control_l2.append(np.mean([movement[c][1] for c in control_cells[cell]], axis=0))
            a, b, c, d = [np.mean(v, axis=0) for v in (case_cos, control_cos, case_l2, control_l2)]
            for layer in range(len(a)):
                rows.append({"record_type": "shared_prefix_boundary", "model_key": model_key,
                    "source_policy": source, "target_policy": target, "metric": metric, "block": layer + 1,
                    "case_count": len(matched), "unique_control_count": len(used_controls),
                    "eligible_prefix_cases": len(eligible), "eligible_prefix_controls": len(controls),
                    "unmatched_case_count": len(unmatched), "natural_only_case_count": len(fixed_cases) - len(eligible),
                    "case_cosine_distance": float(a[layer]), "control_cosine_distance": float(b[layer]),
                    "cosine_case_minus_control": float(a[layer] - b[layer]),
                    "case_relative_l2": float(c[layer]), "control_relative_l2": float(d[layer]),
                    "relative_l2_case_minus_control": float(c[layer] - d[layer])})
    return rows, groups


def during_natural_rows(model_key, metadata, arrays):
    """Actual annotated events, never align answers by an arbitrary ordinal."""
    rows, summaries = [], []
    states = arrays["states"]
    event_counts = {}
    for index, event in enumerate(metadata["events"]):
        onset, last = event["first_answer_token_index"], event["last_answer_token_index"]
        stages = during_stage_indices(onset, last, states.shape[1])
        reference = onset
        summary = {"response_id": metadata["response_id"], "event_index": index,
                   "event": event, "event_onset_reference_answer_token_index": reference,
                   "preceding_answer_token_index": onset - 1 if onset > 0 else None,
                   "stages": stages, "prior_fd_candidate_block": _DA_BLOCKS[model_key],
                   "missing_prior_stage_reason": "Event starts at first answer token; its onset predictor is final input position, but the earlier answer-token stage is unavailable" if onset == 0 else None}
        summaries.append(summary)
        label = event.get("label", event.get("event_type", event.get("kind")))
        if label is None:
            raise ValueError("Selected event lacks its explicit saved-label type")
        label_index = event_counts.get(label, 0)
        event_counts[label] = label_index + 1
        published_layer = 36 if model_key == "qwen3_8b" and label == "false_success" else _DA_BLOCKS[model_key]
        for stage, position in stages.items():
            if position is not None and reference is not None:
                cosine, relative_l2 = during_geometry(states[:, reference, :], states[:, position, :])
            else:
                cosine, relative_l2 = [None] * states.shape[0], [None] * states.shape[0]
            for layer in (published_layer - 1,):
                offset = metadata["answer_offsets"][position] if position is not None else [None, None]
                rows.append({"record_type": "natural_event", "model_key": model_key,
                    "condition": metadata["condition"], "scenario_id": metadata["scenario_id"],
                    "response_id": metadata["response_id"], "role": metadata["role"], "metric": label,
                    "event_index": index, "event_index_within_label": label_index, "stage": stage, "block": layer + 1,
                    "annotation_eligible": event.get("eligible", True),
                    "disclosure_kind": "status_only" if label == "limitation_disclosed" and event["quoted_text"].strip().upper() in ("BLOCKED", "STATUS: BLOCKED", "STATUS:BLOCKED") else "prose" if label == "limitation_disclosed" else None,
                    "case_details_membership": "fabricated_details" in metadata["metrics"],
                    "event_start_char": event["start_char"], "event_end_char": event["end_char"],
                    "event_first_answer_token": onset, "event_last_answer_token": last,
                    "reference_answer_token": reference, "answer_token_index": position,
                    "predictor_position": metadata["predictor_positions"][position] if position is not None else None,
                    "target_token_id": metadata["answer_token_ids"][position] if position is not None else None,
                    "target_char_start": offset[0], "target_char_end": offset[1],
                    "target_text": metadata["response"][offset[0]:offset[1]] if position is not None else None,
                    "saved_next_token_probability": metadata["next_token_probabilities"][position] if position is not None else None,
                    "saved_next_token_log_probability": metadata["next_token_log_probabilities"][position] if position is not None else None,
                    "cosine_from_event_onset": float(cosine[layer]) if cosine[layer] is not None else None,
                    "relative_l2_from_event_onset": float(relative_l2[layer]) if relative_l2[layer] is not None else None})
    return rows, summaries


def during_disclosure_origin(row, review):
    """Describe exact supporting-span positions; never infer or change a label."""
    spans = review["evidence_spans"]["limitation_disclosed"]
    prefix_length = len(row["prefix"]["text"])
    if not spans:
        raise ValueError("A positive disclosure needs at least one exact supporting span")
    if all(span["end"] <= prefix_length for span in spans):
        return "supplied_prefix_only"
    if all(span["start"] >= prefix_length for span in spans):
        return "generated_tail_only"
    return "mixed"


def during_outcome_rows(model_key, selected, continuations, reviews):
    rows = []
    for role in ("case", "control"):
        for policy in _DA_POLICIES:
            for metric in (*_DA_METRICS, "limitation_disclosed"):
                included, excluded, positive, new_ids = [], [], [], []
                disclosure_origins = {"supplied_prefix_only": 0, "generated_tail_only": 0, "mixed": 0}
                members = {t["scenario_id"] for t in selected if t["role"] == role}
                for row in continuations.values():
                    if row["model_key"] != model_key or row["scenario_id"] not in members or row["condition"] != policy:
                        continue
                    response_id = row["response_id"]
                    new_ids.append(response_id)
                    review = reviews[response_id]
                    if (row.get("truncated") or row.get("technical_error") or row.get("finish_reason") != "eos"
                            or not during_known(review, metric)):
                        excluded.append(response_id)
                    else:
                        included.append(response_id)
                        if review["rubric_labels"][metric]:
                            positive.append(response_id)
                            if metric == "limitation_disclosed":
                                disclosure_origins[during_disclosure_origin(row, review)] += 1
                rows.append({"record_type": "new_continuation_outcome", "model_key": model_key,
                    "role": role, "condition": policy, "metric": metric,
                    "case_count": len(members) if role == "case" else None,
                    "new_reports_count": len(new_ids), "numerator": len(positive),
                    "denominator": len(included), "excluded_count": len(excluded),
                    "disclosures_supplied_prefix_only": disclosure_origins["supplied_prefix_only"] if metric == "limitation_disclosed" else None,
                    "disclosures_generated_tail_only": disclosure_origins["generated_tail_only"] if metric == "limitation_disclosed" else None,
                    "disclosures_mixed": disclosure_origins["mixed"] if metric == "limitation_disclosed" else None,
                    "positive_response_ids": json.dumps(positive), "excluded_response_ids": json.dumps(excluded),
                    "cohort_definition": "All selected tasks in this role with new supplied-prefix continuations, scored once in full context; original flag membership does not restrict new-rate denominators"})
    return rows


def during_figure(path, table, groups):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = tuple(_DA_BLOCKS)
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    colors = {"false_success": "#2869a1", "fabricated_details": "#cb6833"}
    stages = ("before", "onset", "middle", "last", "after")
    for i, model in enumerate(models):
        left, right = axes[i]
        count_lines = []
        for metric in _DA_METRICS:
            entries = sorted((r for r in table if r["record_type"] == "shared_prefix_boundary"
                and r["model_key"] == model and r["source_policy"] == "baseline"
                and r["target_policy"] == "evidence_contract" and r["metric"] == metric), key=lambda r: r["block"])
            group = next(g for g in groups if g["model_key"] == model and g["source_policy"] == "baseline"
                         and g["target_policy"] == "evidence_contract" and g["metric"] == metric)
            count_lines.append(f"{_DA_NAMES[metric]}: {group['matched_prefix_cases']} cases / {group['matched_unique_controls']} controls")
            if entries:
                left.plot([r["block"] for r in entries], [r["cosine_case_minus_control"] for r in entries],
                          color=colors[metric], label=_DA_NAMES[metric], linewidth=1.6)
        left.axhline(0, color="#777777", linewidth=.8)
        left.set_title(model.replace("_", " ") + " — same prefix")
        left.set_xlabel("Decoder block (baseline → contract; 1-based)", fontsize=9); left.set_ylabel("Cosine movement: cases − controls")
        left.text(.02, .04, "\n".join(count_lines), transform=left.transAxes, fontsize=8, va="bottom")
        if left.lines and any(line.get_label() in _DA_NAMES.values() for line in left.lines):
            left.legend(loc="upper left" if model == "qwen3_8b" else "upper right", fontsize=8)
        natural = [r for r in table if r["record_type"] == "natural_event" and r["model_key"] == model
                   and r["role"] == "case" and r["event_index_within_label"] == 0 and r["annotation_eligible"]
                   and r["block"] == _DA_BLOCKS[model] and r["case_details_membership"]]
        baseline_ids = {r["scenario_id"] for r in natural if r["condition"] == "baseline" and r["metric"] == "fabricated_details"}
        contract_ids = {r["scenario_id"] for r in natural if r["condition"] == "evidence_contract" and r["metric"] == "limitation_disclosed"}
        paired_ids = baseline_ids & contract_ids
        natural = [r for r in natural if r["scenario_id"] in paired_ids]
        for policy, event_type, color, title in (("baseline", "fabricated_details", "#cb6833", "Baseline invented-detail span"),
                ("evidence_contract", "limitation_disclosed", "#2869a1", "Contract failure admission")):
            entries = [r for r in natural if r["condition"] == policy and r["metric"] == event_type]
            means, counts = [], []
            for stage in stages:
                values = [r["cosine_from_event_onset"] for r in entries if r["stage"] == stage and r["cosine_from_event_onset"] is not None]
                means.append(float(np.mean(values)) if values else np.nan); counts.append(len(values))
            right.plot(range(len(stages)), means, color=color, marker="o", linewidth=1.4,
                       label=title + " (N=" + "/".join(map(str, counts)) + ")")
        right.set_xticks(range(len(stages)), stages)
        right.set_ylabel("Cosine from pre-first-span-token state")
        right.set_title(f"Natural replay — prior block {_DA_BLOCKS[model]}\nDifferent words and histories; descriptive", fontsize=10)
        right.legend(fontsize=8, loc="best")
        right.set_xlabel("Span stage (onset = reference; zero by definition)", fontsize=9)
        for axis in (left, right):
            axis.grid(alpha=.18)
    figure.suptitle("Exploratory during-answer study\nSupplying a saved answer beginning changes the experiment", fontsize=13)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def during_example_text(model_key, model_plan, natural, continuations, reviews, event_rows):
    lines = ["\nEXACT EXAMPLES — " + model_plan["repo_id"], "Displayed case: first lexicographic prefix-eligible case, with its first eligible same-cell control; selected independently of new states or labels. These missing-attachment/expected-answer examples do not represent every task family or pressure condition."]
    candidates = [t for t in model_plan["selected_tasks"] if t["role"] == "case"]
    candidates.sort(key=lambda t: (not any(r["scenario_id"] == t["scenario_id"] and r["model_key"] == model_key for r in continuations.values()), t["scenario_id"]))
    case = candidates[0]
    controls = [t for t in model_plan["selected_tasks"] if t["role"] == "control" and during_cell(t) == during_cell(case)]
    control = next((t for t in controls if any(r["scenario_id"] == t["scenario_id"] and r["model_key"] == model_key for r in continuations.values())), controls[0] if controls else None)
    for task in (case, control):
        if task is None:
            lines.append("No selected same-cell control for this displayed case."); continue
        sid = task["scenario_id"]
        lines.extend([f"\n{task['role'].upper()}: {sid}; cell={during_cell(task)}; original metric membership={task['metrics']}"])
        first = natural.get((model_key, sid, "baseline"))
        if first:
            lines.extend(["EXACT ORIGINAL USER REQUEST INCLUDING TOOL OBSERVATION:", first["native_check"]["original_messages"][1]["content"],
                          "EXACT SAVED BASELINE ANSWER:", first["response"],
                          "Source: " + task["responses"]["baseline"]["source_answer_file"] + ":" + str(task["responses"]["baseline"]["source_answer_row"])])
        for policy in _DA_POLICIES:
            original = natural.get((model_key, sid, policy))
            new = next((r for r in continuations.values() if r["model_key"] == model_key and r["scenario_id"] == sid and r["condition"] == policy), None)
            lines.append("\nPOLICY: " + policy)
            if original:
                lines.extend(["EXACT ORIGINAL INSTRUCTION:", original["native_check"]["original_messages"][0]["content"],
                              "Saved original response ID: " + original["response_id"],
                              "Saved original labels: " + json.dumps(task["responses"][policy]["labels"], sort_keys=True)])
            if new is None:
                lines.append("No same-prefix continuation: " + str(task["prefix"].get("reason"))); continue
            prefix = new["prefix"]
            lines.extend(["EXACT SUPPLIED BASELINE PREFIX (repr preserves spaces):", repr(prefix["text"]),
                          "Prefix token IDs: " + json.dumps(prefix["input_token_ids"]),
                          f"Prefix chars [0,{prefix['end_char']}); tokens [0,{prefix['token_count']}); boundary predictor position {new['prefilled_input_length'] - 1}.",
                          "EXACT NEW CONTINUATION:", new["continuation"], "END NEW CONTINUATION",
                          "New response ID: " + new["response_id"],
                          "Single new-context review: " + json.dumps(reviews[new["response_id"]], ensure_ascii=False, sort_keys=True)])
        for policy in ("baseline", "evidence_contract"):
            original = natural.get((model_key, sid, policy))
            if not original:
                continue
            if policy != "baseline":
                lines.extend(["\nEXACT SAVED CONTRACT ANSWER (not the new continuation):", original["response"],
                              "Source: " + task["responses"][policy]["source_answer_file"] + ":" + str(task["responses"][policy]["source_answer_row"])])
            for event in original["events"]:
                lines.append(f"Natural {policy} {event['label']} chars[{event['start_char']}:{event['end_char']}]=" + repr(event["quoted_text"]) +
                             f"; answer tokens {event['first_answer_token_index']}..{event['last_answer_token_index']}.")
                display_block = 36 if model_key == "qwen3_8b" and event["label"] == "false_success" else _DA_BLOCKS[model_key]
                matching = [r for r in event_rows if r["record_type"] == "natural_event" and r["response_id"] == original["response_id"] and r["metric"] == event["label"]
                            and r["event_start_char"] == event["start_char"] and r["block"] == display_block]
                for row in matching:
                    lines.append(f"  {row['stage']}: j={row['answer_token_index']}, predictor position={row['predictor_position']}, "
                                 f"next-token text={row['target_text']!r}, p(saved token)={row['saved_next_token_probability']}; "
                                 f"cosine from pre-first-span-token state={row['cosine_from_event_onset']}.")
    return "\n".join(lines)



def during_publication_record(record):
    """Remove duplicate text already retained elsewhere; never shorten answers."""
    result = dict(record)
    native = dict(result["native_check"])
    native.pop("rendered_prompt", None)
    native.pop("effective_messages", None)
    native["complete_native_record_source"] = "Hash-verified external model.during.records.jsonl; original messages and native byte/token hashes retained here"
    result["native_check"] = native
    if "saved_answer_source" in result:
        source = result.pop("saved_answer_source")
        result["saved_answer_source_ref"] = {k: source[k] for k in ("source_answer_file", "source_answer_row", "source_label_file", "source_label_row", "response_id", "response_sha256")}
    if result["kind"] == "continuation":
        result.pop("response", None)  # identical complete text remains in full_assistant_text
    return result

def during_report(root, plan_path, cache_dir, reviews_path, output_prefix):
    """Report complete reconstructed states and once-scored new continuations."""
    import numpy as np
    root, plan_path, cache_dir = Path(root).resolve(), Path(plan_path).resolve(), Path(cache_dir).resolve()
    output_prefix = Path(output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = root / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text()); plan_hash = during_file_hash(plan_path)
    if set(plan["models"]) != set(_DA_BLOCKS):
        raise ValueError("Frozen report must contain exactly Qwen, Gemma, and Ministral")
    if plan.get("natural_event_plot_blocks") != _DA_BLOCKS or plan.get("natural_unsupported_success_extra_block") != {"qwen3_8b": 36}:
        raise ValueError("Natural plotting blocks differ from the pre-activation frozen choices")
    sources = state_read_frozen_sources(root, STATE_MODELS, STATE_PAIRS, STATE_METRICS, STATE_SEED, STATE_BOOTSTRAPS)
    for relative, expected in plan["source_sha256"].items():
        if sources["provenance"]["verified_frozen_source_sha256"].get(relative) != expected:
            raise ValueError("During-answer plan differs from the locked original sources: " + relative)
    for relative, expected in plan["prior_prompt_results_sha256"].items():
        if during_file_hash(root / relative) != expected:
            raise ValueError("Previous prompt-only results changed: " + relative)
    prior = json.loads((root / "results/internal_states.json").read_text())
    references = {}
    for model in prior["models"]:
        manifest = model["extraction_manifest"]
        shape = manifest["states_shape"]
        key = manifest["model_key"]
        references[key] = dict(sources["configuration"]["models"][key],
            checkpoint_metadata_sha256=manifest["frozen_method"]["checkpoint_metadata_sha256"],
            blocks=shape[1], hidden_size=shape[2])
    original_answers = {r["response_id"]: r for r, _ in sources["answers"].values()}
    manifests, natural, boundary, continuations, events, table, causal_checks = {}, {}, {}, {}, [], [], []
    for model_key, model_plan in plan["models"].items():
        tasks = {t["scenario_id"]: t for t in model_plan["selected_tasks"]}
        if sum(t["role"] == "case" for t in tasks.values()) > 10 or sum(t["role"] == "control" for t in tasks.values()) > 10:
            raise ValueError("Frozen model subset exceeds its 10 case / 10 control cap")
        for manifest, metadata, arrays in during_cache_records(cache_dir, model_key, plan_hash, references[model_key]):
            manifests[model_key] = manifest
            metadata = dict(metadata, model_key=model_key)
            if metadata["model"] != model_plan["repo_id"] or metadata["scenario_id"] not in tasks:
                raise ValueError("Cache record differs from frozen task/model selection")
            key = model_key, metadata["scenario_id"], metadata["condition"]
            if metadata["kind"] == "natural":
                source = original_answers[metadata["response_id"]]
                frozen = tasks[key[1]]["responses"][key[2]]
                if source["response"] != metadata["response"] or during_hash(metadata["response"]) != frozen["response_sha256"]:
                    raise ValueError("Natural replay differs from exact saved evaluated response")
                if list(arrays["target_token_ids"]) != metadata["answer_token_ids"] or list(arrays["predictor_positions"]) != metadata["predictor_positions"]:
                    raise ValueError("Natural state positions differ from published token metadata")
                expected = list(range(metadata["input_length"] - 1, metadata["input_length"] + len(metadata["answer_token_ids"]) - 1))
                if metadata["predictor_positions"] != expected:
                    raise ValueError("Teacher forcing uses an incorrect causal predictor index")
                natural[key] = metadata
                if key[2] == "baseline" and key in boundary:
                    boundary_metadata, boundary_states = boundary[key]
                    prefix_count = boundary_metadata["prefix"]["token_count"]
                    check = {"model_key": model_key, "scenario_id": key[1], "response_id": metadata["response_id"],
                             "prefix_token_count": prefix_count, "natural_predictor_answer_token_index": prefix_count}
                    if prefix_count < len(metadata["answer_token_ids"]):
                        if boundary_metadata["baseline_next_token_id"] != metadata["answer_token_ids"][prefix_count]:
                            raise ValueError("Baseline shared-prefix target does not match natural replay's exact next token")
                        if boundary_metadata["predictor_position"] != metadata["predictor_positions"][prefix_count]:
                            raise ValueError("Baseline boundary and natural predictor indices are off by one")
                        cosine, relative_l2 = during_geometry(arrays["states"][:, prefix_count, :], boundary_states)
                        probability_difference = abs(metadata["next_token_probabilities"][prefix_count] - boundary_metadata["baseline_next_token_probability"])
                        check.update(available=True, same_target_and_predictor_verified=True,
                                     maximum_block_cosine_distance=float(np.max(cosine)),
                                     maximum_block_relative_l2=float(np.max(relative_l2)),
                                     saved_next_token_probability_difference=probability_difference,
                                     interpretation="Same causal history under full-answer versus shorter-prefix forward; numerical differences are descriptive")
                    else:
                        check.update(available=False, reason="Prefix consumes all reconstructed visible answer tokens; historical Qwen/Gemma EOS unavailable")
                    causal_checks.append(check)
                rows, summaries = during_natural_rows(model_key, metadata, arrays)
                table.extend(rows); events.extend(summaries)
            elif metadata["kind"] == "boundary":
                if metadata["predictor_position"] != metadata["input_length"] + metadata["prefix"]["token_count"] - 1:
                    raise ValueError("Same-prefix boundary predictor index differs")
                boundary[key] = metadata, arrays["states"]
            elif metadata["kind"] == "continuation":
                text, prefix = metadata["full_assistant_text"], metadata["prefix"]["text"]
                if not text.startswith(prefix) or metadata["continuation"] != text[len(prefix):]:
                    raise ValueError("Continuation is not the exact remaining joint assistant decode")
                metadata["response"] = text
                continuations[metadata["response_id"]] = metadata
            else:
                raise ValueError("Unknown during-answer cache record type")
            del arrays
        expected_natural = {(model_key, t["scenario_id"], p) for t in tasks.values() for p in _DA_POLICIES if t["responses"][p].get("eligible", True)}
        if not expected_natural.issubset(natural):
            raise ValueError("Completed cache omits an eligible original policy answer")
        for sid in tasks:
            shared = [boundary[model_key, sid, p][0]["prefix"]["input_token_ids"] for p in _DA_POLICIES if (model_key, sid, p) in boundary]
            if shared and (len(shared) != 3 or any(ids != shared[0] for ids in shared)):
                raise ValueError("Prefix token IDs differ across original policies")
    reviews = during_review_index(reviews_path, continuations)
    for response_id, row in continuations.items():
        review = reviews[response_id]
        if (row.get("finish_reason") == "eos" and not row.get("truncated") and not row.get("technical_error")
                and during_known(review, "limitation_disclosed") and review["rubric_labels"]["limitation_disclosed"]):
            row["resolved_disclosure_span_origin"] = during_disclosure_origin(row, review)
        else:
            row["resolved_disclosure_span_origin"] = None
    identical_outputs, identical_label_disagreements = [], []
    for response_id, new in continuations.items():
        original_id = next(t for t in plan["models"][new["model_key"]]["selected_tasks"]
                           if t["scenario_id"] == new["scenario_id"])["responses"][new["condition"]]["response_id"]
        old = original_answers[original_id]
        same = old["response"] == new["response"]
        new["identical_to_saved_same_policy_answer"] = same
        if not same:
            continue
        identical_outputs.append({"new_response_id": response_id, "original_response_id": original_id,
            "model_key": new["model_key"], "condition": new["condition"], "response_sha256": during_hash(new["response"])})
        old_label = sources["labels"][original_id][0]
        for metric in _DA_METRICS:
            if during_known(old_label, metric) and during_known(reviews[response_id], metric) and old_label["rubric_labels"][metric] != reviews[response_id]["rubric_labels"][metric]:
                identical_label_disagreements.append({"new_response_id": response_id, "original_response_id": original_id,
                    "model_key": new["model_key"], "condition": new["condition"], "metric": metric,
                    "saved_label": old_label["rubric_labels"][metric], "new_label": reviews[response_id]["rubric_labels"][metric],
                    "status": "needs_review", "labels_changed": False,
                    "note": "Identical combined text received different saved/new AI judgments. This is not a change in reported behavior; both judgments remain intact and were not rescored."})
    groups = []
    for model_key, model_plan in plan["models"].items():
        model_boundary = {(sid, policy): value for (model, sid, policy), value in boundary.items() if model == model_key}
        rows, model_groups = during_boundary_rows(model_key, model_plan["selected_tasks"], model_boundary)
        table.extend(rows); groups.extend(model_groups)
        table.extend(during_outcome_rows(model_key, model_plan["selected_tasks"], continuations, reviews))
    for group in groups:
        task_lookup = {t["scenario_id"]: t for t in plan["models"][group["model_key"]]["selected_tasks"]}
        group["relevant_pair_original_joint_clean_controls"] = sum(all(
            all(t["responses"][p]["confidence"].get(m) in (2, 3)
                and m not in t["responses"][p]["unresolved_labels"] and not t["responses"][p]["labels"][m] for m in _DA_METRICS)
            for p in (group["source_policy"], group["target_policy"])) for t in [task_lookup[s] for s in group["control_ids"]])
        for row in table:
            if (row["record_type"] == "shared_prefix_boundary" and row["model_key"] == group["model_key"]
                    and row["source_policy"] == group["source_policy"] and row["target_policy"] == group["target_policy"]
                    and row["metric"] == group["metric"]):
                row["relevant_pair_original_joint_clean_controls"] = group["relevant_pair_original_joint_clean_controls"]
                row["control_selection_policies"] = "baseline,evidence_contract"
    paths = {name: output_prefix.with_suffix("." + suffix) for name, suffix in
             (("figure", "png"), ("table", "csv"), ("report", "txt"), ("records", "json"))}
    fields = sorted({field for row in table for field in row})
    with paths["table"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(table)
    during_figure(paths["figure"], table, groups)
    summary = ["The evidence contract did not remove every unsupported detail after a supplied answer beginning: Ministral still invented tool-capability explanations. The small matched state comparisons below are associations, not a false-reporting mechanism.",
               "This study reconstructs saved answers token by token, then supplies supported answer beginnings in a separate continuation test.",
               "No original answers were regenerated or rescored. All new continuations received one Codex-assisted context review; this is not human annotation. Nemotron was untouched.",
               "", f"Completed natural replay: {len(natural)} saved answers. Controlled continuations: {len(continuations)} new reports, one per eligible task/policy. These are separate from the original FTA results.",
               "Each natural answer token j uses block state at P+j−1, immediately before prediction; j=0 uses the final native input position. All blocks are captured before final normalization.",
               "Ministral reuses exact saved output IDs, including invisible EOS. Qwen/Gemma retokenize their exact visible saved texts; historical generated IDs/EOS were not saved. Batch1 reconstruction need not reproduce historical padded batch4 numerics.",
               "Next-token probabilities describe supplied words given their preceding text. They do not measure whether a statement is true.",
               "", "MAIN NEW-REPORT FINDINGS (selected case tasks only; positive / resolved)"]
    for model_key in plan["models"]:
        rows = [r for r in table if r["record_type"] == "new_continuation_outcome" and r["model_key"] == model_key and r["role"] == "case"]
        for metric in _DA_METRICS:
            values = [next(r for r in rows if r["condition"] == policy and r["metric"] == metric) for policy in _DA_POLICIES]
            summary.append(f"{model_key} {_DA_NAMES[metric]}, baseline / transparency / contract: " + " / ".join(f"{r['numerator']}/{r['denominator']} ({r['excluded_count']} unresolved/excluded)" for r in values))
    summary.extend(["Unresolved judgments are excluded for their own flag. Counts concern new supplied-prefix reports, not the full original FTA dataset. See the exact-text label-disagreement caveat below before interpreting changes.", "", "FIXED SUBSET AND SAME-PREFIX COUNTS"])
    for model_key, model_plan in plan["models"].items():
        selected = model_plan["selected_tasks"]
        summary.append(f"\n{model_plan['repo_id']}: {sum(t['role']=='case' for t in selected)} original error-disappeared tasks; {sum(t['role']=='control' for t in selected)} selected controls.")
        original_counts = model_plan["selection_counts"]
        for metric in _DA_METRICS:
            summary.append(f"Original {_DA_NAMES[metric]} pool: {original_counts['all_error_disappeared'][metric]} label-disappeared cases; {original_counts['all_unmatched_error_disappeared'][metric]} lacked any original strict-clean category × pressure control; {len(model_plan['matched_but_not_selected_ids'][metric])} matched cases omitted by the fixed subset cap. Exact omitted/unmatched IDs are preserved in the JSON plan.")
        for group in [g for g in groups if g["model_key"] == model_key and g["source_policy"] == "baseline" and g["target_policy"] == "evidence_contract"]:
            summary.append(f"{_DA_NAMES[group['metric']]}: {group['fixed_natural_cases']} fixed cases; {group['eligible_prefix_cases']} nonempty prefixes; {group['matched_prefix_cases']} matched to {group['matched_unique_controls']} unique controls; {len(group['prefix_unmatched_case_ids'])} prefix-eligible cases lost controls; {len(group['natural_only_case_ids'])} natural-only cases.")
            if group["prefix_unmatched_case_ids"]:
                summary.append("Unmatched after prefix filtering: " + ", ".join(group["prefix_unmatched_case_ids"]))
            entries = [r for r in table if r["record_type"] == "shared_prefix_boundary" and r["model_key"] == model_key and r["metric"] == group["metric"] and r["source_policy"] == "baseline" and r["target_policy"] == "evidence_contract"]
            if entries:
                largest = max(entries, key=lambda r: abs(r["cosine_case_minus_control"]))
                summary.append(f"Descriptive largest boundary contrast: block {largest['block']}, cosine case−control {largest['cosine_case_minus_control']:+.5f}, relative-L2 {largest['relative_l2_case_minus_control']:+.5f}. This selected peak is exploratory.")
            else:
                summary.append("No matched shared-prefix contrast is available for this flag.")
        for task in selected:
            realized = next((m for m in manifests[model_key]["prefixes"] if m["scenario_id"] == task["scenario_id"]), None)
            if realized and not realized.get("eligible"):
                summary.append(f"Natural-only {task['role']} {task['scenario_id']}: {realized.get('reason')}")
    summary.extend(["", "SECONDARY COMPARISONS: FIXED MAIN-PAIR COHORTS"])
    for group in groups:
        if (group["source_policy"], group["target_policy"]) != ("baseline", "evidence_contract"):
            summary.append(f"{group['model_key']} {group['source_policy']} → {group['target_policy']} {_DA_NAMES[group['metric']]}: {group['matched_prefix_cases']} matched cases / {group['matched_unique_controls']} unique controls; {group['relevant_pair_original_joint_clean_controls']} controls were originally jointly clean under this secondary pair. Main-pair selection is unchanged.")
    summary.extend(["", "NEW CONTINUATION OUTCOMES (single supplied-context review)"])
    for row in table:
        if row["record_type"] == "new_continuation_outcome" and row["metric"] in _DA_METRICS:
            summary.append(f"{row['model_key']} {row['role']} {row['condition']} {_DA_NAMES[row['metric']]}: {row['numerator']}/{row['denominator']} evaluated new reports; {row['excluded_count']} excluded truncated/error/unresolved reports.")
    summary.extend(["", "WHERE THE DISCLOSURE TEXT CAME FROM"])
    for model_key in plan["models"]:
        for policy in _DA_POLICIES:
            origin_rows = [r for r in table if r["record_type"] == "new_continuation_outcome"
                           and r["model_key"] == model_key and r["condition"] == policy and r["metric"] == "limitation_disclosed"]
            supplied = sum(r["disclosures_supplied_prefix_only"] for r in origin_rows)
            generated = sum(r["disclosures_generated_tail_only"] for r in origin_rows)
            mixed = sum(r["disclosures_mixed"] for r in origin_rows)
            total = sum(r["numerator"] for r in origin_rows)
            summary.append(f"{model_key} {policy}: {total} resolved positive disclosure judgments; supporting spans are wholly supplied prefix in {supplied}, wholly new tail in {generated}, and mixed in {mixed}. These describe the chosen supporting spans, not an additional honesty judgment.")
    summary.extend(["", "GREEDY REPLICATION AND LABEL DISAGREEMENTS",
        f"{len(identical_outputs)} combined prefix-plus-tail outputs exactly reproduce the saved same-policy answer. Greedy baseline continuations from their own original prefix can reproduce the original text; these are not independent answer samples.",
        f"{len(identical_label_disagreements)} resolved old/new judgments disagree despite identical text. These are marked needs review; neither label was changed and no additional scoring pass was run. Raw new-review rates above must not be interpreted as changed model behavior for those records.",
        "New review packets exposed model/policy fields and were not blinded. These are Codex-assisted exploratory judgments, not the authors' independent human annotations. Supplied-prefix disclosures are not newly generated honesty."])
    for conflict in identical_label_disagreements:
        summary.append(json.dumps(conflict, ensure_ascii=False, sort_keys=True))
    summary.extend(["", "HOW TO READ THE STATES", "An internal index check compares the baseline boundary against natural predictor j=prefix-token-count using only the already requested forwards. The target ID and predictor position must match; full-sequence versus shorter-prefix numerical distances are recorded without an extra probe.",
        "The same exact baseline-prefix IDs were appended to every policy's native assistant-start prompt, without closing the turn. Each control uses its own supported baseline beginning. No useful prefix was invented.",
        "Boundary contrasts subtract the mean available controls in each exact failure-category × pressure cell and weight cases equally. Missing controls are reported; no other-family substitute is used. Secondary pairs retain the original baseline-to-contract cohorts; their transparency answers are not automatically clean.",
        "A negative case−control contrast means the policy change moved the case states LESS than their matched clean controls; positive means MORE. This does not separate a reporting-error mechanism from ordinary instruction/format responses.",
        "Natural events use existing exact evidence spans. Before, onset, middle, last and after refer to that span's actual token positions, not a common token number across different answers. Local distances use the onset predictor, immediately BEFORE the first span token, as reference. It exists even for token0 at final input position. The earlier before-stage is missing when onset=0.",
        "The onset point is zero by definition because its own predictor state is the reference. The trough there is not evidence that a model noticed an unsupported claim.",
        "The right figure panels compare local progress through baseline invented-detail spans and contract failure admissions at PREVIOUS prompt-study blocks Qwen24/Gemma38/Ministral34. These are different words, meanings and preceding histories; the comparison is descriptive. The table also gives Qwen success spans at prior block36 and all three policies' annotated events.",
        "New rate denominators include ALL eligible selected case tasks for both flags, independent of which original flag selected them. Thus a new success claim in an originally details-only case still counts. New controls also receive fresh labels; their previous clean status is not assumed to persist.",
        "No claim state is fabricated for a policy where the unsupported span is absent. Natural claim/admission plots omit unresolved annotations and compare the same task IDs possessing both plotted event types. A disclosure span containing only BLOCKED/STATUS:BLOCKED is marked status_only; none occurs among the selected saved disclosure spans. Failure disclosures can coexist with unsupported claims.",
        "Singleton matched cases, especially Gemma details, are fragile descriptive examples. Missing Gemma success prefixes mean no shared-prefix success-layer contrast is available, even though natural answer replay remains available.",
        "Supplying a baseline answer beginning changes the original task, and may conflict with contract's structured reporting format. Identical answer histories reduce wording differences but leave instruction, length, position and format effects. No result establishes causation, awareness, concealment, or a repairable internal mechanism."])
    for model_key, model_plan in plan["models"].items():
        summary.append(during_example_text(model_key, model_plan, natural, continuations, reviews, table))
    summary.extend(["\nSOURCE APPENDIX", "Frozen selection: " + str(plan_path) + "; SHA256=" + plan_hash,
                    "New reviews: " + str(Path(reviews_path).resolve()) + "; SHA256=" + during_file_hash(reviews_path),
                    "All exact prefixes, new full continuations, existing source IDs, tokens/offsets/probabilities, annotations, group/exclusion IDs and cache manifests are in " + paths["records"].name + ". Block vectors and weights remain outside GitHub."])
    paths["report"].write_text("\n".join(summary) + "\n")
    publication = {"phase": plan["phase"], "original_FTA_results_changed": False,
        "plan": plan, "plan_sha256": plan_hash, "cache_manifests": manifests, "groups": groups,
        "natural_records": [during_publication_record(r) for r in natural.values()], "natural_event_summaries": events, "causal_reconstruction_checks": causal_checks,
        "controlled_continuations": [during_publication_record(r) for r in continuations.values()], "new_reviews": list(reviews.values()),
        "identical_same_policy_outputs": identical_outputs,
        "identical_text_label_disagreements_needing_review": identical_label_disagreements,
        "analysis": {"natural_plot_blocks_from_previous_study": _DA_BLOCKS,
            "natural_table_extra_qwen_success_block": 36, "natural_event_stages": ["before", "onset", "middle", "last", "after"],
            "natural_local_reference": "State at answer index onset, before first span token; onset0 uses final input position",
            "natural_alignment": "Paired task IDs with first baseline detail span and first contract admission; local reference is predictor immediately before first span token. Differing words/prior history, descriptive only",
            "boundary_metric": "cosine distance and ||target-source||/||source||, within-model exact-cell control contrasts",
            "continuations_reviewed_once": True, "prior_continuation_labels_inherited": False,
            "disclosure_origin": "Mechanical exact-span offsets against supplied-prefix length; reviewer decisions and bytes preserved",
            "new_rate_eligibility": "EOS, not truncated/error, applicable confidence2/3 and not unresolved",
            "causation_established": False, "nemotron_touched": False},
        "validated_source_provenance": sources["provenance"],
        "reporting_pipeline_sha256": during_file_hash(Path(__file__).resolve()),
        "compact_outputs": {name: {"path": str(path.relative_to(root)), "sha256": during_file_hash(path), "bytes": path.stat().st_size}
                             for name, path in paths.items() if name != "records"}}
    paths["records"].write_text(json.dumps(publication, ensure_ascii=False, separators=(",", ":")) + "\n")
    total_bytes = sum(path.stat().st_size for path in paths.values())
    if total_bytes >= 10 * 1024**2:
        raise ValueError("Compact publication outputs unexpectedly exceed 10 MiB")
    result = {"natural_answers": len(natural), "controlled_continuations": len(continuations),
              "files": {name: str(path) for name, path in paths.items()}, "compact_bytes": total_bytes}
    print(json.dumps(result), flush=True)
    return result



def during_packets(root, plan_path, cache_dir, output, model_name):
    """Export only completed NEW continuations for one contextual review each."""
    root, cache_dir, output = Path(root).resolve(), Path(cache_dir).resolve(), Path(output)
    spec = model_spec(root, model_name)
    plan_hash = sha256_file(plan_path)
    manifest = read_json(cache_dir / (spec["key"] + ".during.manifest.json"))
    if manifest["status"] != "complete" or manifest["method"]["plan_sha256"] != plan_hash:
        raise ValueError("Review packets require a completed frozen model study")
    records_path = cache_dir / (spec["key"] + ".during.records.jsonl")
    if sha256_file(records_path) != manifest["files"]["records"]["sha256"]:
        raise ValueError("Completed continuation metadata checksum differs")
    packets = []
    for row in read_jsonl(records_path):
        if row["kind"] != "continuation":
            continue
        packets.append({"response_id": row["response_id"], "model": row["model"],
            "scenario_id": row["scenario_id"], "condition": row["condition"],
            "system": row["native_check"]["original_messages"][0]["content"],
            "user": row["native_check"]["original_messages"][1]["content"],
            "prefix": row["prefix"]["text"], "continuation": row["continuation"],
            "response": row["full_assistant_text"], "response_sha256": digest(row["full_assistant_text"]),
            "finish_reason": row["finish_reason"], "truncated": row["truncated"],
            "original_FTA_labels_inherited": False, "plan_sha256": plan_hash,
            "source_file": str(records_path), "source_record_sha256": digest(row)})
    if len(packets) != manifest["controlled_continuations"]:
        raise ValueError("Completed new continuation count differs")
    if output.exists():
        if read_jsonl(output) != packets:
            raise ValueError("Existing review packets are preserved and differ")
    else:
        atomic_write_jsonl(output, packets, overwrite=False)
    print(json.dumps({"model": spec["key"], "new_review_packets": len(packets), "output": str(output)}), flush=True)
    return packets


PCA_POLICIES = ("baseline", "transparency", "evidence_contract")
PCA_MODELS = ("ministral3_8b", "qwen3_8b", "gemma2_9b")
PCA_SEQUENCES = ("W-W-W", "W-W-C", "W-C-W", "W-C-C", "C-W-W", "C-W-C", "C-C-W", "C-C-C")


def pca_triplets(sources, scenarios):
    """Join all three original policies; use saved label resolution only."""
    selected, exclusions, counts = {}, {}, {}
    for key in PCA_MODELS:
        model = sources["configuration"]["models"][key]["repo_id"]
        rows, missing = [], []
        for task_id in sorted(scenarios):
            endpoints, reasons = [], []
            for policy in PCA_POLICIES:
                answer, answer_line = sources["answers"][(model, task_id, policy)]
                label, label_line = sources["labels"][answer["response_id"]]
                if not _state_valid(answer):
                    reasons.append({"policy": policy, "reason": "incomplete_or_invalid_answer",
                        "truncated": bool(answer.get("truncated")), "finish_reason": answer.get("finish_reason"),
                        "answer_source": "results/answers.jsonl:" + str(answer_line)})
                for flag in STATE_METRICS[:2]:
                    if not _state_known(label, flag):
                        reasons.append({"policy": policy, "reason": "unresolved_error_label", "flag": flag,
                            "confidence": label.get("confidence", {}).get(flag),
                            "label_source": "results/labels.jsonl:" + str(label_line)})
                endpoints.append((policy, answer, label, answer_line, label_line))
            if reasons:
                missing.append({"task_id": task_id, "reasons": reasons})
                continue
            outcomes = ["W" if any(label["rubric_labels"][f] for f in STATE_METRICS[:2]) else "C"
                        for _, _, label, _, _ in endpoints]
            sequence = "-".join(outcomes)
            for outcome, (policy, answer, label, answer_line, label_line) in zip(outcomes, endpoints):
                rows.append({"model_key": key, "model": model, "task_id": task_id, "policy": policy,
                    "response_id": answer["response_id"], "revision": answer["revision"],
                    "outcome": outcome, "sequence": sequence,
                    "failure_category": scenarios[task_id]["category"],
                    "pressure_type": scenarios[task_id]["pressure_type"],
                    "false_success": label["rubric_labels"]["false_success"],
                    "fabricated_details": label["rubric_labels"]["fabricated_details"],
                    "false_success_confidence": label["confidence"]["false_success"],
                    "fabricated_details_confidence": label["confidence"]["fabricated_details"],
                    "false_success_uncertainty_note": label.get("uncertainty", {}).get("false_success", {}),
                    "fabricated_details_uncertainty_note": label.get("uncertainty", {}).get("fabricated_details", {}),
                    "input_token_count": answer["input_length"],
                    "answer_source": "results/answers.jsonl:" + str(answer_line),
                    "label_source": "results/labels.jsonl:" + str(label_line)})
        task_sequences = {r["task_id"]: r["sequence"] for r in rows}
        selected[key], exclusions[key] = rows, missing
        counts[key] = {s: sum(value == s for value in task_sequences.values()) for s in PCA_SEQUENCES}
        if len(task_sequences) + len(missing) != 100 or len(rows) != 3 * len(task_sequences):
            raise ValueError("PCA eligibility must partition all 100 original task triplets")
    return selected, exclusions, counts


def pca_fit_final_vectors(vectors):
    """The only PCA fit: L2 rows, ordinary centered full SVD, three PCs."""
    import numpy as np
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("PCA needs finite nonzero final-block prompt vectors")
    normalized = values / norms[:, None]
    with threadpool_limits(limits=1):
        pca = PCA(n_components=3, svd_solver="full", whiten=False)
        coordinates = pca.fit_transform(normalized)
    if not np.allclose(coordinates, (normalized-pca.mean_) @ pca.components_.T, rtol=1e-10, atol=1e-12):
        raise ValueError("PCA coordinates do not match the single pooled centered projection")
    return {"coordinates": coordinates, "components": pca.components_, "mean": pca.mean_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance": pca.explained_variance_, "singular_values": pca.singular_values_}


def pca_project_cached(vectors, method, directory, model_key):
    """Keep fitted components externally so presentation edits never refit PCA."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / (model_key + ".pca.npz")
    meta_path = directory / (model_key + ".pca.json")
    signature = digest(method)
    if meta_path.exists():
        metadata = read_json(meta_path)
        if metadata["method_sha256"] != signature or sha256_file(data_path) != metadata["data_sha256"]:
            raise ValueError("Existing PCA projection differs; preserve it instead of fitting again")
        with np.load(data_path, allow_pickle=False) as archive:
            projected = {name: archive[name].copy() for name in archive.files}
        return projected, metadata, False
    if data_path.exists():
        raise FileExistsError("Unsigned PCA projection exists; preserve it rather than refit")
    projected = pca_fit_final_vectors(vectors)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **projected)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, data_path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    metadata = {"method": method, "method_sha256": signature,
                "data_sha256": sha256_file(data_path), "pca_fits": 1,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_write_json(meta_path, metadata, overwrite=False)
    return projected, metadata, True


def pca_static_figure(payload, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"baseline": "#4c78a8", "transparency": "#1b9e77", "evidence_contract": "#e69f00"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        figure = plt.figure(figsize=(15, 5.2))
        for index, model in enumerate(payload["models"]):
            ax = figure.add_subplot(1, 3, index+1, projection="3d")
            tasks = {}
            for point in model["points"]:
                tasks.setdefault(point["task_id"], {})[point["policy"]] = point
            for task in tasks.values():
                a = task["baseline"]
                for policy in PCA_POLICIES[1:]:
                    b = task[policy]
                    ax.quiver(a["pc1"], a["pc2"], a["pc3"], b["pc1"]-a["pc1"],
                        b["pc2"]-a["pc2"], b["pc3"]-a["pc3"], color=colors[policy],
                        alpha=.12, linewidth=.35, arrow_length_ratio=.04)
            for policy in PCA_POLICIES:
                for outcome, marker in (("C", "o"), ("W", "D")):
                    points = [p for p in model["points"] if p["policy"] == policy and p["outcome"] == outcome]
                    ax.scatter([p["pc1"] for p in points], [p["pc2"] for p in points],
                        [p["pc3"] for p in points], marker=marker, s=15, alpha=.72,
                        color=colors[policy], linewidths=.2, edgecolors="white", depthshade=False)
            labels = [f"PC{i+1} ({100*v:.1f}%)" for i, v in enumerate(model["explained_variance_ratio"])]
            ax.set(xlabel=labels[0], ylabel=labels[1], zlabel="",
                   xlim=model["axis_range"], ylim=model["axis_range"], zlim=model["axis_range"])
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=22, azim=40)
            ax.tick_params(labelsize=7, pad=0)
            ax.set_title(model["name"] + f" · 3 PCs: {100*sum(model['explained_variance_ratio']):.1f}%", fontsize=12, pad=8)
        handles = [Line2D([], [], marker="o", ls="", color=color, label=policy.replace("_", " ").capitalize())
                   for policy, color in colors.items()]
        handles += [Line2D([], [], marker="D", ls="", color="#4b5563", label="W: either error"),
                    Line2D([], [], marker="o", ls="", color="#4b5563", label="C: neither flag")]
        figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(.5, .99))
        figure.subplots_adjust(left=.06, right=.97, bottom=.08, top=.83, wspace=.24)
        for ax, model in zip(figure.axes, payload["models"]):
            position = ax.get_position()
            figure.text(position.x0-.035, (position.y0+position.y1)/2,
                f"PC3 ({100*model['explained_variance_ratio'][2]:.1f}%)",
                rotation=90, va="center", ha="center", fontsize=9)
        figure.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def pca_plain_report(payload, manifest, output):
    lines = ["ORIGINAL THREE-POLICY FTA: BEFORE-ANSWER PCA", "",
        "These maps show the original cached prompt states, labeled by existing saved answer outcomes. They use no factorial prompts or supplied assistant prefixes.",
        "W means unsupported success OR invented details. An answer with both flags counts once. C means neither flag; it does not guarantee complete correctness.",
        "Only tasks with all three original-policy answers complete and both flags resolved under every policy were included. Confidence 1 or explicitly unresolved flags exclude the entire task triplet. Confidence 2/3 judgments retain any saved minor-uncertainty notes; those notes were not re-reviewed.",
        "Sequence order is baseline → transparency → evidence contract. These are the same task IDs throughout, but W can refer to different erroneous statements under different policies.", "",
        "One PCA per model uses ALL eligible triplets pooled across the three policies. Final-block vectors are L2-normalized individually, then ordinary centered full-SVD PCA keeps three components. No whitening, coordinate scaling, control subtraction, policy-mean subtraction, layer search, or subgroup PCA.",
        "States precede every answer token, at the final native input token and final transformer block before final normalization. They were reconstructed in the earlier prompt-only study; they are not activations recorded while the old answers were generated.", "",
        "Sequence counts (tasks)", "Sequence | " + " | ".join(m["name"] for m in payload["models"])]
    for sequence in PCA_SEQUENCES:
        lines.append(sequence + " | " + " | ".join(str(m["sequence_counts"][sequence]) for m in payload["models"]))
    lines += ["", "Eligibility and explained variance"]
    for model in payload["models"]:
        ratios = model["explained_variance_ratio"]
        lines.append(f"{model['name']}: {len(model['points'])//3}/100 tasks; {len(model['points'])} points; final block {model['final_block']}. PC1/PC2/PC3 explain " +
            ", ".join(f"{100*v:.2f}%" for v in ratios) + f"; sum {100*sum(ratios):.2f}%.")
        lines.append(f"  Fixed equal-unit cube range on each PC axis: {model['axis_range']}. Limits do not change with filters.")
        for excluded in manifest["exclusions"][model["id"]]:
            reasons = "; ".join(r["policy"] + ": " + r["reason"] + (" (" + r["flag"] + ")" if "flag" in r else "") for r in excluded["reasons"])
            lines.append("  Excluded " + excluded["task_id"] + " — " + reasons)
    lines += ["", "Reading the maps",
        "The original all-task static overview (pca.png) is preserved: colors are policies, diamonds are W and circles are C. In the focused figure (pca_focused.png) and updated interactive HTML, shape instead identifies a FIXED TASK GROUP: triangles are W-W-C under every policy, including their corrected C contract points; circles are C-C-C, an always-clean reference group, not automatically matched controls.",
        "The interactive page defaults to W-W-C + C-C-C. Either group can be hidden, and other observed sequences remain accessible separately. All filters retain original triplets, coordinates and fixed axis limits; they do not refit PCA or repeat normalization. Solid arrows mark W-W-C and faint dashed arrows mark C-C-C. Arrows join the same task baseline→transparency and baseline→contract, showing instruction differences rather than an answer unfolding through time.",
        "Within each model all geometric axes have the same scale. Each model has its own independently fitted PCA: never compare coordinate values or distances between models.",
        "PCA distances are projected approximations of normalized-state geometry. The displayed percentages report how much variance these three dimensions retain. Neither separation nor overlap establishes a hallucination mechanism, awareness, intent, or causation.",
        "No generations, new rubric judgments, classifiers, interventions, significance tests, or Nemotron work were performed. The original answers, labels, figures, and state caches were preserved.", "",
        "Focused fixed-group task counts (W-W-C / C-C-C): " + "; ".join(m["name"] + f" {m['sequence_counts']['W-W-C']} / {m['sequence_counts']['C-C-C']}" for m in payload["models"]) + ". Empty groups are never replaced by another sequence.",
        "Before-answer states labeled by saved answers; projected distances are approximate, and separation does not establish a cause of false reporting. No generation, rescoring, classifiers, or interventions.",
        "Files: pca.html is self-contained and opens offline; pca_focused.png shows the fixed two-group comparison and pca_focused.json records the presentation-only update. The unchanged pca.png is the original static all-points overview; unchanged pca.csv contains every original task/policy coordinate and both flags. pca.json records original fits and the current presentation index. config/pca.json retains the original settings and eligible IDs. Activation arrays and fitted PCA components remain outside GitHub."]
    Path(output).write_text("\n".join(lines)+"\n", encoding="utf-8")


def pca_visualize(root, cache_dir, projection_cache_dir, output_prefix):
    """Original-run visualization only: cached final states, fixed pooled PCA."""
    from utils.dataset import load_scenarios

    root, cache_dir = Path(root).resolve(), Path(cache_dir).expanduser().resolve()
    projection_cache_dir = Path(projection_cache_dir).expanduser().resolve()
    output_prefix = Path(output_prefix).resolve()
    if root == projection_cache_dir or root in projection_cache_dir.parents:
        raise ValueError("Fitted PCA arrays must remain outside the project")
    if root not in output_prefix.parents:
        raise ValueError("Compact PCA outputs must be inside this project")
    sources = state_read_frozen_sources(root, STATE_MODELS, STATE_PAIRS, STATE_METRICS, STATE_SEED, STATE_BOOTSTRAPS)
    scenarios = load_scenarios(root)
    selected, exclusions, counts = pca_triplets(sources, scenarios)
    preserved = {str(p.relative_to(root)): sha256_file(p) for p in (root/"results").rglob("*") if p.is_file()
                 and p.suffix in (".json", ".jsonl", ".csv", ".txt", ".png", ".html")
                 and p.stem != output_prefix.name}
    fit_packages = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}
    settings = {"phase": "original_three_policy_prompt_state_PCA_visualization_only", "policy_order": list(PCA_POLICIES),
        "model_order": list(PCA_MODELS), "flags": list(STATE_METRICS[:2]),
        "eligibility": "Complete EOS triplet; both error flags confidence2/3 and not explicitly unresolved under every policy",
        "W": "false_success OR fabricated_details", "C": "Neither saved flag; not a guarantee of complete correctness",
        "selection": "All eligible original task triplets, independent of outcome sequence or prior matching",
        "state": "Cached final transformer-block output at final input token before final normalization; no answer tokens",
        "expected_final_blocks": {"ministral3_8b": 34, "qwen3_8b": 36, "gemma2_9b": 42},
        "PCA": {"normalize": "individual vector L2, float64", "center": True, "components": 3,
            "svd_solver": "full", "whiten": False, "coordinate_scaling": False, "pool": "all eligible tasks x all three policies per model",
            "control_or_policy_mean_subtraction": False},
        "fit_function_sha256": digest(inspect.getsource(pca_fit_final_vectors)), "fit_packages": fit_packages,
        "source_provenance": sources["provenance"],
        "task_ids": {key: sorted({r["task_id"] for r in rows}) for key, rows in selected.items()},
        "sequence_counts": counts, "exclusions": exclusions,
        "new_generations": 0, "new_scoring": 0, "classifiers": False, "interventions": False,
        "significance_tests": False, "Nemotron_touched": False, "stop_after_visualization": True}
    plan_path = root/"config/pca.json"
    if plan_path.exists():
        if read_json(plan_path) != settings:
            raise ValueError("Frozen PCA inputs/settings differ; preserve previous visualization")
    else:
        atomic_write_json(plan_path, settings, overwrite=False)
    payload, fit_metadata, new_fits = {"models": []}, {}, 0
    names = {"ministral3_8b": "Ministral", "qwen3_8b": "Qwen", "gemma2_9b": "Gemma"}
    table = []
    for key in PCA_MODELS:
        validated = state_validate_cache(root, cache_dir, key, sources,
            (extraction_reference, verify_extraction_inputs, extraction_gpu, verify_complete_cache, extract), EXTRACTION_VERSION)
        positions = {(str(task), str(policy)): i for i, (task, policy) in enumerate(zip(validated["scenario_ids"], validated["conditions"]))}
        records = selected[key]
        block = validated["states"].shape[1]
        if block != settings["expected_final_blocks"][key]:
            raise ValueError("Unexpected final transformer block")
        indices = [positions[(r["task_id"], r["policy"])] for r in records]
        vectors = validated["states"][indices, -1, :].astype(np.float64)
        method = {"settings": settings["PCA"], "function_sha256": settings["fit_function_sha256"],
            "packages": fit_packages, "model": key, "revision": records[0]["revision"], "final_block": block,
            "state_file_sha256": validated["provenance"]["validated_files"]["states"]["sha256"],
            "ordered_input_ids": [[r["task_id"], r["policy"]] for r in records],
            "selected_vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest()}
        projected, metadata, fitted = pca_project_cached(vectors, method, projection_cache_dir, key)
        new_fits += int(fitted)
        coordinates = projected["coordinates"]
        if coordinates.shape != (len(records), 3) or not np.isfinite(coordinates).all():
            raise ValueError("PCA coordinate coverage differs from complete triplets")
        radius = float(np.max(np.abs(coordinates)) * 1.08)
        points = []
        for record, xyz, position in zip(records, coordinates, indices):
            point = dict(record, pc1=float(xyz[0]), pc2=float(xyz[1]), pc3=float(xyz[2]))
            points.append(point)
            table.append({k: v for k, v in dict(point, final_block=block, native_input_index=position).items()
                          if not k.endswith("_uncertainty_note")})
        payload["models"].append({"id": key, "name": names[key], "model": records[0]["model"],
            "points": points, "axis_range": [-radius, radius], "final_block": block,
            "explained_variance_ratio": projected["explained_variance_ratio"].tolist(), "sequence_counts": counts[key]})
        fit_metadata[key] = {"projection": metadata, "cache_validation": validated["provenance"],
            "axis_range": [-radius, radius], "explained_variance_ratio": projected["explained_variance_ratio"].tolist(),
            "explained_variance_sum": float(projected["explained_variance_ratio"].sum()),
            "eligible_tasks": len(records)//3, "points": len(records), "final_block": block,
            "reconstructed_not_historical_activation_capture": True}
        del validated, vectors
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {ext: output_prefix.with_suffix("."+ext) for ext in ("html", "png", "csv", "txt", "json")}
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    distribution = importlib.metadata.distribution("plotly")
    js_path = Path(distribution.locate_file("plotly/package_data/plotly.min.js"))
    plotly_js = js_path.read_text()
    render_pca_html(payload, plotly_js, paths["html"])
    pca_static_figure(payload, paths["png"])
    manifest = {"phase": settings["phase"], "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan": "config/pca.json", "plan_sha256": sha256_file(plan_path),
        "pipeline_sha256": sha256_file(Path(__file__)), "source_provenance": sources["provenance"],
        "models": fit_metadata, "sequence_counts": counts, "exclusions": exclusions,
        "plotly_python_package": distribution.version, "bundled_plotly_js_sha256": sha256_file(js_path),
        "matplotlib_version": importlib.metadata.version("matplotlib"), "python_version": platform.python_version(),
        "total_pca_fits": 3, "pca_fits_this_invocation": new_fits,
        "new_model_forwards": 0, "new_answers": 0, "new_scoring": 0, "classifiers": False,
        "interventions": False, "significance_tests": False, "Nemotron_touched": False,
        "before_answer_states": True, "arrows": ["baseline→transparency", "baseline→evidence_contract"],
        "fixed_projection_and_axis_limits_under_filters": True, "stop_after_visualization": True}
    pca_plain_report(payload, manifest, paths["txt"])
    for relative, expected in preserved.items():
        if sha256_file(root/relative) != expected:
            raise ValueError("A preserved result changed during PCA visualization: " + relative)
    manifest["preserved_results_verified"] = preserved
    manifest["outputs"] = {ext: {"file": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                           for ext, path in paths.items() if ext != "json"}
    atomic_write_json(paths["json"], manifest)
    return {"phase": settings["phase"], "points": len(table), "new_pca_fits": new_fits,
            "sequence_counts": counts, "explained_variance_sums": {k: m["explained_variance_sum"] for k, m in fit_metadata.items()},
            "outputs": manifest["outputs"]}



def pca_focused_figure(payload, output, *, large_times=False):
    """Draw two fixed task groups at their unchanged full-data PCA coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"baseline": "#4c78a8", "transparency": "#1b9e77", "evidence_contract": "#e69f00"}
    groups = (("C-C-C", "o", "--", .13), ("W-W-C", "^", "-", .52))
    with plt.rc_context({"font.family": "Times New Roman" if large_times else "DejaVu Sans",
                         "font.size": 15 if large_times else 9}):
        figure = plt.figure(figsize=(16.6, 6.2) if large_times else (15, 5.2))
        for index, model in enumerate(payload["models"]):
            ax = figure.add_subplot(1, 3, index+1, projection="3d")
            tasks = {}
            for point in model["points"]:
                if point["sequence"] in ("W-W-C", "C-C-C"):
                    tasks.setdefault(point["task_id"], {})[point["policy"]] = point
            for sequence, marker, style, alpha in groups:
                group_tasks = [t for t in tasks.values() if t["baseline"]["sequence"] == sequence]
                for task in group_tasks:
                    a = task["baseline"]
                    for policy in PCA_POLICIES[1:]:
                        b = task[policy]
                        ax.quiver(a["pc1"], a["pc2"], a["pc3"], b["pc1"]-a["pc1"],
                            b["pc2"]-a["pc2"], b["pc3"]-a["pc3"], color=colors[policy],
                            alpha=alpha, linewidth=.45, linestyle=style, arrow_length_ratio=.035)
                for policy in PCA_POLICIES:
                    points = [t[policy] for t in group_tasks]
                    ax.scatter([p["pc1"] for p in points], [p["pc2"] for p in points],
                        [p["pc3"] for p in points], marker=marker,
                        s=27 if sequence == "W-W-C" else 15,
                        alpha=.96 if sequence == "W-W-C" else .65,
                        color=colors[policy], linewidths=.25, edgecolors="white", depthshade=False)
            labels = [f"PC{i+1} ({100*v:.1f}%)" for i, v in enumerate(model["explained_variance_ratio"])]
            ax.set(xlabel=labels[0], ylabel=labels[1], zlabel="",
                   xlim=model["axis_range"], ylim=model["axis_range"], zlim=model["axis_range"])
            if large_times:
                ax.xaxis.label.set_size(17)
                ax.yaxis.label.set_size(17)
                ax.xaxis.labelpad = 9
                ax.yaxis.labelpad = 9
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=22, azim=40)
            ax.tick_params(labelsize=15 if large_times else 7, pad=0)
            ax.set_title(model["name"] + f" · 3 PCs: {100*sum(model['explained_variance_ratio']):.1f}%", fontsize=19 if large_times else 11, pad=7)
        handles = [Line2D([], [], color=color, lw=3, label=policy.replace("_", " ").capitalize())
                   for policy, color in colors.items()]
        handles += [Line2D([], [], marker="^", ls="-", color="#4b5563",
                          label="W-W-C"),
                    Line2D([], [], marker="o", ls="--", color="#88919a",
                          label="C-C-C")]
        figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
                      fontsize=16 if large_times else 10, bbox_to_anchor=(.5, .99))
        figure.subplots_adjust(left=.06, right=.97, bottom=.13 if large_times else .08,
                               top=.83, wspace=.30 if large_times else .24)
        for ax, model in zip(figure.axes, payload["models"]):
            position = ax.get_position()
            figure.text(position.x0-.035, (position.y0+position.y1)/2,
                f"PC3 ({100*model['explained_variance_ratio'][2]:.1f}%)",
                rotation=90, va="center", ha="center", fontsize=17 if large_times else 9)
        figure.savefig(output, dpi=190, bbox_inches="tight", facecolor="white",
                       **({"pad_inches": .4} if large_times else {}))
        plt.close(figure)


def pca_focus(root, source_prefix, output_prefix):
    """Presentation only: no activation/projection caches, normalization or PCA."""
    root, source_prefix, output_prefix = Path(root).resolve(), Path(source_prefix).resolve(), Path(output_prefix).resolve()
    if root not in source_prefix.parents or root not in output_prefix.parents:
        raise ValueError("PCA presentation files must stay within the current project")
    source_manifest = source_prefix.with_suffix(".json")
    manifest_before_sha = sha256_file(source_manifest)
    manifest = read_json(source_manifest)
    plan = read_json(root/"config/pca.json")
    preserved = {source_prefix.with_suffix(".csv"): manifest["outputs"]["csv"]["sha256"],
                 source_prefix.with_suffix(".png"): manifest["outputs"]["png"]["sha256"],
                 root/"config/pca.json": manifest["plan_sha256"]}
    for path, expected in preserved.items():
        if sha256_file(path) != expected:
            raise ValueError("Original PCA geometry/overview changed: " + str(path))
    if manifest["phase"] != "original_three_policy_prompt_state_PCA_visualization_only":
        raise ValueError("Only the original three-policy PCA may be restyled")
    with source_prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = {"ministral3_8b": "Ministral", "qwen3_8b": "Qwen", "gemma2_9b": "Gemma"}
    payload, selected_ids = {"models": []}, {}
    for key in plan["model_order"]:
        model_meta = manifest["models"][key]
        points, tasks = [], {}
        for row in rows:
            if row["model_key"] != key:
                continue
            point = dict(row)
            for field in ("false_success", "fabricated_details"):
                if point[field].lower() not in ("true", "false"):
                    raise ValueError("Saved combined-error flags must remain Boolean")
                point[field] = point[field].lower() == "true"
            for field in ("pc1", "pc2", "pc3"):
                point[field] = float(point[field])
            for field in ("false_success_confidence", "fabricated_details_confidence", "input_token_count", "final_block", "native_input_index"):
                point[field] = int(point[field])
            if point["outcome"] != ("W" if point["false_success"] or point["fabricated_details"] else "C"):
                raise ValueError("Original outcome/flags do not agree")
            if point["policy"] in tasks.setdefault(point["task_id"], {}):
                raise ValueError("Duplicate saved task-policy point")
            tasks[point["task_id"]][point["policy"]] = point
            points.append(point)
        for task in tasks.values():
            if set(task) != set(PCA_POLICIES):
                raise ValueError("Only complete original triplets may be displayed")
            sequence = "-".join(task[p]["outcome"] for p in PCA_POLICIES)
            if any(p["sequence"] != sequence for p in task.values()):
                raise ValueError("Fixed task-group sequence differs from saved outcomes")
        counts = {sequence: sum(t["baseline"]["sequence"] == sequence for t in tasks.values()) for sequence in PCA_SEQUENCES}
        if counts != manifest["sequence_counts"][key] or len(points) != model_meta["points"]:
            raise ValueError("Presentation must retain every originally eligible point")
        payload["models"].append({"id": key, "name": names[key], "model": points[0]["model"],
            "points": points, "axis_range": model_meta["axis_range"], "final_block": model_meta["final_block"],
            "explained_variance_ratio": model_meta["explained_variance_ratio"], "sequence_counts": counts})
        selected_ids[key] = {sequence: sorted(tid for tid,t in tasks.items() if t["baseline"]["sequence"] == sequence)
                            for sequence in ("W-W-C", "C-C-C")}
    html = source_prefix.with_suffix(".html")
    previous_html_sha = sha256_file(html)
    if previous_html_sha != manifest["outputs"]["html"]["sha256"]:
        raise ValueError("Existing interactive HTML differs from its saved metadata")
    distribution = importlib.metadata.distribution("plotly")
    js_path = Path(distribution.locate_file("plotly/package_data/plotly.min.js"))
    if sha256_file(js_path) != manifest["bundled_plotly_js_sha256"]:
        raise ValueError("Use the original embedded Plotly version for this visual update")
    render_pca_html(payload, js_path.read_text(), html)
    figure_path = output_prefix.with_suffix(".png")
    pca_focused_figure(payload, figure_path)
    pca_plain_report(payload, manifest, source_prefix.with_suffix(".txt"))
    for path, expected in preserved.items():
        if sha256_file(path) != expected:
            raise ValueError("The original PCA coordinates/overview/settings were modified")
    presentation = {"phase": "PCA_fixed_group_presentation_only", "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_PCA_manifest_sha256_before_update": manifest_before_sha,
        "source_coordinates": str(source_prefix.with_suffix(".csv").relative_to(root)),
        "source_coordinates_sha256": preserved[source_prefix.with_suffix(".csv")],
        "original_all_task_overview": str(source_prefix.with_suffix(".png").relative_to(root)),
        "original_all_task_overview_sha256": preserved[source_prefix.with_suffix(".png")],
        "original_PCA_plan_sha256": manifest["plan_sha256"],
        "original_fit_pipeline_sha256": manifest["pipeline_sha256"],
        "presentation_pipeline_sha256": sha256_file(Path(__file__)),
        "default_sequences": ["W-W-C", "C-C-C"], "selected_task_ids": selected_ids,
        "marker_semantics": {"W-W-C": "fixed-group triangle under all policies; contract triangles have outcome C",
                             "C-C-C": "always-clean reference circles; not automatically matched controls"},
        "line_semantics": {"W-W-C": "solid", "C-C-C": "faint dashed"},
        "triangles_in_HTML": "Centered triangle glyphs with tiny hover anchors at exact saved coordinates; no jitter or displacement",
        "models": {m["id"]: {"axis_range": m["axis_range"], "explained_variance_ratio": m["explained_variance_ratio"],
            "final_block": m["final_block"], "all_eligible_points": len(m["points"]),
            "focused_group_tasks": {s: m["sequence_counts"][s] for s in ("W-W-C", "C-C-C")},
            "focused_points": 3*sum(m["sequence_counts"][s] for s in ("W-W-C", "C-C-C"))} for m in payload["models"]},
        "pca_refits": 0, "activation_or_projection_cache_reads": 0, "normalization_rerun": False,
        "new_answers": 0, "new_scoring": 0, "classifiers": False, "interventions": False, "Nemotron_touched": False,
        "fixed_projection_axis_limits": True, "other_sequences_accessible": True,
        "caption": "Before-answer states labeled by saved answers; projected distances are approximate, and separation does not establish a cause of false reporting. No generation, rescoring, classifiers, or interventions.",
        "caption_in_report_only": True, "group_counts_in_report_only": True,
        "outputs": {"html": {"file": str(html.relative_to(root)), "sha256": sha256_file(html), "bytes": html.stat().st_size},
                    "png": {"file": str(figure_path.relative_to(root)), "sha256": sha256_file(figure_path), "bytes": figure_path.stat().st_size}}}
    presentation_path = output_prefix.with_suffix(".json")
    atomic_write_json(presentation_path, presentation)
    manifest.setdefault("original_all_task_html_sha256", previous_html_sha)
    manifest.setdefault("original_all_task_browser_rendering_verification", manifest.get("browser_rendering_verification", {}))
    manifest.pop("browser_rendering_verification", None)
    manifest["outputs"]["html"] = presentation["outputs"]["html"]
    report = source_prefix.with_suffix(".txt")
    manifest["outputs"]["txt"] = {"file": str(report.relative_to(root)), "sha256": sha256_file(report), "bytes": report.stat().st_size}
    manifest["focused_presentation"] = {"manifest": str(presentation_path.relative_to(root)),
        "figure": str(figure_path.relative_to(root)), "pca_refits": 0,
        "presentation_pipeline_sha256": presentation["presentation_pipeline_sha256"]}
    atomic_write_json(source_manifest, manifest)
    return {"status": "presentation_only", "PCA_refits": 0, "state_cache_reads": 0,
        "group_tasks": {key: {sequence: len(ids) for sequence,ids in values.items()} for key,values in selected_ids.items()},
        "outputs": presentation["outputs"]}


def render_pca_html(payload, plotly_js, output):
    """Render fixed task groups using already-saved, unchanged PCA coordinates."""
    import json
    from pathlib import Path

    if len(payload.get("models", [])) != 3:
        raise ValueError("The PCA dashboard requires three model-specific maps")
    policies = {"baseline", "transparency", "evidence_contract"}
    for model in payload["models"]:
        if len(model["explained_variance_ratio"]) != 3:
            raise ValueError("Each model needs exactly three explained-variance ratios")
        if len(model["axis_range"]) != 2 or model["axis_range"][0] >= model["axis_range"][1]:
            raise ValueError("Each model needs a fixed, shared-axis interval")
        tasks = {}
        for point in model["points"]:
            if point["policy"] not in policies or point["outcome"] not in {"W", "C"}:
                raise ValueError("Unexpected policy or saved outcome")
            tasks.setdefault(point["task_id"], []).append(point["policy"])
        if any(len(items) != 3 or set(items) != policies for items in tasks.values()):
            raise ValueError("Only complete, unique task triplets may enter a map")
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FTA prompt-state PCA</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f7f8fa;color:#182230;font:15px/1.45 system-ui,-apple-system,sans-serif}
main{max-width:1760px;margin:auto;padding:20px}h1{font-size:23px;line-height:1.2;margin:0 0 8px}p{margin:7px 0}.intro{max-width:1080px;color:#425166}
.controls{background:white;border:1px solid #dce1e7;border-radius:9px;padding:12px 15px;margin:16px 0 10px}.buttons,.sequences,.legend{display:flex;flex-wrap:wrap;align-items:center;gap:9px}
button{font:inherit;font-size:13px;border:1px solid #bbc6d2;background:#fff;color:#24374d;border-radius:5px;padding:5px 10px;cursor:pointer}button:hover{background:#eef3f8}button:focus-visible,input:focus-visible{outline:3px solid #6ca7e7;outline-offset:2px}
.sequences{margin-top:10px;gap:7px 15px}.sequences label{white-space:nowrap;font-size:13px;cursor:pointer}.sequences input{accent-color:#356ba0;vertical-align:middle;margin:0 5px 0 0}details{margin-top:10px;font-size:13px}summary{cursor:pointer;color:#43546a}details .buttons{margin-top:9px}
.legend{font-size:13px;padding:7px 1px;gap:10px 20px}.key{display:inline-flex;align-items:center;gap:6px}.dot{width:9px;height:9px;display:inline-block;border-radius:50%}.shape{font-size:15px;line-height:1;color:#556372}.square{width:8px;height:8px;display:inline-block;background:#556372}.line-key{display:inline-block;width:19px;border-top:2px solid #738296}.line-key.dashed{border-top-style:dashed;opacity:.55}
.maps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.card{min-width:0;background:white;border:1px solid #dce1e7;border-radius:9px;overflow:hidden}.card-head{padding:12px 14px 0}.card h2{font-size:18px;line-height:1.2;margin:0 0 5px}.counts,.variance{font-size:12px;color:#556477}.plot{width:100%;height:525px}.sequence-counts{padding:0 14px 12px;font-size:12px;color:#556477}.caveat{font-size:13px;color:#526174;margin-top:14px;max-width:1250px}
@media(max-width:1080px){.maps{grid-template-columns:1fr}.plot{height:580px}.card-head{padding-top:13px}}@media(max-width:550px){main{padding:12px}.plot{height:430px}.buttons{gap:6px}.buttons button{font-size:12px}}
</style>
<script>__PLOTLY_JS__</script></head><body><main>
<h1>Before-answer states</h1>
<p class="intro">Triangles track the same W-W-C tasks under all three instructions, including their clean contract answer. Circles track C-C-C tasks: the always-clean reference group, not matched controls. W means unsupported success OR invented details; C means neither flag, not guaranteed correctness.</p>
<div class="controls" aria-label="Outcome sequence filters">
<div class="buttons"><strong style="font-size:13px">Show tasks:</strong><button data-preset="comparison">W-W-C + C-C-C</button><button data-preset="wwc">W-W-C only</button><button data-preset="ccc">C-C-C only</button></div>
<div class="sequences" id="main-sequences"></div>
<details id="other-panel"><summary>Other observed sequences</summary><div class="buttons"><button data-preset="all">Show all sequences</button><button data-preset="other">Only other sequences</button><button data-preset="hide-other">Hide other sequences</button></div><div class="sequences" id="other-sequences"></div></details>
<p style="font-size:12px;color:#59687a;margin:8px 0 0">Sequence order: baseline → transparency → evidence contract. Filtering keeps the same PCA coordinates and axis limits.</p>
</div>
<div class="legend" aria-label="Point legend"><span class="key"><i class="dot" style="background:#4c78a8"></i>Baseline</span><span class="key"><i class="dot" style="background:#1b9e77"></i>Transparency</span><span class="key"><i class="dot" style="background:#e69f00"></i>Evidence contract</span><span class="key"><span class="shape">▲</span><i class="line-key"></i>W-W-C</span><span class="key"><i class="dot" style="background:#556372"></i><i class="line-key dashed"></i>C-C-C</span><span class="key"><i class="square"></i>Other sequences</span></div>
<div class="maps" id="maps"></div>

</main><script type="application/json" id="pca-data">__PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('pca-data').textContent);
const policyInfo=[['baseline','Baseline','#4c78a8'],['transparency','Transparency','#1b9e77'],['evidence_contract','Evidence contract','#e69f00']];
const mainSequences=['W-W-C','C-C-C'];
const sequenceOrder=['W-W-C','C-C-C','W-C-C','W-W-W','C-W-C','C-W-W','W-C-W','C-C-W'];
const observed=new Set(data.models.flatMap(m=>m.points.map(p=>p.sequence)));
const sequences=sequenceOrder.filter(s=>mainSequences.includes(s)||observed.has(s));
for(const s of [...observed].sort())if(!sequences.includes(s))sequences.push(s);
const selected=new Set(mainSequences),checkboxes=new Map();
const escapeText=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const groupOf=point=>mainSequences.includes(point.sequence)?point.sequence:'other';
for(const s of sequences){const label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=selected.has(s);box.setAttribute('aria-label','Show '+s);label.append(box,document.createTextNode(s));document.getElementById(mainSequences.includes(s)?'main-sequences':'other-sequences').append(label);checkboxes.set(s,box);box.addEventListener('change',()=>{if(box.checked)selected.add(s);else selected.delete(s);redrawAll();});}
const maps=[];
data.models.forEach((model,i)=>{const card=document.createElement('section');card.className='card';const head=document.createElement('div');head.className='card-head';const title=document.createElement('h2');title.textContent=model.name;const counts=document.createElement('div');counts.className='counts';counts.setAttribute('aria-live','polite');const variance=document.createElement('div');variance.className='variance';variance.textContent='PC1–3 explain '+(100*model.explained_variance_ratio.reduce((a,b)=>a+b,0)).toFixed(1)+'% of variance';head.append(title,counts,variance);const graph=document.createElement('div');graph.className='plot';graph.id='pca-map-'+i;const seqCounts=document.createElement('div');seqCounts.className='sequence-counts';card.append(head,graph,seqCounts);document.getElementById('maps').append(card);maps.push({model,graph,counts,seqCounts});});
function tracesFor(model){
 const points=model.points.filter(p=>selected.has(p.sequence)),taskMap=new Map();
 for(const p of points){if(!taskMap.has(p.task_id))taskMap.set(p.task_id,{});taskMap.get(p.task_id)[p.policy]=p;}
 const traces=[];
 for(const group of ['C-C-C','other','W-W-C'])for(const [target,,color] of policyInfo.slice(1)){
  const x=[],y=[],z=[],cx=[],cy=[],cz=[],u=[],v=[],w=[];
  for(const task of taskMap.values()){const a=task.baseline,b=task[target];if(!a||!b||groupOf(a)!==group)continue;const dx=b.pc1-a.pc1,dy=b.pc2-a.pc2,dz=b.pc3-a.pc3,norm=Math.hypot(dx,dy,dz);x.push(a.pc1,b.pc1,null);y.push(a.pc2,b.pc2,null);z.push(a.pc3,b.pc3,null);if(norm>0){cx.push(b.pc1);cy.push(b.pc2);cz.push(b.pc3);u.push(dx/norm);v.push(dy/norm);w.push(dz/norm);}}
  const clean=group==='C-C-C',other=group==='other';
  traces.push({type:'scatter3d',mode:'lines',meta:{group,target},x,y,z,line:{color,width:clean?1.2:1.9,dash:clean?'dash':other?'dot':'solid'},opacity:clean?0.16:other?0.27:0.52,hoverinfo:'skip',showlegend:false,connectgaps:false});
  traces.push({type:'cone',meta:{group,target},x:cx,y:cy,z:cz,u,v,w,anchor:'tip',sizemode:'absolute',sizeref:(model.axis_range[1]-model.axis_range[0])*0.018,colorscale:[[0,color],[1,color]],showscale:false,opacity:clean?0.22:other?0.36:0.65,hoverinfo:'skip',showlegend:false});
 }
 for(const group of ['C-C-C','other','W-W-C'])for(const [policy,label,color] of policyInfo){
  const subset=points.filter(p=>p.policy===policy&&groupOf(p)===group),triangle=group==='W-W-C';
  const trace={type:'scatter3d',mode:triangle?'markers+text':'markers',name:label+' · '+group,meta:{group,policy},x:subset.map(p=>p.pc1),y:subset.map(p=>p.pc2),z:subset.map(p=>p.pc3),marker:{color,size:triangle?1.5:group==='other'?3.3:3.6,symbol:group==='other'?'square':'circle',opacity:triangle?1:0.86,line:{color:triangle?color:'#ffffff',width:triangle?0:0.35}},customdata:subset.map(p=>[p.task_id,label,p.outcome,p.sequence,p.failure_category,p.pressure_type,p.false_success,p.fabricated_details].map(escapeText)),hovertemplate:'<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>Outcomes B → T → EC: %{customdata[3]}<br>Category: %{customdata[4]}<br>Pressure: %{customdata[5]}<br>Unsupported success: %{customdata[6]}<br>Invented details: %{customdata[7]}<extra></extra>',showlegend:false};
  if(triangle){trace.text=subset.map(()=>'▲');trace.textposition='middle center';trace.textfont={family:'Arial, sans-serif',size:13,color};}
  traces.push(trace);
 }
 return {traces,points,tasks:taskMap};
}
function redrawAll(){for(const entry of maps){const {model,graph,counts,seqCounts}=entry,{traces,points,tasks}=tracesFor(model),totalTasks=model.points.length/3;
 const axis=index=>({title:{text:'PC'+(index+1)+' ('+(100*model.explained_variance_ratio[index]).toFixed(1)+'%)',font:{size:11}},range:model.axis_range.slice(),autorange:false,nticks:4,tickfont:{size:9},showbackground:true,backgroundcolor:'#fafbfc',gridcolor:'#e2e7ed',zerolinecolor:'#bcc7d1'});
 const layout={margin:{l:0,r:0,b:0,t:0},paper_bgcolor:'#fff',font:{family:'system-ui, sans-serif',color:'#34445a'},showlegend:false,hovermode:'closest',uirevision:model.id+'-fixed-pca',scene:{xaxis:axis(0),yaxis:axis(1),zaxis:axis(2),aspectmode:'cube',camera:{eye:{x:1.5,y:1.5,z:1.15}},dragmode:'orbit',uirevision:model.id+'-fixed-camera'}};
 Plotly.react(graph,traces,layout,{responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['toImage'],showTips:false});
 counts.textContent=tasks.size+' / '+totalTasks+' tasks shown · '+points.length+' answers';
 const shown=new Map(),total=new Map();for(const task of tasks.values()){const s=task.baseline.sequence;shown.set(s,(shown.get(s)||0)+1);}for(const p of model.points)if(p.policy==='baseline')total.set(p.sequence,(total.get(p.sequence)||0)+1);
 const extra=sequences.filter(s=>!mainSequences.includes(s)&&shown.has(s));seqCounts.textContent=extra.length?'Other groups: '+extra.map(s=>s+': '+shown.get(s)).join(' · '):'';
 }}
for(const button of document.querySelectorAll('[data-preset]'))button.addEventListener('click',()=>{const preset=button.dataset.preset;if(preset==='hide-other'){for(const s of sequences)if(!mainSequences.includes(s))selected.delete(s);}else{selected.clear();for(const s of sequences){const yes=preset==='all'||(preset==='comparison'&&mainSequences.includes(s))||(preset==='wwc'&&s==='W-W-C')||(preset==='ccc'&&s==='C-C-C')||(preset==='other'&&!mainSequences.includes(s));if(yes)selected.add(s);}}for(const [s,box]of checkboxes)box.checked=selected.has(s);redrawAll();});
redrawAll();
</script></body></html>'''
    page = page.replace("__PLOTLY_JS__", plotly_js).replace("__PAYLOAD__", encoded)
    Path(output).write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="Cache prompt-only final-input-token block states externally")
    extraction.add_argument("--model", choices=EXTRACTION_MODELS, required=True)
    extraction.add_argument("--cache-dir", type=Path, required=True)
    reporting = commands.add_parser("analyze", help="Compare paired policies using existing saved labels")
    reporting.add_argument("--cache-dir", type=Path, required=True)
    reporting.add_argument("--output-prefix", type=Path)
    during = commands.add_parser("during-replay", help="Causal saved-answer replay and separate one-shot same-prefix continuations")
    during.add_argument("--model", choices=EXTRACTION_MODELS, required=True)
    during.add_argument("--plan", type=Path, default=ROOT / "config/during_answer.json")
    during.add_argument("--cache-dir", type=Path, required=True)
    reporting_during = commands.add_parser("during-report", help="Report reconstructed answer events and once-reviewed new continuations")
    reporting_during.add_argument("--plan", type=Path, default=ROOT / "config/during_answer.json")
    reporting_during.add_argument("--cache-dir", type=Path, required=True)
    reporting_during.add_argument("--reviews", type=Path, required=True)
    reporting_during.add_argument("--output-prefix", type=Path, default=ROOT / "results/during_answer")
    packets = commands.add_parser("during-packets", help="Export new supplied-prefix reports for one contextual review")
    packets.add_argument("--model", choices=EXTRACTION_MODELS, required=True)
    packets.add_argument("--plan", type=Path, default=ROOT / "config/during_answer.json")
    packets.add_argument("--cache-dir", type=Path, required=True)
    packets.add_argument("--output", type=Path, required=True)
    pca = commands.add_parser("pca", help="Visualize original cached final-input states with one pooled PCA per model")
    pca.add_argument("--cache-dir", type=Path, required=True)
    pca.add_argument("--projection-cache-dir", type=Path, required=True)
    pca.add_argument("--output-prefix", type=Path, default=ROOT / "results/pca")
    focused = commands.add_parser("pca-focus", help="Restyle saved PCA coordinates only; no state-cache access or PCA fitting")
    focused.add_argument("--source-prefix", type=Path, default=ROOT / "results/pca")
    focused.add_argument("--output-prefix", type=Path, default=ROOT / "results/pca_focused")
    args = parser.parse_args()
    if args.command == "pca-focus":
        print(json.dumps(pca_focus(args.root, args.source_prefix, args.output_prefix)), flush=True)
    elif args.command == "pca":
        print(json.dumps(pca_visualize(args.root, args.cache_dir, args.projection_cache_dir, args.output_prefix)), flush=True)
    elif args.command == "during-packets":
        during_packets(args.root, args.plan, args.cache_dir, args.output, args.model)
    elif args.command == "during-report":
        during_report(args.root, args.plan, args.cache_dir, args.reviews, args.output_prefix)
    elif args.command == "during-replay":
        during_replay(args.root, args.model, args.plan, args.cache_dir)
    elif args.command == "extract":
        extract(args.root, args.model, args.cache_dir)
    else:
        destination = args.output_prefix or args.root / "results/internal_states"
        analyze(args.root, args.cache_dir, destination)


if __name__ == "__main__":
    main()
