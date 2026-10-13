"""Extract prompt-only decoder block states from the exact saved FTA inputs.

Vectors stay in an external cache. No answer tokens, logits, generation,
interventions, or semantic scoring are involved in extraction.
"""
from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.root, args.model, args.cache_dir)
    else:
        destination = args.output_prefix or args.root / "results/internal_states"
        analyze(args.root, args.cache_dir, destination)


if __name__ == "__main__":
    main()
