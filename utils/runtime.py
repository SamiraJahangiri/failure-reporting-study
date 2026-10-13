"""Small inference runtime for natural reports from official pinned checkpoints."""
import datetime
import importlib.metadata
import os
import platform
import time

from utils.models import cached_snapshot, load_tokenizer, native_input, verify_snapshot


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def packages():
    versions = {}
    for name in ("torch", "torchvision", "torchaudio", "transformers", "accelerate", "safetensors", "mistral-common", "huggingface-hub",
                 "numpy", "tokenizers", "triton", "mamba-ssm", "causal-conv1d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def compute_preflight(spec):
    import torch
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")):
        raise RuntimeError("Generation requires a scheduler allocation; do not run models on a login node")
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(i))}
               for i in range(torch.cuda.device_count())]
    needs = spec["resources"]
    eligible = [gpu for gpu in devices if gpu["total_memory_bytes"] >= needs["min_gpu_memory_gib"] * 1024**3 * 0.96
                and gpu["compute_capability"][0] >= needs["min_compute_capability_major"]]
    if len(eligible) < needs["min_gpus"]:
        raise RuntimeError("Insufficient resources for " + spec["repo_id"] + ": requires " +
                           str(needs) + "; allocated GPUs: " + str(devices))
    return {"checked_at": utc_now(), "scheduler_job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("PBS_JOBID")),
            "scheduler_partition": os.environ.get("SLURM_JOB_PARTITION"), "gpus": devices,
            "python": platform.python_version(), "packages": packages(), "cuda_runtime": torch.version.cuda}


def nemotron_kernel_preflight():
    """Require native fast kernels: installed Transformers' Torch fallback has upstream issue 47246."""
    from transformers.integrations import lazy_load_kernel
    from transformers.utils.import_utils import resolve_internal_import
    convolution = lazy_load_kernel("causal-conv1d")
    mamba = lazy_load_kernel("mamba-ssm")
    functions = {"causal_conv1d_fn": getattr(convolution, "causal_conv1d_fn", None),
                 "causal_conv1d_update": getattr(convolution, "causal_conv1d_update", None)}
    for path in ("ops.triton.selective_state_update.selective_state_update",
                 "ops.triton.ssd_combined.mamba_chunk_scan_combined",
                 "ops.triton.ssd_combined.mamba_split_conv1d_scan_combined"):
        functions[path] = resolve_internal_import(mamba, chained_path=path)
    missing = [name for name, function in functions.items() if not callable(function)]
    if missing:
        raise RuntimeError("Nemotron native fast kernels are unavailable: " + ", ".join(missing) +
                           "; refusing the incorrect Torch fallback documented at https://github.com/huggingface/transformers/issues/47246")
    return {"backend": "native_mamba_ssm_and_causal_conv1d", "functions_available": sorted(functions),
            "torch_fallback_permitted": False, "upstream_correctness_issue": "https://github.com/huggingface/transformers/issues/47246"}


class ReportGenerator:
    def __init__(self, spec):
        import torch
        from transformers import AutoModelForCausalLM, GenerationConfig
        self.torch, self.spec = torch, spec
        self.hardware = compute_preflight(spec)
        self.snapshot_info = verify_snapshot(spec, require_weights=True)
        self.tokenizer = load_tokenizer(spec)
        self.kernel_info = nemotron_kernel_preflight() if spec["key"] == "nemotron3_super120b" else None
        kwargs = {"local_files_only": True, "trust_remote_code": False, "device_map": "auto",
                  "torch_dtype": torch.bfloat16, "attn_implementation": "eager"}
        if spec.get("dequantize_official_fp8"):
            from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration
            kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
            model_class = Mistral3ForConditionalGeneration
        else:
            model_class = AutoModelForCausalLM
        self.loading_info = None
        if spec["key"] == "nemotron3_super120b":
            self.model, information = model_class.from_pretrained(
                str(cached_snapshot(spec)), output_loading_info=True, **kwargs)
            self.loading_info = {name: sorted(value) if isinstance(value, (set, list)) else value
                                 for name, value in information.items()}
            failures = {name: self.loading_info.get(name, []) for name in
                        ("missing_keys", "mismatched_keys", "error_msgs")}
            failures["unexpected_keys"] = [name for name in self.loading_info.get("unexpected_keys", [])
                                           if not name.startswith("mtp.")]
            if any(failures.values()):
                counts = {name: len(value) for name, value in failures.items() if value}
                examples = {name: value[:10] for name, value in failures.items() if value}
                raise RuntimeError("Official Nemotron weights did not load completely: " +
                                   str(counts) + "; examples: " + str(examples))
        else:
            self.model = model_class.from_pretrained(str(cached_snapshot(spec)), **kwargs)
        if self.kernel_info:
            from transformers.models.nemotron_h import modeling_nemotron_h
            if not modeling_nemotron_h.is_fast_path_available:
                raise RuntimeError("Nemotron model initialization selected the Torch fallback; generation is blocked")
        self.model.eval()
        if any(str(device) in ("cpu", "disk") for device in getattr(self.model, "hf_device_map", {}).values()):
            raise RuntimeError("Official model did not fit allocated GPU memory; CPU/disk offloading is not substituted")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.input_device = self.model.get_input_embeddings().weight.device
        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        official = self.model.generation_config
        self.generation_config.eos_token_id = official.eos_token_id or self.tokenizer.eos_token_id
        self.generation_config.pad_token_id = official.pad_token_id
        if self.generation_config.pad_token_id is None:
            eos = self.generation_config.eos_token_id
            self.generation_config.pad_token_id = eos[0] if isinstance(eos, list) else eos
        for key, value in spec["generation"].items():
            if key in ("seed", "repetitions", "batch_size", "settings_difference_from_existing"):
                continue
            setattr(self.generation_config, key, value)
        torch.manual_seed(spec["generation"]["seed"])
        torch.cuda.manual_seed_all(spec["generation"]["seed"])

    def manifest(self):
        return {"model": self.spec["repo_id"], "revision": self.spec["revision"],
                "snapshot": self.snapshot_info, "hardware": self.hardware,
                "model_class": type(self.model).__name__, "tokenizer_class": type(self.tokenizer).__name__,
                "template_kwargs": self.spec.get("template_kwargs", {}), "system_role_adapter": None,
                "assistant_prefill": "", "generation": self.spec["generation"],
                "effective_generation_config": self.generation_config.to_dict(),
                "native_kernel_preflight": self.kernel_info,
                "checkpoint_loading_info": self.loading_info,
                "hf_device_map": getattr(self.model, "hf_device_map", {}),
                "weights_precision": "bfloat16", "official_fp8_dequantized": self.spec.get("dequantize_official_fp8", False),
                "inference_mode": True, "native_input_encoding": "apply_chat_template(tokenize=True); no prompt re-encoding"}

    def answer(self, record):
        encoded, provenance = native_input(self.tokenizer, self.spec, record, tensors=True)
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        config = getattr(self.model.config, "text_config", self.model.config)
        limit = getattr(config, "max_position_embeddings", 262144)
        if provenance["input_length"] + self.spec["generation"]["max_new_tokens"] > limit:
            raise RuntimeError("Original input and generation allowance exceed native context; no truncation permitted")
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, generation_config=self.generation_config)[0]
        continuation = output[encoded["input_ids"].shape[-1]:].tolist()
        eos = self.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        stopped = bool(continuation and continuation[-1] in eos)
        return dict(provenance, response=self.tokenizer.decode(continuation, skip_special_tokens=True,
                                                              clean_up_tokenization_spaces=False),
                    raw_generated_text=self.tokenizer.decode(continuation, skip_special_tokens=False,
                                                            clean_up_tokenization_spaces=False),
                    output_token_ids=continuation, token_count=len(continuation),
                    finish_reason="eos" if stopped else "length", truncated=not stopped,
                    generation_seconds=round(time.monotonic() - started, 4), technical_error=None)
