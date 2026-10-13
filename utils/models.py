"""Pinned official checkpoints and native interfaces; no checkpoint substitutions."""
import hashlib
import json
import os
from pathlib import Path

from utils.io import read_json, sha256_file


def digest(value):
    payload = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_spec(root, name):
    config = read_json(Path(root) / "config/models.json")
    matches = [(key, spec) for key, spec in config["models"].items()
               if name in (key, spec["repo_id"])]
    if len(matches) != 1:
        raise ValueError("Choose one configured official model: " + str(name))
    key, spec = matches[0]
    return dict(spec, key=key, generation=config["new_generation"])


def cached_snapshot(spec):
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        cache = str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    return Path(cache) / ("models--" + spec["repo_id"].replace("/", "--")) / "snapshots" / spec["revision"]


def verify_snapshot(spec, require_weights=False):
    """Check the requested revision and complete sharded weights without rereading 230 GiB."""
    snapshot = cached_snapshot(spec)
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError("Official pinned snapshot is unavailable: " + str(snapshot))
    config = read_json(snapshot / "config.json")
    if spec["architecture"] not in config.get("architectures", []):
        raise ValueError("Cached checkpoint architecture differs from the configured official model")
    weights = []
    index = snapshot / "model.safetensors.index.json"
    if require_weights:
        if index.exists():
            weights = sorted(set(read_json(index)["weight_map"].values()))
        elif (snapshot / "model.safetensors").exists():
            weights = ["model.safetensors"]
        else:
            raise FileNotFoundError("Official safetensor weights are not downloaded")
        missing = [name for name in weights if not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0]
        if missing:
            raise FileNotFoundError("Missing official weight shards: " + ", ".join(missing))
    hashes = {name: sha256_file(snapshot / name) for name in
              ("config.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json",
               "generation_config.json", "tokenizer.json", "tekken.json", "special_tokens_map.json")
              if (snapshot / name).is_file()}
    return {"repo_id": spec["repo_id"], "revision": spec["revision"], "snapshot": str(snapshot),
            "metadata_sha256": hashes, "weight_files": weights,
            "weight_bytes": sum((snapshot / name).stat().st_size for name in weights),
            "weights_verified_by": "pinned Hugging Face download; index membership and nonempty local shard checks"}


def load_tokenizer(spec):
    snapshot = cached_snapshot(spec)
    if spec["tokenizer_backend"] == "mistral_common":
        from transformers import MistralCommonBackend
        tokenizer = MistralCommonBackend.from_pretrained(str(snapshot), local_files_only=True)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            raise ValueError("Official checkpoint has no native chat template")
    return tokenizer


def native_input(tokenizer, spec, record, tensors=False):
    """Tokenize original system/user strings directly with the official native interface."""
    messages = [{"role": "system", "content": record["system"]},
                {"role": "user", "content": record["user"]}]
    kwargs = dict(spec.get("template_kwargs", {}), add_generation_prompt=True)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if not isinstance(rendered, str) or any(message["content"] not in rendered for message in messages):
        raise ValueError("The native formatter changed a supplied author message; no automatic adapter is allowed")
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                            return_tensors="pt" if tensors else None, **kwargs)
    ids = encoded["input_ids"]
    if tensors:
        ids = ids[0].tolist()
    elif ids and isinstance(ids[0], list):
        ids = ids[0]
    return encoded, {"original_messages": messages, "effective_messages": messages,
                     "native_template_mode": "native_system_user", "assistant_prefill": "",
                     "native_prompt": rendered, "native_prompt_sha256": digest(rendered),
                     "native_input_token_ids_sha256": digest(ids), "input_length": len(ids),
                     "logical_input_sha256": digest(record)}
