# Third-party notices

The repository's [MIT License](LICENSE), copyright (c) 2026 Samira Jahangiri, covers study-specific code and documentation. The following upstream material retains its own attribution and license.

## Failure-Transparent Agents

The unchanged dataset, policies, formatter, and annotation guide come from [Junru Zhu's Failure-Transparent Agents v0.3.0](https://github.com/junru-zhu/failure-transparent-agents/tree/v0.3.0). Original URLs, file sizes, and SHA-256 hashes are recorded in [config/source.json](config/source.json). Supplied logical inputs and extracted policies were reused from the existing task bundle; they are not attributed to nonexistent upstream download paths.

The upstream files are licensed under the **MIT License**, copyright **(c) 2026 Junru Zhu**. The full notice is preserved in [data/FTA_LICENSE](data/FTA_LICENSE). Retain it when redistributing this material. No endorsement by dataset or model authors is implied.

The generated logical model inputs use the bundled original policy strings and formatter. The upstream dataset SHA-256 is `4bb2f63bc0bf03570021246943d4c1d871d0fe29e78edcdcfe740c0f3ee716ea`.

## Models and dependencies

The results name [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), revision `b968826d9c46dd6066d109eabc6255188de91218`, and [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it), revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`. Their native chat-template text and revision metadata are retained as run provenance. **Model weights are not redistributed.** Obtain any model files separately and comply with the respective model's license and access terms.

Additional official repositories and pinned revisions are recorded in [config/models.json](config/models.json): [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512), and [NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16). Meta's gated access and Llama license apply to Llama; NVIDIA's Nemotron Open Model License applies to Nemotron. Runtime dequantization of Mistral's official FP8 weights is disclosed. Access and resource blockers are recorded separately from completed inference.

Dependencies installed for inference or plotting retain their own licenses. Requirement files specify the recorded environment; they do not redistribute those packages or change their licensing.
