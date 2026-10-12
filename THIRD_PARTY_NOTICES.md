# Third-party notices

The repository's [MIT License](LICENSE), copyright (c) 2026 Samira Jahangiri, covers study-specific code and documentation. The following upstream material retains its own attribution and license.

## Failure-Transparent Agents

The exact replay includes the unchanged task dataset and selected source/documentation files from [Junru Zhu's Failure-Transparent Agents repository, release v0.3.0](https://github.com/junru-zhu/failure-transparent-agents/tree/v0.3.0). The bundle records source URLs, file sizes, and SHA-256 hashes in [MANIFEST.json](experiments/fta_exact_replay_20261012_170305/fta_exact_tasks/MANIFEST.json).

The upstream files are licensed under the **MIT License**, copyright **(c) 2026 Junru Zhu**. The full copyright, permission, and warranty notice is preserved in [the bundled upstream LICENSE](experiments/fta_exact_replay_20261012_170305/fta_exact_tasks/upstream/LICENSE), including inside `fta_exact_tasks.zip`. Retain this notice when redistributing copies or substantial portions of the upstream material. This notice does not assert ownership of that material or endorsement by its author.

The generated logical model inputs use the bundled original policy strings and formatter. The upstream dataset SHA-256 is `4bb2f63bc0bf03570021246943d4c1d871d0fe29e78edcdcfe740c0f3ee716ea`.

## Models and dependencies

The results name [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), revision `b968826d9c46dd6066d109eabc6255188de91218`, and [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it), revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`. Their native chat-template text and revision metadata are retained as run provenance. **Model weights are not redistributed.** Obtain any model files separately and comply with the respective model's license and access terms.

Dependencies installed for inference or plotting retain their own licenses. Requirement files specify the recorded environment; they do not redistribute those packages or change their licensing.
