# Failure reporting study

A compact replay of the unchanged **Failure-Transparent Agents v0.3.0** dataset: 100 tasks, the authors' three instruction policies, and 300 frozen system/user inputs per model. Each supplied tool observation lacks the evidence needed to establish the requested outcome.

The completed Qwen3-8B and Gemma-2-9b-it replay is preserved. Its two greedy repetitions were identical, so comparisons select repetition 0: 300 answers per model. Their existing labels are reused without scoring them again. New official models generate one report per task/policy. New answers receive one Codex-assisted full-answer rubric review; this is not independent human annotation or reproduction of the authors' published human scores.

## Current scope

Only the original tasks, failed-tool observations, policies, and message formatting are used. No rewritten questions, added reminders, synthetic success/recovery histories, probes, or activation interventions are part of this phase. Model-native chat templates encode the frozen system/user strings. Only the original evidence-contract instruction requests structured reporting; ordinary answers are not forced to use status labels.

[Model pins and decoding settings](config/models.json), [source hashes](config/source.json), [saved answers](results/answers.jsonl), and [saved labels](results/labels.jsonl) are the primary records. [The summary](results/summary.json) records denominators and uncertainty. [Exact examples](results/examples.txt) contain full copied instructions, requests, tool observations, and evaluated answers.

## Completed results

Ministral 3 8B generated **300 new answers** on Newton using its official pinned checkpoint and native Mistral Common template. All 300 ended normally; none were truncated. They were scored once. Combined with 600 reused Qwen/Gemma answers, the comparison contains 900 distinct model/task/policy inputs. All 1,200 historical Qwen/Gemma reports remain preserved separately.

The table shows positive judgments / valid answers with resolved judgments, in **baseline / transparency / evidence-contract** order. A label concerns an answer, not a fraction of its words. Missing or unresolved judgments are not counted as negative. The figure's “Recovery step” counts a suggested next action in the answer.

| Model | Unsupported success | Invented details | Failure disclosed |
| --- | --- | --- | --- |
| Qwen3-8B | 25/98 · 20/98 · 0/99 | 29/100 · 24/100 · 0/99 | 75/100 · 77/100 · 99/99 |
| Gemma-2-9b-it | 8/99 · 6/100 · 0/95 | 9/100 · 8/100 · 1/95 | 88/99 · 88/100 · 95/95 |
| Ministral 3 8B | 31/100 · 23/100 · 0/100 | 50/99 · 36/99 · 4/99 | 77/100 · 80/100 · 100/100 |

The evidence contract reduced unsupported success reports in these saved answers, but it did not remove every invented detail. For Ministral, invented **requested facts** were separately flagged in 38/99 baseline, 22/98 transparency, and 1/99 resolved evidence-contract answers. Some invented details concern other assertions. One answer can both admit failure and supply an unsupported answer: the saved resume example says no document was provided, then confirms the user's suggested employer as though it were listed in that resume.

![Original FTA rubric comparison](results/summary.png)

The figure excludes six truncated historical answers and excludes unresolved judgments separately for each label. These Codex-assisted labels and differing old/new token limits do not establish an internal mechanism or a controlled ranking of the models.

## Blocked models

**Llama 3.1 8B Instruct: 0/300 answers.** The official pinned repository returned HTTP 403 with both available Hugging Face credential slots, and approved cached weights were not found. [The access record](results/blocker_llama3_1_8b.json) contains the failure without credentials.

**Nemotron 3 Super 120B BF16: 0/300 answers.** Its official weights are cached and all 300 native inputs were checked. [NVIDIA specifies eight H100-80GB GPUs](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16); Newton's eligible nodes were occupied. The [Slurm resource check](results/nemotron_resources.json) estimated October 16, 2026 at 11:39:58 EDT, several days after this run. This is an estimate, not a reservation. The held generation job was cancelled before allocation or model loading. Native-kernel setup and its compatibility checks are recorded separately; no smaller or quantized checkpoint was substituted.

[Cache/input readiness](results/nemotron_readiness.json) records the 50 official shards and native-template checks. [Runtime setup](results/nemotron_runtime_setup.json) records source builds and compatibility checks, including failed attempts. Native fast kernels are also **unverified**: Mamba 2.3.2's import failed with the installed Triton, and the compatible 2.3.1 source build was stopped before installation once the multi-day GPU blocker was confirmed. Its required GPU guard has not passed. Runtime readiness is separate from obtaining the required GPU allocation.

The intended expansion was 900 new answers across three models. **300 new answers actually ran**; the two blocked models have no error-rate estimates and no bars in the figure. Their pinned settings and the generation pipeline remain available for a later run when access and resources permit.

## Layout

- `data/`: unchanged original dataset, supplied logical inputs, annotation guide, and original formatter/policy source for verification.
- `config/`: unchanged policy strings, pinned source/model metadata, archive and preservation records, scheduler settings.
- `utils/`: five helper modules for IO, input verification, native models, inference runtime, and scoring/report aggregation.
- `pipelines/`: three entry points for preparation, generation, and scoring/reporting.
- `results/`: consolidated comparison answers/labels, model manifests, summary, figure, exact examples, and immutable completed replay in `preserved_exact/`.

Obsolete custom pilots, diagnostics, intervention code, historical review campaigns, and duplicate exports are recoverable outside the active project at `newton:/lustre/fs1/home/sa746371/fta_project_archive`. [Archive metadata](config/archive.json) describes recovery. The previous public Git history is retained; this refactor uses a normal commit, not a history rewrite. Model caches and the original inference environment are preserved outside the repository.

## Run the three pipelines

From the repository root, Python 3.11+ preparation needs only the standard library:

```sh
python3 -B -m pipelines.prepare
```

It checks the seven frozen file hashes, all 100 tasks, three policies, and all 300 messages against the original author formatter. It consumes the supplied inputs rather than rebuilding them. `--fetch-missing` fetches missing upstream files from v0.3.0 and verifies their recorded hashes; `--source-bundle` recovers supplied prebuilt inputs/policies when necessary. Existing altered files are rejected and left untouched.

Use official pinned weights in an external Hugging Face cache and the recorded inference dependencies. Generation requires an allocated GPU node. Check native templates on CPU before generation:

```sh
python3 -B -m pipelines.generate preflight --model ministral3_8b --templates-only
python3 -B -m pipelines.generate run --model ministral3_8b
```

The same pipeline accepts `llama3_1_8b` and `nemotron3_super120b`. It resumes missing slots and preserves saved reports. Infrastructure errors stop the job and are recorded separately; they do not become fabricated answer rows. Qwen/Gemma generation is disabled in this phase. Weight access and hardware requirements are checked before model loading; blocked models do not receive invented results or zero-error bars.

New settings use greedy decoding, a 2,048-token cap, and one repetition. The old replay used a 160-token cap and two identical repetitions. These are different generation conditions; compare the recorded denominators and settings rather than treating model differences as a controlled token-budget experiment. Nemotron uses its documented native non-thinking option; Ministral uses the official Mistral Common interface and documented BF16 dequantization of official FP8 weights.

Scoring uses one full-response judgment per new answer under the six original rubric labels, plus separately recorded reporting diagnostics. Export packets and save the reviewer judgments outside the active project:

```sh
python3 -B -m pipelines.report packets --output /external/review/packets.jsonl
python3 -B -m pipelines.report score --reviews /external/review/reviews.jsonl
python3 -B -m pipelines.report report
```

The importer requires every label, exact positive evidence spans, confidence, and one review per unscored answer. It rejects repeated reviews and preserves old judgments. Reports exclude technical failures, empty/truncated answers, and confidence-1 labels from the applicable denominators. Labels overlap: one answer may admit failure and also assert an unsupported fact. The figure uses the original rubric; supplementary requested-fact and header diagnostics remain in the JSON summary.

## Attribution

Study code uses the [MIT License](LICENSE). Original FTA files retain [Junru Zhu's MIT notice](data/FTA_LICENSE). Official model weights, caches, environments, credentials, and old archives are excluded from Git; see [third-party notices](THIRD_PARTY_NOTICES.md).
