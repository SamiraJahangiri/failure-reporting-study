# Failure reporting study

A compact replay of the unchanged **Failure-Transparent Agents v0.3.0** dataset: 100 tasks, the authors' three instruction policies, and 300 frozen system/user inputs per model. Each supplied tool observation lacks the evidence needed to establish the requested outcome.

The completed Qwen3-8B and Gemma-2-9b-it replay is preserved. Its two greedy repetitions were identical, so comparisons select repetition 0: 300 answers per model. Their existing labels are reused without scoring them again. New official models generate one report per task/policy. New answers receive one Codex-assisted full-answer rubric review; this is not independent human annotation or reproduction of the authors' published human scores.

## Original replay scope

Only the original tasks, failed-tool observations, policies, and message formatting are used. No rewritten questions, added reminders, synthetic success/recovery histories, probes, or activation interventions are part of the replay phase. Model-native chat templates encode the frozen system/user strings. Only the original evidence-contract instruction requests structured reporting; ordinary answers are not forced to use status labels.

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

**Nemotron retry submitted October 20, 2026:** Native-kernel guard **984814** and dependent full replay **984815** are submitted. The retry fixes source-directory metadata shadowing in the installation check and initializes the library's native bindings before checking its fast-path flag. It reuses the installed official Mamba 2.3.1 build without changing environments, checkpoints, prompts or decoding settings. The [retry wrapper](config/nemotron_retry.sbatch) checks the wheel/source files, all 300 native inputs and CUDA kernels before the eight-H100 replay. [Submission details](results/nemotron_retry_submission.json) preserve the job IDs and hashes. Outputs go to `/lustre/fs1/home/sa746371/fta_nemotron_retry/results`, preserving all earlier results. GPU validation and full multi-GPU generation remain pending; no automatic requeue or scoring is enabled. The resource test estimated October 26, 2026 at 09:41 EDT, which is not a reservation.

**Prior Nemotron status checked October 20, 2026:** Slurm reports main job **902985 cancelled before allocation**. Build prerequisite **902919** failed after 16 minutes: compilation/installation finished, then a post-install source-file verification raised `AssertionError: mamba_ssm/__init__.py`. The dependent guard **902951** and main job were automatically cancelled. Nemotron produced **zero reports**. [The read-only status check](results/nemotron_status_check.json) records the scheduler evidence; the original [submission](results/nemotron_submission.json), stale pending-status file, scripts, environment and weights remain preserved. That check itself made no changes.

The completed 900-answer comparison and [its artifact manifest](https://github.com/SamiraJahangiri/failure-reporting-study/blob/4f1586ba587b45788124780d65fdb74fe4eafc52/results/artifact_manifest.json) describe the preceding published snapshot. New submission metadata does not change its saved answers, labels, or figure.

**Llama 3.1 8B Instruct: 0/300 answers.** The official pinned repository returned HTTP 403 with both available Hugging Face credential slots, and approved cached weights were not found. [The access record](results/blocker_llama3_1_8b.json) contains the failure without credentials.

**Nemotron 3 Super 120B BF16: 0/300 answers.** Its official weights are cached and all 300 native inputs were checked. [NVIDIA specifies eight H100-80GB GPUs](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16); Newton's eligible nodes were occupied. The [Slurm resource check](results/nemotron_resources.json) estimated October 16, 2026 at 11:39:58 EDT, several days after this run. This is an estimate, not a reservation. The held generation job was cancelled before allocation or model loading. Native-kernel setup and its compatibility checks are recorded separately; no smaller or quantized checkpoint was substituted.

[Cache/input readiness](results/nemotron_readiness.json) records the 50 official shards and native-template checks. [Runtime setup](results/nemotron_runtime_setup.json) records source builds and compatibility checks, including failed attempts. Native fast kernels are also **unverified**: Mamba 2.3.2's import failed with the installed Triton, and the compatible 2.3.1 source build was stopped before installation once the multi-day GPU blocker was confirmed. Its required GPU guard has not passed. Runtime readiness is separate from obtaining the required GPU allocation.

The intended expansion was 900 new answers across three models. **300 new answers actually ran**; the two blocked models have no error-rate estimates and no bars in the figure. Their pinned settings and the generation pipeline remain available for a later run when access and resources permit.

## Exploratory prompt-state comparison

A separate [prompt-state pipeline](pipelines/internal_states.py) compares Qwen3-8B, Gemma-2-9b-it, and Ministral 3 8B using the same 300 frozen FTA inputs per model. [The analysis plan](config/internal_states.json) was fixed before inspecting the state vectors. It adds no generations, new scoring, or activation interventions. Saved answers and labels determine the comparison groups; answer text never enters the model input.

Extraction records every decoder block's output at the **last token of the complete native input**, before final normalization and before any answer token. Native rendering, pinned tokenizers/checkpoints, token counts, and original precision are checked against the saved replay. Qwen/Gemma historical token-ID arrays were not saved; their original encoder is independently checked against direct native tokenization. Ministral also matches the saved token-ID hashes. Qwen/Gemma use FP16 with eager attention on V100-32GB GPUs; Ministral uses the official FP8 checkpoint dequantized to BF16 with eager attention on an H100. Qwen/Gemma's original generation used batches of four with left padding; these prompt-only reconstructions use one unpadded input at a time. They are newly reconstructed prompt states, not historical activations saved during generation.

The main comparison is baseline → evidence contract; baseline → transparency and transparency → evidence contract are secondary. Cases where a saved unsupported-success or invented-detail label disappears are compared with different tasks free of **both** saved flags under both policies, matched within the same failure category and pressure type. Missing controls and unresolved labels remain visible. The report separates angular movement, relative L2 movement, direction coherence, and input-length differences. Small matched groups, different task content, policy wording, and position changes prevent causal or mechanism claims; layer rankings and bootstrap intervals are descriptive.

The largest main-pair contrasts occur at Qwen block 36 for unsupported success and block 24 for invented details, Gemma block 38 for both, and Ministral block 34 for both. Qwen has only two matched success cases. Gemma and Ministral show less movement than their controls at these peaks; movement directions remain strongly aligned with controls. These candidates do not isolate a false-reporting mechanism.

[Figure](results/internal_states.png) · [Layer table](results/internal_states.csv) · [Report with complete saved input/output examples](results/internal_states.txt) · [Analysis manifest](results/internal_states.json)

Keep vectors and extraction provenance in an external cache. On allocated GPU nodes with each model's recorded environment, run extraction separately:

```sh
python3 -B -m pipelines.internal_states extract --model qwen3_8b --cache-dir /external/fta_internal_state_cache
python3 -B -m pipelines.internal_states extract --model gemma2_9b --cache-dir /external/fta_internal_state_cache
python3 -B -m pipelines.internal_states extract --model ministral3_8b --cache-dir /external/fta_internal_state_cache
python3 -B -m pipelines.internal_states analyze --cache-dir /external/fta_internal_state_cache
```

Complete, verified caches are reused. Publish the four compact report artifacts with the code; keep vectors outside Git. This isolated entry point leaves the existing replay pipelines, saved inputs/answers/labels, and Nemotron's 18 frozen submission files, environment, and queued job unchanged.

## Small during-answer study

The follow-up uses a fixed exploratory subset selected from the existing labels before any new activations: five Qwen, six Gemma and ten Ministral error-disappeared tasks, plus ten matched clean controls per model. Unsupported-success and invented-detail memberships stay separate. All three original policies are included. [The frozen plan](config/during_answer.json) records the selected IDs, copied annotation spans, actual supported baseline prefixes, missing controls and exclusions.

Causal teacher forcing reconstructs each saved answer. Answer token `j` is predicted from block output at `P + j - 1`, so the first token uses the final input position. Ministral reuses saved output IDs; Qwen/Gemma retokenize saved visible text because historical output IDs were not saved. Token offsets and next-token probabilities describe supplied words, not factual truth. Natural comparisons align annotated claims/admissions and remain descriptive because preceding words differ.

A separate test appends each eligible task's own supported baseline beginning, using identical prefix IDs under the three policies and leaving the assistant turn unfinished. It generates one greedy continuation per policy, with each model's original tail-token limit. These new tails receive one contextual Codex-assisted review; original labels are not transferred. Supplied failure admissions are part of the input, so they cannot establish newly generated honesty. Prefilling baseline prose can also conflict with the contract's requested reporting format.

Completed: **153 saved-answer reconstructions and 132 new supplied-prefix continuations**, all ending with EOS. Original FTA answers and labels were preserved. The table counts invented-detail judgments among selected case tasks with usable prefixes; denominators exclude unresolved judgments separately for each policy. All control results and unsupported-success counts are in the linked report/table.

| Model | Baseline | Transparency | Evidence contract |
| --- | --- | --- | --- |
| Qwen3-8B | 3/3 | 0/4 | 0/4 |
| Gemma-2-9b-it | 1/1 | 0/1 | 0/1 |
| Ministral 3 8B | 7/8 | 3/9 | 2/8 |

Gemma has only one prefix-eligible case. Ministral still invented auxiliary tool-capability details under the contract. Main-pair invented-detail state movement was smaller than matched controls at the largest contrasts: Qwen block 36, Gemma 38 and Ministral 25. Qwen's unsupported-success contrast has one case; Gemma has none. These associations leave instruction, position and format effects unresolved. Natural plots align actual annotated spans, with onset set to zero by definition, and retain different preceding words.

**46** combined outputs exactly reproduced saved same-policy answers. **Two** old/new AI label decisions disagree despite identical text and remain marked needs review, without changing either label. Raw new-review counts therefore cannot be treated as changed behavior for those records. Reviews were unblinded Codex-assisted AI judgments, not independent human annotation.

[During-answer figure](results/during_answer.png) · [Results table](results/during_answer.csv) · [Plain-language report and exact prefixes/continuations](results/during_answer.txt) · [Exact token records and new judgments](results/during_answer.json)

The existing internal-state pipeline adds three commands without changing replay helpers or Nemotron's frozen dependencies:

```sh
python3 -B -m pipelines.internal_states during-replay --model qwen3_8b --cache-dir /external/fta_during_answer_cache
python3 -B -m pipelines.internal_states during-packets --model qwen3_8b --cache-dir /external/fta_during_answer_cache --output /external/qwen_review_packets.jsonl
python3 -B -m pipelines.internal_states during-report --cache-dir /external/fta_during_answer_cache --reviews /external/new_reviews.jsonl
```

Run `during-replay` separately for Qwen, Gemma and Ministral on allocated GPUs. Compressed block vectors stay in three external SQLite caches. Completed records are immutable; a durable pending-generation marker prevents silently repeating a continuation after a crash. All previous FTA and prompt-only results remain intact. The prompt-only report's original source fingerprint refers to [its published commit](https://github.com/SamiraJahangiri/failure-reporting-study/commit/46c73b3c1b24bd8803b423446e807e0dbc39c6c5).


## Frozen evidence-rule × reporting-structure experiment

A separate prompting-only factorial experiment uses all 100 original FTA questions and failed-tool observations. [The frozen plan](config/factorial.json) and [400 exact logical inputs](data/factorial_inputs.jsonl) were saved before new generation. This is a new ablation, not the authors' original evidence contract.

The four systems are `00=B`, `10=B+R`, `01=B+S` and `11=B+R+S`, joined with exactly two newlines, in B/R/S order. B is the original FTA baseline verbatim. R is the supplied evidence rule; S requests STATUS, EVIDENCE, LIMITATION and NEXT_ACTION headings. S carries meaningful evidence-related content, so its effect is not purely cosmetic. Questions, tool observations and native role adapters are unchanged. No answer beginnings, evaluator information, hidden-state extraction or interventions enter generation.

The planned matrix contains 1,200 answers: 300 saved original baselines reused after native-input/settings checks, plus 900 new greedy answers. Qwen/Gemma retain batch4 FP16/eager on V100-32GB and a 160-token cap; Ministral retains native batch1 BF16 dequantization on H100 and a 2,048-token cap. New Qwen/Gemma batches interleave the three missing arms within sorted tasks; historical co-batched peers and padded widths differed. The frozen `generation_order` records this actual schedule; a less precise `batch_provenance` phrase does not override it.

Each exact original-context/answer-text unit receives one fresh, globally shuffled, metadata-hidden Codex-assisted six-label review, including reused baseline text. Identical occurrences share that judgment. Original published labels are preserved separately. Headings/style can still reveal formatting, and this is not independent human annotation.

Per-condition rates keep their own resolved denominators. Average R and S effects, interaction `r11-r10-r01+r00`, and each active arm versus baseline use the same fully resolved four-condition task set per model/metric. Twenty thousand paired task-bootstrap resamples preserve each task's four answers together. Useful help, recovery suggestions, refusal, failure category and pressure breakdowns remain descriptive. These all-failure tasks cannot establish preservation of genuine successes; inconclusive intervals do not establish equivalence.

The isolated entry point preserves Nemotron's frozen dependencies:

```sh
python3 -B -m pipelines.factorial prepare
python3 -B -m pipelines.factorial run --model qwen3_8b --cache-dir /external/fta_factorial_cache
python3 -B -m pipelines.factorial packets --cache-dir /external/fta_factorial_cache --output /external/review/packets.jsonl
python3 -B -m pipelines.factorial score --reviews /external/review/reviews.jsonl
python3 -B -m pipelines.factorial report --cache-dir /external/fta_factorial_cache --reviews /external/review/reviews.jsonl
```

Run each model separately on an allocated GPU. Raw journals, native-input checks and crash-protection ledgers stay outside Git. Completed batches are immutable; an unresolved attempt is not silently generated again. Compact answers, labels, one figure/table and exact four-condition examples will be saved under `results/factorial/` after completion.

## Layout

- `data/`: unchanged original dataset, supplied logical inputs, annotation guide, and original formatter/policy source for verification.
- `config/`: unchanged policy strings, pinned source/model metadata, archive and preservation records, scheduler settings.
- `utils/`: five helper modules for IO, input verification, native models, inference runtime, and scoring/report aggregation.
- `pipelines/`: three replay entry points for preparation, generation, and scoring/reporting, plus isolated entry points for the internal-state studies and the prompting-only factorial experiment. The isolation preserves the dependencies frozen for the submitted Nemotron job.
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
