FTA original-task handoff

Start with codex_prompt.txt. This package includes every one of the authors' 100
released v0.3.0 task records, unchanged, plus all three policy strings and 300
ready-to-send logical inputs. Only system and user in model_inputs.jsonl belong
in model context. The IDs and all evaluator fields are outside model context.

The task set has 20 neutral requests and 80 requests across four pressure types.
Every task lacks required successful tool evidence. The model chooses its reply;
the code must not force it to claim success or failure.

No model generations or research results were produced while building this
package. The original policy/formatter files and annotation guide are included
under upstream, along with their MIT license and copyright notice. The task
dataset is an exact byte copy. model_inputs.jsonl was produced by the original
formatter function bodies and condition definitions; only the surrounding
provider imports were omitted for offline construction.

Native tokenization, system-role support, decoding, and local-model settings
remain model-specific. These logical messages do not imply identical tokenized
inputs to the API models studied by the authors. The public task release and
the later paper also differ in their description of scoring provenance.
