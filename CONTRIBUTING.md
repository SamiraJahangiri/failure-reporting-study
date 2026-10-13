# Contributing

Corrections to documentation, reproducibility helpers, or scoring are welcome. For a scoring issue, identify the response or blinded review ID, the disputed label, and the exact response span. Explain how the available tool observation supports or fails to support the claim; preserve uncertainty where the evidence does not resolve it.

This phase uses only the unchanged FTA v0.3.0 dataset, three policies, and author-formatted messages. Preserve `data/`, the original policies, and `results/preserved_exact/`. Keep model revisions and settings in `config/models.json` and their runtime manifests. Use the three pipelines and at most five modules in `utils/`; avoid task folders, timestamp folders, and duplicate exports. Do not introduce custom tasks, reminders, recovery variants, probes, or activation experiments here.

Run the offline checks after a change:

```sh
python3 -B -m pipelines.prepare
```

Preparation requires Python 3.11+ and the standard library. It checks original file hashes and all 300 messages. Do not change a source hash to make altered inputs pass. Generate one report per model/task/policy; resume only missing reports. New responses receive one full-answer rubric review, with exact quotations and uncertainty, through `pipelines/report.py`. Existing Qwen/Gemma labels are reused unchanged. Codex-assisted annotations are not independent human annotation. Preserve identical historical greedy repetitions but do not count both in comparisons.

Keep contributions lightweight: omit model weights, caches, environments, credentials, old archives, and large historical CSVs. Preserve the upstream MIT notice in `data/FTA_LICENSE`. The recoverable archive is outside the active project, as recorded in `config/archive.json`; the previous Git history also remains available.
