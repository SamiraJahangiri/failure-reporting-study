# Contributing

Corrections to documentation, reproducibility helpers, or scoring are welcome. For a scoring issue, identify the response or blinded review ID, the disputed label, and the exact response span. Explain how the available tool observation supports or fails to support the claim; preserve uncertainty where the evidence does not resolve it.

Preserve the published replay's frozen inputs, source hashes, raw responses, original ratings, and manifests. Put new results in a new run directory, and label changed prompts, synthetic controls, model revisions, decoding, scoring methods, or task selection explicitly. A modified-input extension should not be described as the exact replay. Do not treat duplicate greedy repetitions as independent samples or describe Codex-assisted labels as human annotations.

Run the offline checks after a change:

```sh
python3 scripts/verify_publication.py
```

The public verifier requires Python 3.11+ and the standard library. Changes to frozen material require a separate, clearly identified corrected artifact and provenance; do not silently update the manifest to make an altered archive pass. Test helper changes on fresh temporary outputs. The archived summarization and finalization scripts protect existing files with exclusive writes and depend on omitted original heavy artifacts.

Keep contributions lightweight: omit model weights, checkpoints, activation tensors, credentials, and large raw CSVs. Preserve the upstream MIT notice for any copied FTA material, as explained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Reports containing `private_*` blinding identifiers may be cited by ID; those filenames describe annotation metadata.
