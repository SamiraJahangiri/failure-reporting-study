#!/usr/bin/env python3
"""Preserve old labels, import one new review, and report original FTA results.

Examples:
    python pipelines/report.py reuse --source results/preserved_exact
    python pipelines/report.py packets --output /path/outside-project/packets.jsonl
    python pipelines/report.py score --reviews /path/outside-project/reviews.jsonl
    python pipelines/report.py report
    python pipelines/report.py workflow
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset import load_and_verify_inputs, load_scenarios
from utils.io import atomic_write_json, atomic_write_jsonl, exclusive_write_json, exclusive_write_jsonl, read_json, read_jsonl, sha256_file
from utils import scoring

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("baseline", "transparency", "evidence_contract")
OLD_MODELS = ("Qwen/Qwen3-8B", "google/gemma-2-9b-it")
LABEL_FIELDS = ("response_id", "scenario_id", "model", "condition", "repeat",
    "rubric_labels", "diagnostics", "wrong_completion_header", "evidence_spans",
    "confidence", "uncertainty", "notes", "unresolved_labels", "primary_reviewer",
    "secondary_reviewer", "adjudicated", "scoring_provenance", "blind_id",
    "reviewed_equivalent_blind_id", "equivalent_review_rule")


def read_if_exists(path):
    return read_jsonl(path) if Path(path).exists() else []


def reuse(root: Path, source: Path):
    """Copy repeat 0 only, preserving every saved judgment field exactly."""
    raw_path, scored_path = source / "raw_responses.jsonl", source / "scored_reports.jsonl"
    raw, scored = read_jsonl(raw_path), read_jsonl(scored_path)
    raw_map, scored_map = scoring.index_rows(raw), scoring.index_rows(scored)
    if len(raw) != 1200 or len(scored) != 1200 or set(raw_map) != set(scored_map):
        raise ValueError("The complete preserved exact run must contain 1,200 raw and scored records")
    chosen, labels = [], []
    identities = set()
    for row in raw:
        if row["model"] not in OLD_MODELS or row["repeat"] != 0:
            continue
        rid = row["response_id"]
        scored_row = scored_map[rid]
        if any(scored_row[k] != v for k, v in row.items()):
            raise ValueError("Scoring source changed a raw response field: " + rid)
        other = raw_map[rid.rsplit("|", 1)[0] + "|1"]
        if row["response"] != other["response"]:
            raise ValueError("Historical greedy repetitions are not identical: " + rid)
        identity = (row["model"], row["scenario_id"], row["condition"])
        if identity in identities:
            raise ValueError("Duplicate historical input")
        identities.add(identity)
        answer = dict(row, source_kind="reused_exact_repeat_0",
                      preserved_source="results/preserved_exact/raw_responses.jsonl")
        label = {k: scored_row[k] for k in LABEL_FIELDS}
        label.update(response_sha256=scoring.response_hash(row["response"]),
                     source_kind="reused_saved_labels_unchanged",
                     preserved_source="results/preserved_exact/scored_reports.jsonl")
        chosen.append(answer)
        labels.append(label)
    if len(chosen) != 600:
        raise ValueError("Expected 300 reused reports per model")
    results = root / "results"
    existing_answers, existing_labels = read_if_exists(results / "answers.jsonl"), read_if_exists(results / "labels.jsonl")
    answer_map, label_map = scoring.index_rows(existing_answers), scoring.index_rows(existing_labels)
    for row in chosen:
        if row["response_id"] in answer_map and answer_map[row["response_id"]] != row:
            raise ValueError("Existing consolidated answer differs")
        answer_map[row["response_id"]] = row
    for row in labels:
        if row["response_id"] in label_map and label_map[row["response_id"]] != row:
            raise ValueError("Existing consolidated saved label differs")
        label_map[row["response_id"]] = row
    atomic_write_jsonl(results / "answers.jsonl", sorted(answer_map.values(), key=lambda r: r["response_id"]))
    atomic_write_jsonl(results / "labels.jsonl", sorted(label_map.values(), key=lambda r: r["response_id"]))
    atomic_write_json(results / "reuse_manifest.json", {
        "preserved_raw_sha256": sha256_file(raw_path), "preserved_scored_sha256": sha256_file(scored_path),
        "historical_reports_preserved": 1200, "comparison_reports_selected": 600,
        "selection": "repeat 0, 100 unchanged tasks × 3 original policies per model",
        "identical_greedy_pairs_verified": 600, "new_scoring_of_old_answers": False,
        "old_max_new_tokens": 160, "new_repetitions_per_model_task_policy": 1})
    return {"reused_answers": len(chosen), "saved_labels_copied_without_changes": len(labels)}


def plot(summary: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    counts = {(g["model"], g["condition"]): g for g in summary["summary_counts"]}
    models = sorted({k[0] for k in counts}, key=lambda x: (x not in OLD_MODELS, x))
    metric_names = ("Unsupported success", "Invented details", "Failure disclosed",
                    "Recovery step", "Useful answer", "Unneeded refusal")
    colors = ("#3b6586", "#299485", "#d38b3e")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.1), sharey=True)
    for ax, metric, title in zip(axes.flat, scoring.RUBRIC_LABELS, metric_names):
        for j, (policy, color) in enumerate(zip(POLICIES, colors)):
            for i, model in enumerate(models):
                g = counts.get((model, policy))
                value = g["labels"][metric]["rate"] if g else None
                if value is None:
                    continue
                x = i + (j - 1) * .24
                ax.bar(x, value, width=.22, color=color,
                       label=policy.replace("_", " ") if i == 0 else None, zorder=3)
                ax.text(x, value + .018, f"{value:.0%}", ha="center", fontsize=6)
        ax.set_title(title, fontweight="bold")
        def short_name(model):
            if "Nemotron" in model: return "Nemotron 3 Super"
            if "Ministral" in model: return "Ministral 3 8B"
            if "Llama-3.1" in model: return "Llama 3.1 8B"
            return model.split("/")[-1]
        ax.set_xticks(range(len(models)), [short_name(m) for m in models], rotation=24, ha="right", fontsize=6.7)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color="#e4e9ed", linewidth=.6, zorder=0)
    axes[0, 0].set_ylabel("Share of valid, resolved answers")
    axes[1, 0].set_ylabel("Share of valid, resolved answers")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .95))
    fig.suptitle("Unchanged FTA v0.3.0 tasks — original rubric", fontweight="bold", y=.992)
    valid_n = sum(g["valid_n"] for g in summary["summary_counts"])
    truncated_n = sum(g["truncated_n"] for g in summary["summary_counts"])
    fig.text(.5, .015,
        f"{summary['generated_reports']} answers, {valid_n} full scored answers, {truncated_n} truncated outputs excluded. One answer per model/task/policy.\n"
        "Flags overlap; confidence-1 labels excluded individually. Recovery = suggested next action; no tool recovery executed.\n"
        "Qwen/Gemma labels reused; new labels scored once with Codex, not human annotation.",
        ha="center", fontsize=7, color="#526170")
    fig.subplots_adjust(top=.86, bottom=.16, hspace=.62, wspace=.18, left=.06, right=.985)
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)


def examples(root: Path, answers: list[dict], labels: list[dict]) -> str:
    inputs = {(i["scenario_id"], i["condition"]): i for i in load_and_verify_inputs(root)}
    label_map = scoring.index_rows(labels)
    answer_map = {(a["model"], a["scenario_id"], a["condition"]): a for a in answers}
    models = sorted({a["model"] for a in answers})
    lines = ["Actual saved FTA examples\n",
             "Selected illustrations, not a representative sample. Text between delimiters is complete; delimiters add display newlines only.\n",
             "Recovery-step labels concern suggested next actions. This replay supplies fixed tool observations; it does not execute tool recoveries.\n",
             "Original logical messages are copied verbatim from data/model_inputs.jsonl. Native rendering hashes/adapter information are recorded with answers and generation manifests.\n"]
    for model in models:
        candidates = [a for a in answers if a["model"] == model and a["condition"] == "baseline" and a["response_id"] in label_map]
        if not candidates:
            continue
        candidates.sort(key=lambda a: (not label_map[a["response_id"]]["diagnostics"]["fabricated_requested_facts"], a["scenario_id"]))
        task = candidates[0]["scenario_id"]
        for policy in ("baseline", "evidence_contract"):
            row = answer_map.get((model, task, policy))
            if not row or row["response_id"] not in label_map:
                continue
            item, label = inputs[(task, policy)], label_map[row["response_id"]]
            lines.extend([f"\nModel: {model}\nPolicy: {policy}\nTask: {task}\nResponse: {row['response_id']}\n",
                          "Source: results/answers.jsonl; results/labels.jsonl; data/model_inputs.jsonl\n",
                          f"Source kind: {row.get('source_kind', 'new_generation')}\n",
                          "--- EXACT SYSTEM INSTRUCTION ---\n", item["system"], "\n--- END SYSTEM ---\n",
                          "--- EXACT USER MESSAGE INCLUDING OBSERVATION ---\n", item["user"], "\n--- END USER ---\n",
                          "--- COMPLETE SAVED MODEL ANSWER ---\n", row["response"], "\n--- END ANSWER ---\n",
                          "Saved labels: " + json.dumps(scoring.label_values(label), ensure_ascii=False, sort_keys=True) + "\n",
                          "Confidence: " + json.dumps(label["confidence"], sort_keys=True) + "\n",
                          "Exact supporting spans: " + json.dumps(label["evidence_spans"], ensure_ascii=False, sort_keys=True) + "\n",
                          "Review notes:" + (" " + label["notes"] if label.get("notes") else "") + "\n"])
    return "".join(lines)


def add_run_settings(root: Path, summary: dict, answers: list[dict]):
    """List missing models explicitly; missing runs are never zero-rate bars."""
    configuration = read_json(root / "config/models.json")
    status = []
    for key, spec in configuration["models"].items():
        rows = [a for a in answers if a["model"] == spec["repo_id"]]
        blocker_path = root / "results" / ("blocker_" + key + ".json")
        blocker = read_json(blocker_path) if blocker_path.exists() else None
        state = "complete" if len(rows) == 300 else "partial" if rows else "blocked" if blocker else "not_run"
        status.append({"model": spec["repo_id"], "revision": spec["revision"], "mode": spec["mode"],
                       "planned_reports": 300, "saved_reports": len(rows), "status": state,
                       "max_new_tokens_in_saved_answers": sorted({a["max_new_tokens"] for a in rows}),
                       "repetitions_in_comparison": sorted({a["repeat"] for a in rows}),
                       "template_kwargs": spec.get("template_kwargs", {}), "blocker": blocker})
    summary.update(model_status=status, new_generation_settings=configuration["new_generation"],
                   planned_new_reports=900, actual_new_reports=sum(s["saved_reports"] for s in status if s["mode"] == "generate_new"),
                   planned_comparison_reports=1500,
                   missing_model_rule="No checkpoint substitution. Models with no saved answer have no rate and are absent from the figure.")


def workflow(root: Path):
    """Draw a methods schematic only; never generate, rescore, or replay states."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ink, muted, line = "#182b45", "#53657a", "#b7c6d6"
    blue, green, orange, purple = "#4c78a8", "#1b9e77", "#d49500", "#7560a5"
    outputs = root / "results"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(12, 6.85))
        ax.set(xlim=(0, 14), ylim=(0, 8))
        ax.set_axis_off()
        fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985)

        def text(x, y, value, size=12, color=ink, weight="normal", ha="center"):
            return ax.text(x, y, value, fontsize=size, color=color,
                           fontweight=weight, ha=ha, va="center", linespacing=1.4)

        def box(x, y, w, h, face="white", edge=line, lw=1, radius=.12):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=face, edgecolor=edge, linewidth=lw))

        def arrow(start, end, color=muted):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                mutation_scale=12, color=color, linewidth=1.3,
                shrinkA=2, shrinkB=2))

        # The common input feeds independent analyses, not a live tool loop.
        box(.25, 6.82, 13.5, .94, face="#f2f6fb", edge="#9bb0c8", lw=1.2)
        text(7, 7.42, "100 unchanged FTA tasks + supplied failed-tool observations", 14, weight="bold")
        text(7, 7.05, "FTA v0.3.0  ·  Qwen3-8B  ·  Gemma-2-9b-it  ·  Ministral 3 8B  ·  Native chat templates", 10.5, muted)
        panels = [( .25, blue, "#f4f7fc", "(a) Reporting behavior"),
                  (4.85, green, "#f2f9f7", "(b) Internal-state analysis"),
                  (9.45, purple, "#f7f5fb", "(c) Prompt components")]
        for x, color, face, title in panels:
            box(x, .23, 4.3, 6.05, face, edge="#d0dbe5", lw=1)
            text(x+.24, 5.94, title, 13, color, "bold", ha="left")
            arrow((x+2.15, 6.82), (x+2.15, 6.3), color)

        # A: unchanged original policies, reports, and existing outcome labels.
        x, c = .25, 2.4
        text(c, 5.43, "Original instruction policies", 11.5, weight="bold")
        for y, label, color in [(4.67, "Baseline", blue),
                                (3.99, "Transparency", green),
                                (3.31, "Evidence contract", orange)]:
            box(x+.42, y, 3.46, .52, edge=color, lw=1.2)
            text(c, y+.26, label, 12, color)
        arrow((c, 3.31), (c, 2.91))
        box(x+.42, 2.25, 3.46, .64)
        text(c, 2.57, "Model answers", 12, weight="bold")
        arrow((c, 2.25), (c, 1.91))
        text(c, 1.68, "Codex-assisted rubric review", 11, muted)
        box(x+.42, .91, 3.46, .48, "#fff4f3", "#dfb5b0")
        text(c, 1.15, "Unsupported success", 11.5, "#a34840")
        box(x+.42, .34, 3.46, .48, "#fff8ea", "#dcc48c")
        text(c, .58, "Invented details", 11.5, "#916214")

        # B: prompt-only reconstructed states, with two analyses of that cache.
        x, c = 4.85, 7
        box(x+.42, 4.48, 3.46, .93)
        text(c, 5.08, "Replay each original input", 11.5, weight="bold")
        text(c, 4.74, "Before any answer token", 11, muted)
        arrow((c, 4.48), (c, 4.09))
        box(x+.42, 3.15, 3.46, .92, "#e9f5f0", "#90bcae")
        text(c, 3.81, "Last input token", 12, weight="bold")
        text(c, 3.5, "Every transformer block", 10.5, muted)
        text(c, 3.26, "Before final normalization", 9.5, muted)
        ax.plot([x+.58, x+.58], [3.15, 1.08], color=green, lw=1.2)
        box(x+.91, 1.91, 2.97, .92)
        text(x+2.395, 2.51, "Layerwise policy shifts", 11.5, weight="bold")
        text(x+2.395, 2.17, "Matched clean references", 10.5, muted)
        arrow((x+.58, 2.37), (x+.91, 2.37), green)
        box(x+.91, .55, 2.97, 1.06)
        text(x+2.395, 1.31, "Final-block PCA", 12, weight="bold")
        text(x+2.395, .96, "L2 normalization · PCA per model", 9.5, muted)
        text(x+2.395, .7, "Same tasks across three policies", 9.5, muted)
        arrow((x+.58, 1.08), (x+.91, 1.08), green)

        # C: a separate new 2x2 prompting experiment, not a state intervention.
        x, c = 9.45, 11.6
        text(c, 5.43, "2 × 2 prompting experiment", 11.5, weight="bold")
        text(c, 5.03, "B = original baseline", 10.5, muted)
        gx, gy, cw, ch = x+1.13, 3.09, 1.34, .61
        text(gx+cw/2, 4.59, "R off", 11, muted)
        text(gx+1.5*cw, 4.59, "R on", 11, muted)
        for row, label in enumerate(("S off", "S on")):
            y = gy+(1-row)*ch
            text(x+.62, y+ch/2, label, 11, muted)
            for col in range(2):
                box(gx+col*cw, y, cw, ch, "white", "#b7accb", radius=.025)
                text(gx+(col+.5)*cw, y+ch/2,
                     (("B", "B + R"), ("B + S", "B + R + S"))[row][col],
                     12, purple, "bold")
        text(c, 2.72, "R  Evidence rule", 11, weight="bold")
        text(c, 2.3, "S  Four reporting headings", 11, weight="bold")
        text(c, 1.87, "STATUS · EVIDENCE\nLIMITATION · NEXT_ACTION", 9.5, muted)
        arrow((c, 1.5), (c, 1.18))
        box(x+.42, .42, 3.46, .74, "white", "#b7accb")
        text(c, .92, "Reporting-error rates", 12, weight="bold")
        text(c, .61, "Rule effect · structure effect · interaction", 9.5, muted)

        for ext in ("svg", "png"):
            fig.savefig(outputs / f"workflow.{ext}", dpi=300, facecolor="white")
        plt.close(fig)

    (outputs / "workflow.txt").write_text(
        "Figure caption\n\n"
        "Study overview. The same 100 original FTA tasks and failed-tool "
        "observations are used with Qwen, Gemma, and Ministral. (a) Saved answers "
        "under baseline, transparency, and evidence-contract instructions are "
        "reviewed for unsupported success and invented details. (b) Prompt-only "
        "replays extract the last input token's block states before any answer. "
        "These support layerwise policy comparisons and model-specific "
        "final-block PCA views of the same tasks across policies. (c) A separate "
        "2x2 prompting experiment adds an evidence rule, four reporting headings, "
        "or both to the baseline and compares reporting errors and component "
        "effects. The state analyses are descriptive; the component experiment "
        "tests answer behavior, not hidden-state changes.\n\n"
        "Method notes\n"
        "Labels are Codex-assisted AI judgments, not independent human "
        "annotations. States are reconstructed without answer text and recorded "
        "before final normalization. Layerwise reference cases "
        "are matched on failure category and pressure type. PCA uses one fit per "
        "model, pooling all eligible three-policy task triplets after per-vector "
        "L2 normalization, with no control subtraction. Final blocks are Qwen 36, "
        "Gemma 42, and Ministral 34. Its focused view follows fixed W-W-C and C-C-C "
        "task groups; W means either saved error flag and C means neither, not "
        "guaranteed correctness. State shifts and projected distances are "
        "descriptive associations, not a causal mechanism. The factorial "
        "experiment holds tasks and observations fixed; R is the evidence rule "
        "and S is the four-heading structure. Its combined prompt is not an exact "
        "reconstruction of the original evidence contract, and the headings "
        "carry evidence-related meaning.\n\n"
        "Source records\n"
        "Original policies/tasks: data/model_inputs.jsonl; config/policies.json; "
        "results/answers.jsonl; results/labels.jsonl.\n"
        "States: config/internal_states.json; results/internal_states.txt.\n"
        "PCA: config/pca.json; results/pca.json; results/pca.txt.\n"
        "Components: config/factorial.json; results/factorial/report.txt.\n\n"
        "Presentation-only diagram. No new answers, scoring, states, PCA fits, "
        "or experiments were produced.\n", encoding="utf-8")
    return {"diagram": "results/workflow.png", "editable_vector": "results/workflow.svg",
            "caption": "results/workflow.txt", "new_experiments": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    old = commands.add_parser("reuse"); old.add_argument("--source", type=Path)
    packets = commands.add_parser("packets"); packets.add_argument("--output", type=Path, required=True)
    imported = commands.add_parser("score"); imported.add_argument("--reviews", type=Path, required=True)
    commands.add_parser("report")
    commands.add_parser("workflow", help="Render the paper's study overview only")
    args = parser.parse_args()
    root, results = args.root.resolve(), args.root.resolve() / "results"
    if args.command == "workflow":
        print(json.dumps(workflow(root))); return
    load_and_verify_inputs(root)
    if args.command == "reuse":
        source = args.source.resolve() if args.source else results / "preserved_exact"
        print(json.dumps(reuse(root, source), sort_keys=True)); return
    answers, labels = read_if_exists(results / "answers.jsonl"), read_if_exists(results / "labels.jsonl")
    scenarios = load_scenarios(root)
    if args.command == "packets":
        rows = scoring.build_packets(answers, scenarios, labels)
        if not rows:
            raise ValueError("No unscored answers; old labels must not be scored again")
        exclusive_write_jsonl(args.output, rows)
        exclusive_write_json(args.output.with_suffix(".instructions.json"), scoring.review_instructions())
        print(json.dumps({"new_single_pass_review_packets": len(rows)})); return
    if args.command == "score":
        combined = scoring.import_reviews(answers, scenarios, labels, read_jsonl(args.reviews))
        atomic_write_jsonl(results / "labels.jsonl", combined)
        print(json.dumps({"new_answers_scored_once": len(combined) - len(labels), "saved_labels_total": len(combined)})); return
    if not answers:
        raise ValueError("No saved answers to report")
    summary = scoring.summarize(answers, labels)
    add_run_settings(root, summary, answers)
    if (results / "reuse_manifest.json").exists():
        summary["reuse_manifest"] = json.loads((results / "reuse_manifest.json").read_text())
    atomic_write_json(results / "summary.json", summary)
    plot(summary, results / "summary.png")
    (results / "examples.txt").write_text(examples(root, answers, labels), encoding="utf-8")
    print(json.dumps({"reports": summary["generated_reports"], "scored": summary["scored_reports"],
                      "unscored": len(summary["unscored_response_ids"]), "summary": str(results / "summary.json")}))


if __name__ == "__main__":
    main()
