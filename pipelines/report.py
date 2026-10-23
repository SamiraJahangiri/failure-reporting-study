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


def workflow(root: Path, output_dir: Path | None = None):
    """Draw a methods schematic only; never generate, rescore, or replay states."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ink, muted, line = "#182b45", "#53657a", "#b7c6d6"
    blue, green, orange, purple = "#4c78a8", "#1b9e77", "#d49500", "#7560a5"
    outputs = output_dir or root / "figures"
    outputs.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 14,
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(12.6, 6.05))
        ax.set(xlim=(0, 14), ylim=(0, 6.8))
        ax.set_axis_off()
        fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985)

        def text(x, y, value, size=14, color=ink, weight="normal", ha="center"):
            if "\n" in value:
                raise ValueError("Workflow labels must be single-line")
            return ax.text(x, y, value, fontsize=size, color=color,
                           fontweight=weight, ha=ha, va="center")

        def box(x, y, w, h, face="white", edge=line, lw=1, radius=.12):
            patch = FancyBboxPatch((x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=face, edgecolor=edge, linewidth=lw)
            ax.add_patch(patch)
            return patch

        nodes = []

        def node(x, y, w, h, label, face="white", edge=line, color=ink,
                 weight="normal", size=14):
            patch = box(x, y, w, h, face, edge, lw=1.2)
            label_artist = text(x+w/2, y+h/2, label, size, color, weight)
            nodes.append((patch, label_artist))

        def arrow(start, end, color=muted):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                mutation_scale=12, color=color, linewidth=1.3,
                shrinkA=2, shrinkB=2))

        # The common input feeds independent analyses, not a live tool loop.
        node(.25, 5.96, 13.5, .58,
             "100 tasks with supplied tool-failure observations",
             face="#f2f6fb", edge="#9bb0c8", weight="bold", size=16)
        text(7, 5.61, "Qwen3-8B · Gemma-2-9b-it · Ministral 3 8B", 12, muted)
        panels = [( .25, blue, "#f4f7fc", "(a) Reporting behavior"),
                  (4.85, green, "#f2f9f7", "(b) Internal-state analysis"),
                  (9.45, purple, "#f7f5fb", "(c) Prompt components")]
        for x, color, face, title in panels:
            box(x, .23, 4.3, 4.8, face, edge="#d0dbe5", lw=1)
            text(x+.24, 4.68, title, 14, color, "bold", ha="left")
            arrow((x+2.15, 5.34), (x+2.15, 5.05), color)

        # A: unchanged original policies, reports, and existing outcome labels.
        x, c = .25, 2.4
        text(c, 4.2, "Reporting instructions", 13, weight="bold")
        for y, label, color in [(3.45, "Baseline", blue),
                                (2.78, "Transparency", green),
                                (2.11, "Evidence contract", orange)]:
            node(x+.33, y, 3.22, .5, label, edge=color, color=color)
            ax.plot([x+3.55, x+3.93], [y+.25]*2, color=muted, lw=1.2)
        # A common collector makes the three instructions alternatives.
        ax.plot([x+3.93, x+3.93, c], [3.7, 1.94, 1.94], color=muted, lw=1.2)
        arrow((c, 1.94), (c, 1.68))
        node(x+.33, 1.08, 3.64, .6, "Model answers", weight="bold")
        arrow((c, 1.08), (c, .9))
        node(x+.33, .3, 3.64, .6, "Reporting errors",
             face="#fff4f3", edge="#dfb5b0", color="#a34840", weight="bold")

        # B: prompt-only reconstructed states, with two analyses of that cache.
        x, c = 4.85, 7
        text(c, 4.2, "Same reporting instructions as (a)", 12, muted)
        node(x+.33, 3.15, 3.64, .7, "States before answering",
             face="#e9f5f0", edge="#90bcae", weight="bold")
        ax.plot([c, c, x+.27, x+.27], [3.15, 2.85, 2.85, 1.04], color=green, lw=1.2)
        node(x+.65, 2.0, 3.43, .64, "Layerwise state differences")
        node(x+.65, .72, 3.43, .64, "Final-block PCA")
        arrow((x+.27, 2.32), (x+.65, 2.32), green)
        arrow((x+.27, 1.04), (x+.65, 1.04), green)

        # C: a separate new 2x2 prompting experiment, not a state intervention.
        x, c = 9.45, 11.6
        text(c, 4.2, "Alternative reporting instructions", 12, muted)
        for row, labels in enumerate((("Baseline", "Rule only"),
                                      ("Headings only", "Rule + headings"))):
            for col, label in enumerate(labels):
                node(x+.1+col*2.1, 3.23-row*.9, 2.0, .62, label,
                     edge="#b7accb", color=purple)
        # A group bracket, with no arrows between the four conditions.
        ax.plot([x+.1, x+.1, x+4.2, x+4.2], [2.15, 2.02, 2.02, 2.15], color=muted, lw=1.2)
        arrow((c, 2.02), (c, 1.68))
        node(x+.24, 1.08, 3.82, .6, "Model answers", weight="bold")
        arrow((c, 1.08), (c, .9))
        node(x+.24, .3, 3.82, .6, "Compare reporting errors", weight="bold")

        # Check label fit in the actual renderer rather than wrapping text.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for patch, label_artist in nodes:
            boundary = patch.get_window_extent(renderer)
            label_bounds = label_artist.get_window_extent(renderer)
            if not (boundary.x0+5 < label_bounds.x0 and label_bounds.x1 < boundary.x1-5
                    and boundary.y0 < label_bounds.y0 and label_bounds.y1 < boundary.y1):
                raise ValueError("Workflow label needs a wider box: " + label_artist.get_text())
        for ext in ("pdf", "png"):
            fig.savefig(outputs / f"workflow.{ext}", dpi=300, facecolor="white")
        plt.close(fig)

    (outputs / "workflow_caption.txt").write_text(
        "Figure caption\n\n"
        "Study overview using the tasks, supplied failure observations, and "
        "original three reporting policies of Zhu et al. Tasks and observations "
        "stay fixed; no live tools or recovery actions are executed. (a) Reporting "
        "errors are unsupported success claims (completion without tool evidence) "
        "or invented details (unsupported requested or auxiliary facts). "
        "(b) Exploratory before-answer states support layerwise comparisons and "
        "final-block PCA under the same three instructions. (c) Four alternative "
        "instructions test an evidence rule requiring supported claims and "
        "headings that organize the response.\n\n"
        "Citation note: Use the existing Zhu et al. bibliography entry in "
        "paper.tex; its key cannot be confirmed without the manuscript.\n\n"
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
    return {"diagram": str(outputs / "workflow.png"), "vector": str(outputs / "workflow.pdf"),
            "caption": str(outputs / "workflow_caption.txt"), "new_experiments": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    old = commands.add_parser("reuse"); old.add_argument("--source", type=Path)
    packets = commands.add_parser("packets"); packets.add_argument("--output", type=Path, required=True)
    imported = commands.add_parser("score"); imported.add_argument("--reviews", type=Path, required=True)
    commands.add_parser("report")
    diagram = commands.add_parser("workflow", help="Render the paper's study overview only")
    diagram.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root, results = args.root.resolve(), args.root.resolve() / "results"
    if args.command == "workflow":
        print(json.dumps(workflow(root, args.output_dir))); return
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
