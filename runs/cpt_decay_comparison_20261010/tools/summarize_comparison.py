"""Summarize a pinned-checkpoint LR comparison without loading model weights.

The experiment may be a directory containing plan.json, or plan.json itself.
Paths in that plan are relative to the experiment directory. Each arm provides
evaluation_history.json, training.jsonl, probes.json, and result.json. Initial
fixed evaluation rows are checked against each other; no model loading occurs.
Partial experiments are valid and cannot produce a completed conclusion.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


GROUPS = ("seen_train", "unseen_train", "validation", "uniform_train")
GROUP_LABELS = {
    "seen_train": "Fixed early-seen training blocks",
    "unseen_train": "Fixed unseen training blocks",
    "validation": "Fixed held-out validation",
    "uniform_train": "Fixed uniform training probe",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    temporary.replace(path)


def normalize_arms(plan: dict) -> list[dict]:
    raw = plan.get("arms", [])
    if isinstance(raw, dict):
        arms = [dict(value, id=key) for key, value in raw.items()]
    else:
        arms = [dict(value) for value in raw]
    for arm in arms:
        arm.setdefault("id", arm.get("name"))
        if not arm.get("id"):
            raise ValueError("Each arm must have an id or name")
        arm.setdefault("label", arm["id"])
        arm.setdefault("run_directory", arm.get("run_dir", arm.get("output", arm["id"])))
    if not arms or len({arm["id"] for arm in arms}) != len(arms):
        raise ValueError("Plan must contain arms with distinct ids")
    return arms


def histories(run: Path, start: int, end: int) -> tuple[list[dict], list[dict]]:
    path = run / "evaluation_history.json"
    if not path.is_file():
        return [], []
    raw_rows = read_json(path)
    snapshots = {}
    for row in raw_rows:
        offset = int(row["offset"])
        step = int(row["global_step"])
        if step != start + offset or not start <= step <= end:
            raise ValueError(f"Evaluation step/offset inconsistent: {path}, {step}/{offset}")
        metrics = {}
        for group in GROUPS:
            metric = row.get(group)
            if metric is None or not math.isfinite(float(metric["loss"])):
                raise ValueError(f"Missing or nonfinite evaluation: {path}, {group}, step {step}")
            metrics[group] = metric
        if offset in snapshots and snapshots[offset]["metrics"] != metrics:
            raise ValueError(f"Conflicting duplicate evaluation at offset {offset}: {path}")
        snapshots[offset] = dict(offset=offset, step=step, metrics=metrics)
    ordered = [snapshots[offset] for offset in sorted(snapshots)]
    flattened = [dict(group=group, step=row["step"], offset=row["offset"], **row["metrics"][group])
                 for row in ordered for group in GROUPS]
    return flattened, ordered


def check_comparison(arms: list[dict], plan: dict, expected_updates: int) -> dict:
    checks = {"probe_files_checked": 0, "evaluations_checked": 0,
              "initial_loss_tolerance": 1e-6, "initial_loss_differences": {},
              "fixed_samples_consistent": True, "result_fingerprint_keys_checked": []}
    expected_probe_sha = plan.get("probes_json_sha256", plan.get("probes_sha256"))
    first_probe_sha = None
    reference_metrics = None
    reference_initial = None
    complete_results = []
    for arm in arms:
        probe = Path(arm["run_path"]) / "probes.json"
        if probe.is_file():
            actual = sha(probe)
            if expected_probe_sha is not None and actual != expected_probe_sha:
                raise ValueError(f"Probe file differs from pinned plan: {probe}")
            if first_probe_sha is not None and actual != first_probe_sha:
                raise ValueError(f"Probe files differ across arms: {probe}")
            first_probe_sha = actual
            checks["probe_files_checked"] += 1
        elif arm["snapshots"]:
            raise ValueError(f"Evaluation history has no fixed probe file: {probe}")
        for snapshot in arm["snapshots"]:
            checks["evaluations_checked"] += 1
            if reference_metrics is None:
                reference_metrics = snapshot["metrics"]
            for group in GROUPS:
                for key in ("supervised_tokens", "input_tokens", "blocks"):
                    expected = reference_metrics[group].get(key)
                    if expected is None or snapshot["metrics"][group].get(key) != expected:
                        raise ValueError(f"Evaluation size differs: {arm['id']}/{group}/{key}")
            if snapshot["offset"] == 0:
                if ("expected_initial_validation" in plan
                        and abs(snapshot["metrics"]["validation"]["loss"] - plan["expected_initial_validation"])
                        > float(plan.get("initial_validation_tolerance", 1e-6))):
                    raise ValueError(f"Initial validation differs from pinned checkpoint: {arm['id']}")
                if reference_initial is None:
                    reference_initial = snapshot["metrics"]
                differences = {group: snapshot["metrics"][group]["loss"] - reference_initial[group]["loss"]
                               for group in GROUPS}
                if any(abs(value) > 1e-6 for value in differences.values()):
                    raise ValueError(f"Initial checkpoint evaluations differ: {arm['id']}, {differences}")
                checks["initial_loss_differences"][arm["id"]] = differences
        result = arm.get("result")
        if result and result.get("status") == "complete":
            complete_results.append(result)
            if (int(result.get("start_step", -1)) != arm["start_step"]
                    or int(result.get("end_step", -1)) != arm["end_step"]
                    or int(result.get("optimizer_updates_applied", -1)) != expected_updates):
                raise ValueError(f"Complete arm counters differ from pinned plan: {arm['id']}")
            if result.get("production_files_modified") is not False:
                raise ValueError(f"Arm does not attest unchanged production files: {arm['id']}")
            if result.get("probes_sha256") != first_probe_sha:
                raise ValueError(f"Arm result probe hash differs from file: {arm['id']}")
            if result.get("evaluation_history") != read_json(Path(arm["run_path"]) / "evaluation_history.json"):
                raise ValueError(f"Arm result/history disagree: {arm['id']}")
            offsets = {row["offset"] for row in arm["snapshots"]}
            if 0 not in offsets or expected_updates not in offsets:
                raise ValueError(f"Complete arm lacks initial/end fixed evaluation: {arm['id']}")
            expected_offsets = set(plan.get("evaluation_offsets", (0, expected_updates)))
            if offsets != expected_offsets:
                raise ValueError(f"Complete arm evaluation offsets differ from plan: {arm['id']}")
            rows = arm["training_rows"]
            expected_steps = list(range(arm["start_step"] + 1, arm["end_step"] + 1))
            if [int(row["step"]) for row in rows] != expected_steps:
                raise ValueError(f"Complete arm has incomplete or reordered training log: {arm['id']}")
            if (sum(int(row["input_tokens"]) for row in rows) != int(result.get("new_input_tokens", -1))
                    or sum(int(row["supervised_tokens"]) for row in rows) != int(result.get("new_supervised_tokens", -1))):
                raise ValueError(f"Logged token counts differ from complete result: {arm['id']}")
            if ("new_tokens_per_arm" in plan
                    and int(result["new_input_tokens"]) != int(plan["new_tokens_per_arm"])):
                raise ValueError(f"Input token budget differs from pinned plan: {arm['id']}")
    checks["probes_json_sha256"] = first_probe_sha
    identity_keys = ("initial_optimizer_tensor_sha256", "initial_rng", "checkpoint_files",
                     "checkpoint_training_state_file", "selected_block_indices_sha256",
                     "training_token_and_label_values_sha256", "checked_probe_hashes", "probes_sha256")
    for key in identity_keys:
        if any(key not in value or value[key] is None for value in complete_results):
            raise ValueError(f"Missing cross-arm identity field: {key}")
        if complete_results and any(value[key] != complete_results[0][key] for value in complete_results[1:]):
            raise ValueError(f"Cross-arm identity field differs: {key}")
        if len(complete_results) > 1:
            checks["result_fingerprint_keys_checked"].append(key)
    return checks


def training_rows(run: Path, start: int, end: int) -> list[dict]:
    path = run / "training.jsonl"
    if not path.is_file():
        return []
    result = {}
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # A running trainer may not have finished its last line.
            raise
        step = int(row["step"])
        if start < step <= end:
            if int(row["offset"]) != step - start:
                raise ValueError(f"Training step/offset inconsistent: {path}, step {step}")
            if not math.isfinite(float(row["loss"])):
                raise ValueError(f"Nonfinite training metric: {path}, step {step}")
            result[step] = row
    return [result[step] for step in sorted(result)]


def bin_training(rows: list[dict], start: int, window: int) -> list[dict]:
    buckets = {}
    for row in rows:
        buckets.setdefault((int(row["step"]) - start - 1) // window, []).append(row)
    result = []
    for group in buckets.values():
        weights = [int(row.get("supervised_tokens", row.get("input_tokens", 1))) for row in group]
        total = sum(weights)
        result.append({"first_step": int(group[0]["step"]), "last_step": int(group[-1]["step"]),
                       "updates": len(group), "supervised_tokens": total,
                       "step": sum(int(row["step"]) * weight for row, weight in zip(group, weights)) / total,
                       "loss": sum(float(row["loss"]) * weight for row, weight in zip(group, weights)) / total})
    return result


def save_figure(figure: Any, destination: Path) -> list[str]:
    paths = []
    for suffix in ("png", "svg", "pdf"):
        path = destination.with_suffix("." + suffix)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path))
    return paths


def plots(report: Path, arms: list[dict], common: dict | None, start: int,
          tokens_per_update: int) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.20})
    output = []
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for axis, group in zip(axes.flat, GROUPS):
        if common:
            axis.scatter([0], [common["metrics"][group]["loss"]], color="black", s=32,
                         label="Common checkpoint", zorder=4)
        for index, arm in enumerate(arms):
            snapshots = arm["snapshots"]
            if not snapshots:
                continue
            xs = [row["offset"] * tokens_per_update / 1e9 for row in snapshots]
            ys = [row["metrics"][group]["loss"] for row in snapshots]
            axis.plot(xs, ys, marker="o", color=colors(index), label=arm["label"])
            axis.annotate(f"{ys[-1]:.6f}", (xs[-1], ys[-1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set(title=GROUP_LABELS[group], xlabel="Additional training input tokens (B)", ylabel="Cross-entropy loss")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    figure.suptitle("Same-checkpoint learning-rate comparison: fixed evaluations", fontsize=12)
    output += save_figure(figure, report / "fixed_evaluation_comparison")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    batch_axis, val_axis, probe_axis, lr_axis = axes.flat
    for index, arm in enumerate(arms):
        color = colors(index)
        bins = arm["training_bins"]
        if bins:
            batch_axis.plot([(row["step"] - start) * tokens_per_update / 1e9 for row in bins],
                            [row["loss"] for row in bins], marker=".", color=color, label=arm["label"])
        for group, axis in (("validation", val_axis), ("uniform_train", probe_axis)):
            points = {int(row["step"]): row["loss"] for row in arm["history"] if row["group"] == group}
            if common:
                points[start] = common["metrics"][group]["loss"]
            if arm.get("evaluation_result"):
                points[arm["end_step"]] = arm["evaluation_result"]["metrics"][group]["loss"]
            if points:
                steps = sorted(points)
                axis.plot([(step - start) * tokens_per_update / 1e9 for step in steps],
                          [points[step] for step in steps], marker=".", color=color, label=arm["label"])
        rows = arm["training_rows"]
        if rows:
            lr_axis.plot([(int(row["step"]) - start) * tokens_per_update / 1e9 for row in rows],
                         [float(row.get("effective_fp32_lr", row["lr"])) for row in rows], color=color, label=arm["label"])
    for axis, title, ylabel in ((batch_axis, "Actual training batches: weighted window mean", "Cross-entropy loss"),
                              (val_axis, "Fixed held-out validation", "Cross-entropy loss"),
                              (probe_axis, "Fixed uniform training probe", "Cross-entropy loss"),
                              (lr_axis, "Applied global learning rate", "Learning rate")):
        axis.set(title=title, xlabel="Additional training input tokens (B)", ylabel=ylabel)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    lr_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    figure.suptitle("Same-checkpoint branch trajectories", fontsize=12)
    output += save_figure(figure, report / "training_validation_curves")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-window", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--validation-improvement", type=float, default=1e-3,
                        help="Practical loss improvement criterion, not statistical significance")
    parser.add_argument("--training-regression-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.training_window <= 0:
        parser.error("--training-window must be positive")
    if args.validation_improvement < 0 or args.training_regression_tolerance < 0:
        parser.error("Candidate criterion thresholds must be nonnegative")
    plan_path = args.experiment / "plan.json" if args.experiment.is_dir() else args.experiment
    root = plan_path.parent.resolve()
    plan = read_json(plan_path)
    report = (args.output or root / "report").resolve()
    report.mkdir(parents=True, exist_ok=True)
    start = int(plan["start_step"])
    end = int(plan.get("end_step", start + int(plan.get("steps", plan.get("updates_per_arm", plan.get("branch_steps", 6104))))))
    tokens_per_update = int(plan.get("batch_tokens", plan.get("tokens_per_update", 32768)))
    warnings = []
    common = None
    arms = normalize_arms(plan)
    for arm in arms:
        arm["start_step"] = start
        arm["end_step"] = int(arm.get("end_step", end))
        run = resolve(root, arm["run_directory"])
        arm["run_path"] = str(run)
        arm["history"], arm["snapshots"] = histories(run, start, arm["end_step"])
        arm["evaluation_result"] = next((row for row in arm["snapshots"] if row["step"] == end), None)
        initial = next((row for row in arm["snapshots"] if row["offset"] == 0), None)
        if common is None and initial is not None:
            common = initial
        result_path = run / "result.json"
        arm["result"] = read_json(result_path) if result_path.is_file() else None
        if arm["result"] is None or arm["result"].get("status") != "complete":
            warnings.append(f"Branch training is not complete: {arm['id']}")
        arm["training_rows"] = training_rows(run, start, arm["end_step"])
        arm["training_bins"] = bin_training(arm["training_rows"], start, args.training_window)
    checks = check_comparison(arms, plan, end - start)
    fixed_rows, changes, contrasts, history_rows, training_bins = [], [], [], [], []
    if common:
        for group in GROUPS:
            fixed_rows.append(dict(arm="common_start", label="Common checkpoint", step=start, additional_tokens=0,
                                   group=group, **common["metrics"][group]))
    for arm in arms:
        result = arm["evaluation_result"]
        if result:
            for group in GROUPS:
                metric = result["metrics"][group]
                fixed_rows.append(dict(arm=arm["id"], label=arm["label"], step=arm["end_step"],
                                       additional_tokens=(arm["end_step"] - start) * tokens_per_update,
                                       group=group, **metric))
                if common:
                    before = common["metrics"][group]["loss"]
                    changes.append(dict(arm=arm["id"], group=group, start_loss=before, end_loss=metric["loss"],
                                        loss_change=metric["loss"] - before,
                                        perplexity_ratio=math.exp(metric["loss"] - before)))
        history_rows += [dict(arm=arm["id"], additional_tokens=(row["step"] - start) * tokens_per_update, **row)
                         for row in arm["history"]]
        training_bins += [dict(arm=arm["id"], **row) for row in arm["training_bins"]]
    baseline_id = plan.get("baseline_arm", arms[0]["id"])
    baseline = next((arm for arm in arms if arm["id"] == baseline_id), None)
    if baseline is None:
        raise ValueError(f"Unknown baseline_arm: {baseline_id}")
    if baseline["evaluation_result"]:
        for arm in arms:
            if arm["id"] == baseline_id or not arm["evaluation_result"]:
                continue
            if arm["end_step"] != baseline["end_step"]:
                warnings.append(f"Cannot contrast different endpoints: {arm['id']} vs {baseline_id}")
                continue
            for group in GROUPS:
                before = baseline["evaluation_result"]["metrics"][group]["loss"]
                after = arm["evaluation_result"]["metrics"][group]["loss"]
                contrasts.append(dict(arm=arm["id"], baseline_arm=baseline_id, group=group,
                                      baseline_loss=before, arm_loss=after, loss_difference=after - before,
                                      perplexity_ratio=math.exp(after - before)))
    save_csv(report / "fixed_evaluations.csv", fixed_rows,
             ["arm", "label", "step", "additional_tokens", "group", "loss", "perplexity", "supervised_tokens", "input_tokens", "blocks"])
    save_csv(report / "changes_from_start.csv", changes,
             ["arm", "group", "start_loss", "end_loss", "loss_change", "perplexity_ratio"])
    save_csv(report / "arm_contrasts.csv", contrasts,
             ["arm", "baseline_arm", "group", "baseline_loss", "arm_loss", "loss_difference", "perplexity_ratio"])
    save_csv(report / "fixed_history.csv", history_rows,
             ["arm", "group", "step", "additional_tokens", "loss", "perplexity", "supervised_tokens", "input_tokens", "blocks"])
    save_csv(report / "training_window_means.csv", training_bins,
             ["arm", "first_step", "last_step", "step", "updates", "supervised_tokens", "loss"])
    artifacts = [] if args.no_plots else plots(report, arms, common, start, tokens_per_update)
    complete = (bool(common) and checks["probe_files_checked"] == len(arms)
                and len(checks["initial_loss_differences"]) == len(arms)
                and all(arm["evaluation_result"] is not None and arm["result"]
                        and arm["result"].get("status") == "complete" for arm in arms))
    criteria = {"evaluated": complete, "validation_improvement_threshold": args.validation_improvement,
                "seen_unseen_regression_tolerance": args.training_regression_tolerance,
                "interpretation": "A practical candidate criterion for this local comparison; not statistical significance.",
                "by_arm": {}}
    for arm in arms:
        if arm["id"] == baseline_id:
            continue
        differences = {row["group"]: row["loss_difference"] for row in contrasts if row["arm"] == arm["id"]}
        initial_changes = {row["group"]: row["loss_change"] for row in changes if row["arm"] == arm["id"]}
        criteria["by_arm"][arm["id"]] = {
            "criteria_clear": (complete and differences.get("validation", math.inf) <= -args.validation_improvement
                               and initial_changes.get("validation", math.inf) <= -args.validation_improvement
                               and all(differences.get(group, math.inf) <= args.training_regression_tolerance
                                       and initial_changes.get(group, math.inf) <= args.training_regression_tolerance
                                       for group in ("seen_train", "unseen_train"))),
            "loss_differences_vs_constant": differences,
            "loss_changes_from_common_start": initial_changes,
        }
    summary = {
        "status": "complete" if complete else "partial", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path.resolve()), "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
        "start_step": start, "end_step": end, "tokens_per_update": tokens_per_update,
        "checks": checks, "fixed_evaluations": fixed_rows, "changes_from_start": changes,
        "arm_contrasts": contrasts, "candidate_criteria": criteria, "artifacts": artifacts, "warnings": warnings,
        "arm_progress": {arm["id"]: {"training_rows": len(arm["training_rows"]),
                         "latest_logged_step": max((int(row["step"]) for row in arm["training_rows"]), default=None),
                         "fixed_evaluation_complete": bool(arm["evaluation_result"])} for arm in arms},
        "limit": "One matched short branch comparison estimates the local effect of the LR schedule. It does not establish the cause of all earlier loss rebound or the effect of a full 10B-token schedule. Lower LR also scales the AdamW decay update.",
    }
    save_json(report / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "report": str(report), "checks": checks,
                      "arm_contrasts": contrasts, "warnings": warnings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
