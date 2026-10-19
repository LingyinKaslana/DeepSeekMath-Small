"""Plot an ongoing CPT run and its predecessor from frozen CPU-only inputs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from plot_cpt_train_validation import input_tokens


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge_evaluations(histories, expected_tokens, manifest, label):
    rows = {}
    for history in histories:
        for item in history:
            step = int(item["step"])
            require(item["loss_backend"] == "native eager", f"Non-native {label} evaluation")
            require(math.isfinite(item["loss"]), f"Nonfinite {label} evaluation")
            for key, value in expected_tokens.items():
                require(item[key] == value, f"Changed {label} evaluation size: {key}")
            if "cumulative_tokens" in item:
                require(item["cumulative_tokens"] == input_tokens(step, manifest), f"Wrong {label} input prefix at {step}")
            if step in rows:
                require(rows[step] == item, f"Inconsistent overlapping {label} point at step {step}")
            rows[step] = item
    return rows


def planned_lr(tokens, policy):
    fraction = min(1., max(0., (tokens - policy["start_input_tokens"]) /
                              (policy["end_input_tokens"] - policy["start_input_tokens"])))
    return policy["start_lr"] + fraction * (policy["end_lr"] - policy["start_lr"])


def plot_curves(output, points, means, learning_rates, manifest, summary, zoom=False):
    blue, orange, green, muted = "#2766A3", "#D08434", "#168572", "#637384"
    policy = manifest["lr_intervention"]
    start_b, end_b = policy["start_input_tokens"] / 1e9, policy["end_input_tokens"] / 1e9
    latest_b = points[-1]["input_tokens_b"]
    training_complete = points[-1]["input_tokens"] >= manifest["configured_input_tokens"]
    xleft = 4.5 if zoom else 0.
    xright = max(latest_b + .12, end_b + .13)
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai"))
    figure, axes = plt.subplots(3, 1, figsize=(13.4, 10.8), sharex=True,
                               gridspec_kw={"height_ratios": [1.15, 1.15, .9]}, facecolor="white")
    figure.subplots_adjust(left=.09, right=.96, bottom=.15, top=.82, hspace=.37)
    figure.text(.09, .955, "数学继续预训练：学习率衰减后的曲线" + ("（近期放大）" if zoom else "（完整轨迹）"),
                fontsize=21, weight="bold", color="#152A40")
    figure.text(.09, .913, "Qwen2.5-Coder-1.5B · 全参数 / 优化版 · 5B→6B：LR 从 3e-6 线性降至 3e-7",
                fontsize=11.5, color=muted)
    figure.text(.09, .872, f"截至 {now:%m月%d日 %H:%M}（北京时间） · 共同评估 step {summary['common_evaluation_step']:,}"
                f" / {latest_b:.3f}B tokens · 固定评估未平滑", fontsize=11, color=muted)

    values = [p for p in points if p["input_tokens_b"] >= xleft]
    x = [p["input_tokens_b"] for p in values]
    axes[0].plot(x, [p["validation_loss"] for p in values], color=blue, lw=2., marker="o", ms=2.7,
                 label="固定验证集 · 2.10M tokens")
    axes[0].set_title("固定验证集 loss", loc="left", fontsize=12, pad=10)
    axes[0].set_ylabel("平均 token loss")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)
    axes[1].plot(x, [p["fixed_train_probe_loss"] for p in values], color=orange, lw=2., marker="o", ms=2.7,
                 label="固定训练集抽样 · 0.524M tokens")
    windows = [p for p in means if p["input_tokens_b"] >= xleft]
    axes[1].plot([p["input_tokens_b"] for p in windows], [p["token_weighted_batch_loss"] for p in windows],
                 color=green, lw=1.5, alpha=.85, label="实际 batch · 每 1,000 步加权均值")
    axes[1].set_title("固定训练集抽样与实际训练 batch loss", loc="left", fontsize=12, pad=10)
    axes[1].set_ylabel("平均 token loss")
    axes[1].legend(loc="upper left", ncols=1, frameon=False, fontsize=10)
    lr_values = [row for row in learning_rates if row["input_tokens_b"] >= xleft]
    axes[2].plot([row["input_tokens_b"] for row in lr_values], [row["effective_fp32_lr"] / 1e-6 for row in lr_values],
                 color=blue, lw=2, label="实测：每步实际应用 LR")
    if latest_b < end_b:
        axes[2].plot([latest_b, end_b, xright],
                     [learning_rates[-1]["effective_fp32_lr"] / 1e-6, policy["end_lr"] / 1e-6, policy["end_lr"] / 1e-6],
                     color="#8764AA", lw=1.8, ls="--", label="未来计划：降至 3e-7 后保持")
    elif not training_complete:
        axes[2].plot([latest_b, latest_b + .12],
                     [policy["end_lr"] / 1e-6, policy["end_lr"] / 1e-6],
                     color="#8764AA", lw=1.8, ls="--", label="未来计划：保持 3e-7")
    axes[2].set_title("实际应用的全局学习率", loc="left", fontsize=12, pad=10)
    axes[2].set_ylabel("LR（单位 1e-6）")
    axes[2].set_xlabel("累计训练输入 tokens（B，十亿）", labelpad=9)
    axes[2].legend(loc="upper right", frameon=False, fontsize=10)
    axes[2].set_ylim(-.08, 3.3)

    for axis in axes:
        axis.axvspan(start_b, end_b, color="#90B8DB", alpha=.13, zorder=0)
        if not training_complete and latest_b < xright:
            axis.axvspan(latest_b, xright, facecolor="#F1F3F5", alpha=.5, zorder=0)
        axis.axvline(start_b, color=blue, lw=1.1, alpha=.75)
        axis.axvline(end_b, color="#8764AA", ls="--" if latest_b < end_b else "-", lw=1.1, alpha=.8)
        if not zoom:
            warmup_b = input_tokens(manifest["schedule"]["warmup_steps"], manifest) / 1e9
            axis.axvline(warmup_b, color="#9BAAB8", ls=":", lw=1.1)
        axis.grid(axis="y", color="#E4EAF0", lw=.8)
        axis.set_axisbelow(True)
        axis.set_xlim(xleft - (.02 if zoom else .04), xright)
    axes[0].text(start_b - (.015 if zoom else .04), 1.025, "5B：开始降 LR", transform=axes[0].get_xaxis_transform(),
                 ha="right", va="bottom", fontsize=9.5, color=blue)
    axes[0].text(end_b - (.012 if zoom else .025), 1.025,
                 "6B：计划降至 3e-7" if latest_b < end_b else "6B：已降至 3e-7", transform=axes[0].get_xaxis_transform(),
                 ha="right", va="bottom", fontsize=9.5, color="#8764AA")
    if not zoom:
        axes[0].text(warmup_b + .03, 1.025, "warmup 结束", transform=axes[0].get_xaxis_transform(),
                     va="bottom", fontsize=9.5, color=muted)
        best = summary["best_validation"]
        axes[0].scatter([best["input_tokens_b"]], [best["validation_loss"]], color=blue, s=30, zorder=5)
        axes[0].annotate(f"最低 {best['validation_loss']:.4f} · {best['input_tokens_b']:.3f}B",
                         (best["input_tokens_b"], best["validation_loss"]), xytext=(18, 18),
                         textcoords="offset points", fontsize=9, color=blue,
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2},
                         arrowprops={"arrowstyle": "-", "color": blue})
    for axis, key, color in ((axes[0], "validation_loss", blue), (axes[1], "fixed_train_probe_loss", orange)):
        value = points[-1][key]
        axis.scatter([latest_b], [value], color=color, s=35, zorder=5)
        axis.annotate(f"最新 {value:.4f}", (latest_b, value), xytext=(-10, -21), textcoords="offset points",
                      ha="right", fontsize=9.5, color=color,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    axes[2].scatter([latest_b], [learning_rates[-1]["effective_fp32_lr"] / 1e-6], color=blue, s=32, zorder=5)
    axes[2].annotate(f"最新 LR {learning_rates[-1]['effective_fp32_lr']:.2e}",
                     (latest_b, learning_rates[-1]["effective_fp32_lr"] / 1e-6), xytext=(-11, 15),
                     textcoords="offset points", ha="right", fontsize=9.5, color=blue,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    figure.text(.09, .088, "蓝色浅带为 5B→6B 衰减区间；已完成全部训练，曲线截至最终固定评估。" if training_complete else
                "蓝色浅带为 5B→6B 衰减区间；灰色区域和 LR 虚线为尚未完成的计划，未预测未来 loss。",
                fontsize=10, color=muted)
    figure.text(.09, .057, "固定训练 / 验证样本不同，loss 的绝对差值不能直接解释为过拟合；batch loss 使用变化中的样本。",
                fontsize=9.5, color=muted)
    figure.text(.09, .027, "实际 batch 按预测目标数加权，仅保留完整 1,000 步窗口；三组曲线均截止最后共同固定评估点。",
                fontsize=9.5, color=muted)
    stem = "lr_transition_recent" if zoom else "lr_transition_overview"
    artifacts = []
    for extension in ("png", "svg", "pdf"):
        destination = output / f"{stem}.{extension}"
        figure.savefig(destination, dpi=180, facecolor="white")
        artifacts.append(str(destination))
    plt.close(figure)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-step", type=int, help="Keep an existing shared evaluation endpoint when redrawing")
    args = parser.parse_args()
    run, previous, output = args.run.resolve(), args.previous_run.resolve(), args.output.resolve()
    sources = []
    frozen = {}

    def payload(path):
        if path not in frozen:
            frozen[path] = path.read_bytes()
            sources.append({"path": str(path), "bytes": len(frozen[path]),
                            "sha256": hashlib.sha256(frozen[path]).hexdigest()})
        return frozen[path]

    def load(path):
        return json.loads(payload(path))

    old_manifest, manifest = load(previous / "manifest.json"), load(run / "manifest.json")
    old_result = load(previous / "result.json")
    resume_path = Path(manifest["arguments"]["resume"]).resolve()
    require(resume_path == Path(old_result["checkpoint"]).resolve(), "Run does not resume previous run's final checkpoint")
    require(resume_path.parent == previous, "Resume checkpoint outside previous run")
    checkpoint = load(resume_path / "checkpoint_manifest.json")
    resume_step = int(manifest["resumed_completed_steps"])
    require(checkpoint["completed_steps"] == old_result["completed_steps"] == resume_step, "Resume checkpoint step mismatch")
    require(checkpoint["trained_tokens"] == manifest["resumed_input_tokens"] == old_result["trained_tokens"], "Resume checkpoint token mismatch")
    require(checkpoint["signature"] == old_manifest["signature"] == manifest["original_signature"], "Checkpoint signature chain mismatch")
    require(old_manifest["order_metadata"] == manifest["order_metadata"], "Training shuffle order changed")
    old_probe, probe = load(previous / "train_probe.json"), load(run / "train_probe.json")
    require(old_probe == probe == manifest["train_probe"] == old_manifest["train_probe"] == checkpoint["train_probe"], "Fixed training probe changed")
    require(manifest["lr_intervention_audit"]["original_schedule"] == old_manifest["schedule"], "Original LR schedule mismatch")
    require(manifest["lr_intervention_audit"]["policy"] == manifest["lr_intervention"], "LR intervention audit mismatch")
    for key in ("data_manifest_sha256", "token_file_sha256", "sequence_length", "effective_batch", "epochs"):
        require(manifest["signature"][key] == old_manifest["signature"][key], f"Data/batch signature changed: {key}")
    baseline = load(previous / "validation_before.json")
    require(baseline["step"] == 0, "Previous run has no step-0 validation")
    validation = merge_evaluations([[baseline], load(previous / "validation_history.json"),
                                    load(run / "validation_history.json"), [load(run / "validation_before.json")]],
                                   {"input_tokens": 2097152, "supervised_tokens": 2096128, "blocks": 1024}, manifest, "validation")
    probes = merge_evaluations([load(previous / "train_probe_history.json"), load(run / "train_probe_history.json")],
                               {key: probe[key] for key in ("input_tokens", "supervised_tokens", "blocks")}, manifest, "train probe")
    common_steps = sorted(set(validation) & set(probes))
    require(common_steps and common_steps[0] == 0, "No common step-0 fixed evaluation")
    if args.cutoff_step is not None:
        require(args.cutoff_step in common_steps and args.cutoff_step >= resume_step,
                "Requested cutoff is not a shared evaluation after resume")
        common_steps = [step for step in common_steps if step <= args.cutoff_step]
    cutoff = common_steps[-1]
    points = [{"step": step, "input_tokens": input_tokens(step, manifest),
               "input_tokens_b": input_tokens(step, manifest) / 1e9,
               "validation_loss": validation[step]["loss"], "fixed_train_probe_loss": probes[step]["loss"]}
              for step in common_steps]
    training = {}
    source_row_counts = {}
    for training_run in (previous, run):
        count = 0
        for line in io.BytesIO(payload(training_run / "training.jsonl")):
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            step = int(row["step"])
            if step > cutoff:
                break
            if step in training:
                require(training[step] == row, f"Conflicting overlapping training log at {step}")
            training[step] = row
            count += 1
        source_row_counts[str(training_run)] = count
    require(sorted(training) == list(range(1, cutoff + 1)), "Merged training log contains gaps")
    require(source_row_counts[str(previous)] == resume_step, "Previous training log does not end at resume step")
    rows = [training[step] for step in range(1, cutoff + 1)]
    for row in rows:
        step = row["step"]
        require(row["optimizer_applied"] is True and math.isfinite(row["loss"]), f"Bad training update at {step}")
        require(row["cumulative_tokens"] == input_tokens(step, manifest), f"Input token prefix mismatch at {step}")
        require(row["input_tokens"] == input_tokens(step, manifest) - input_tokens(step - 1, manifest), f"Per-update input count mismatch at {step}")
        require(row["supervised_tokens"] > 0 and row["supervised_tokens"] < row["input_tokens"], f"Bad supervised-token count at {step}")
    means = []
    for offset in range(0, len(rows) - 999, 1000):
        window = rows[offset:offset + 1000]
        targets = sum(row["supervised_tokens"] for row in window)
        means.append({"start_step": window[0]["step"], "end_step": window[-1]["step"],
                      "input_tokens_b": window[-1]["cumulative_tokens"] / 1e9,
                      "updates": len(window), "supervised_tokens": targets,
                      "token_weighted_batch_loss": sum(row["loss"] * row["supervised_tokens"] for row in window) / targets})
    require(means and means[-1]["end_step"] == cutoff // 1000 * 1000,
            "Last batch-loss window does not match the last complete 1000-step window")
    learning_rates = [{"step": row["step"], "input_tokens": row["cumulative_tokens"],
                       "input_tokens_b": row["cumulative_tokens"] / 1e9, "nominal_lr": row["lr"],
                       "effective_fp32_lr": row.get("effective_fp32_lr", row["lr"])} for row in rows]
    policy = manifest["lr_intervention"]
    for row in rows[resume_step:]:
        require(abs(row["lr"] - planned_lr(input_tokens(row["step"] - 1, manifest), policy)) < 1e-15,
                f"Observed LR differs from policy at {row['step']}")
    dependency_path = Path(__file__).parent / "plot_cpt_train_validation.py"
    payload(dependency_path)
    now = datetime.now(timezone.utc)
    before_decay = max((point for point in points if point["input_tokens"] <= policy["start_input_tokens"]), key=lambda point: point["step"])
    summary = {"generated_utc": now.isoformat(), "generated_beijing": now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
               "run": str(run), "previous_run": str(previous), "resume_step": resume_step,
               "common_evaluation_step": cutoff, "latest": points[-1], "baseline": points[0],
               "requested_cutoff_step": args.cutoff_step,
               "last_fixed_evaluation_before_decay": before_decay,
               "changes_since_last_pre_decay_evaluation": {key: points[-1][key] - before_decay[key]
                    for key in ("validation_loss", "fixed_train_probe_loss")},
               "best_validation": min(points, key=lambda point: point["validation_loss"]),
               "best_fixed_train_probe": min(points, key=lambda point: point["fixed_train_probe_loss"]),
               "latest_applied_lr": learning_rates[-1], "batch_mean_latest": means[-1],
               "batch_mean_excluded_tail_updates": cutoff - means[-1]["end_step"],
               "batch_mean_window_updates": 1000, "batch_mean_weighting": "Supervised-token weighted, complete nonoverlapping windows, x at window end",
               "fixed_evaluation_point_count": len(points), "training_rows": len(rows), "source_training_row_counts": source_row_counts,
               "checks": {"checkpoint_chain": True, "same_fixed_train_probe": True, "same_dataset_and_orders": True,
                          "continuous_training_steps": True, "overlap_consistency": True,
                          "exact_input_token_prefix": True, "observed_lr_matches_intervention": True},
               "train_probe_indices_sha256": probe["indices_sha256"], "train_probe_tokens_sha256": probe["tokens_sha256"],
               "policy": policy, "sources": sources, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "fixed_curves_smoothed": False, "training_unchanged": True,
               "limits": ["Different fixed sample losses do not measure overfitting by their absolute gap.",
                          "This observational trajectory includes a first-to-second-epoch transition at 5B as LR decay begins.",
                          "Dashed future LR comes from the current policy, not observed training."]}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "fixed_evaluation.csv", points)
    write_csv(output / "batch_loss_means.csv", means)
    write_csv(output / "applied_learning_rates.csv", learning_rates)
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    require(font.is_file(), "Required Chinese font is missing")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(), "font.size": 10.5,
                         "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    summary["artifacts"] = plot_curves(output, points, means, learning_rates, manifest, summary)
    summary["artifacts"] += plot_curves(output, points, means, learning_rates, manifest, summary, zoom=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("common_evaluation_step", "latest", "latest_applied_lr", "batch_mean_latest", "changes_since_last_pre_decay_evaluation", "artifacts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
