"""Plot fixed native train/validation evaluations and observed batch-loss means.

Read-only and CPU-only. Freeze the input bytes before drawing so an active
trainer can continue publishing logs independently of this report.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def read_json(path):
    for attempt in range(5):
        payload = path.read_bytes()
        try:
            return json.loads(payload), payload
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.05)


def input_tokens(step, manifest):
    length = manifest["signature"]["sequence_length"]
    batch = manifest["signature"]["effective_batch"]
    budget = manifest["dataset_input_tokens"]
    epoch_steps = math.ceil(math.ceil(budget / length) / batch)
    epochs, local_steps = divmod(step, epoch_steps)
    tokens = epochs * budget
    if local_steps:
        selected = min(local_steps * batch, math.ceil(budget / length))
        tokens += selected * length
        tail = budget % length
        if tail and manifest["order_metadata"]["files"][epochs]["tail_position"] < selected:
            tokens -= length - tail
    return tokens


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-window", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_window <= 0:
        parser.error("--batch-window must be positive")
    run = args.run.resolve()
    sources = []

    def load(name):
        data, payload = read_json(run / name)
        sources.append({"path": str(run / name), "sha256": hashlib.sha256(payload).hexdigest()})
        return data

    manifest = load("manifest.json")
    # A continuation keeps global step counters and evaluation history, while
    # each invocation owns only its newly trained log rows. Follow the actual
    # checkpoint chain and trim any parent run at the resumed checkpoint.
    lineage = [(run, manifest, None)]
    visited = {run}
    current = manifest
    while current.get("arguments", {}).get("resume"):
        checkpoint = Path(current["arguments"]["resume"]).resolve()
        parent = checkpoint.parent
        if parent in visited:
            raise ValueError("Cyclic checkpoint history in curve sources")
        checkpoint_manifest, payload = read_json(checkpoint / "checkpoint_manifest.json")
        sources.append({"path": str(checkpoint / "checkpoint_manifest.json"),
                        "sha256": hashlib.sha256(payload).hexdigest()})
        current, payload = read_json(parent / "manifest.json")
        sources.append({"path": str(parent / "manifest.json"),
                        "sha256": hashlib.sha256(payload).hexdigest()})
        lineage.append((parent, current, checkpoint_manifest["completed_steps"]))
        visited.add(parent)
    lineage.reverse()
    data_manifest = load("data_manifest.json")
    sequence_length = manifest["signature"]["sequence_length"]
    validation_blocks = data_manifest["splits"]["validation"]["tokens"] // sequence_length
    current_before = load("validation_before.json")
    if lineage[0][0] == run:
        baseline = current_before
    else:
        baseline_path = lineage[0][0] / "validation_before.json"
        baseline, payload = read_json(baseline_path)
        sources.append({"path": str(baseline_path), "sha256": hashlib.sha256(payload).hexdigest()})
    validation = [baseline, current_before, *load("validation_history.json")]
    probes = load("train_probe_history.json")
    v_by_step = {}
    for item in validation:
        assert item["loss_backend"] == "native eager"
        assert item["blocks"] == validation_blocks
        assert item["input_tokens"] == validation_blocks * sequence_length
        assert item["supervised_tokens"] == validation_blocks * (sequence_length - 1)
        assert math.isfinite(item["loss"])
        if item["step"] in v_by_step:
            assert v_by_step[item["step"]] == item
        v_by_step[item["step"]] = item
    p_by_step = {}
    for item in probes:
        assert item["loss_backend"] == "native eager"
        assert item["input_tokens"] == manifest["train_probe"]["input_tokens"]
        assert item["supervised_tokens"] == manifest["train_probe"]["supervised_tokens"]
        assert math.isfinite(item["loss"])
        assert item["cumulative_tokens"] == input_tokens(item["step"], manifest)
        if item["step"] in p_by_step:
            assert p_by_step[item["step"]] == item
        p_by_step[item["step"]] = item
    steps = sorted(set(v_by_step) & set(p_by_step))
    assert steps and steps[0] == 0
    cutoff = steps[-1]
    points = [{"step": step, "input_tokens": input_tokens(step, manifest),
               "input_tokens_b": input_tokens(step, manifest) / 1e9,
               "validation_loss": v_by_step[step]["loss"],
               "fixed_train_probe_loss": p_by_step[step]["loss"]} for step in steps]

    rows = []
    for directory, _, parent_cutoff in lineage:
        training_path = directory / "training.jsonl"
        training_payload = training_path.read_bytes()
        sources.append({"path": str(training_path), "sha256": hashlib.sha256(training_payload).hexdigest()})
        limit = cutoff if parent_cutoff is None else min(cutoff, parent_cutoff)
        for line in io.BytesIO(training_payload):
            if not line.endswith(b"\n"):
                break  # The active writer may be partway through its last row.
            item = json.loads(line)
            if item["step"] > limit:
                break
            assert item["step"] == len(rows) + 1
            assert item["optimizer_applied"] is True and math.isfinite(item["loss"])
            assert item["cumulative_tokens"] == input_tokens(item["step"], manifest)
            rows.append(item)
    assert len(rows) == cutoff and rows
    means = []
    for offset in range(0, len(rows), args.batch_window):
        window = rows[offset:offset + args.batch_window]
        if len(window) != args.batch_window:
            continue
        targets = sum(item["supervised_tokens"] for item in window)
        means.append({"start_step": window[0]["step"], "end_step": window[-1]["step"],
                      "input_tokens_b": window[-1]["cumulative_tokens"] / 1e9,
                      "supervised_tokens": targets, "updates": len(window),
                      "token_weighted_batch_loss": sum(item["loss"] * item["supervised_tokens"]
                                                       for item in window) / targets})
    assert means

    # Count which fixed training-corpus samples have actually been visited.
    probe_indices = np.array(manifest["train_probe"]["indices"], dtype=np.int64)
    seen = np.zeros(len(probe_indices), dtype=bool)
    epoch_steps = manifest["epoch_steps"]
    full_epochs, local_steps = divmod(cutoff, epoch_steps)
    if full_epochs:
        seen[:] = True
    elif local_steps:
        name = manifest["order_metadata"]["files"][0]["name"]
        order = np.load(run / "orders" / name, mmap_mode="r")
        prefix = order[:local_steps * manifest["signature"]["effective_batch"]]
        seen[:] = np.isin(probe_indices, prefix)

    now = datetime.now(timezone.utc)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "fixed_evaluation.csv", points)
    write_csv(output / "batch_loss_means.csv", means)
    best_val = min(points, key=lambda p: p["validation_loss"])
    best_probe = min(points, key=lambda p: p["fixed_train_probe_loss"])
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    blue, orange, green, muted = "#2766A3", "#D08434", "#168572", "#637384"
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]}, facecolor="white")
    fig.subplots_adjust(left=.09, right=.965, bottom=.19, top=.82, hspace=.36)
    fig.text(.09, .948, "数学继续预训练：训练集与验证集曲线", fontsize=22,
             weight="bold", color="#152A40")
    model_label = manifest["backend_config"]["model"].split("/")[-1]
    execution_label = "原生" if manifest.get("execution_mode") == "standard_native" else "优化版"
    peak_lr_label = f"{manifest['schedule']['peak_lr']:.0e}"
    target_b = manifest["configured_input_tokens"] / 1e9
    fig.text(.09, .907, f"{model_label}  ·  全参数 / {execution_label}  ·  峰值 LR {peak_lr_label}  ·  loss 越低越好",
             fontsize=11.5, color=muted)
    fig.text(.09, .865,
             f"截至 {now.astimezone(ZoneInfo('Asia/Shanghai')):%m月%d日 %H:%M}（北京时间）  ·  "
             f"共同评估到 step {cutoff:,} / {points[-1]['input_tokens_b']:.3f}B tokens  ·  目标 {target_b:g}B",
             fontsize=11, color=muted)
    ax = axes[0]
    x = [p["input_tokens_b"] for p in points]
    for key, color, label, best in (
        ("validation_loss", blue, f"验证集 · 固定 {baseline['input_tokens'] / 1e6:.2f}M tokens", best_val),
        ("fixed_train_probe_loss", orange, f"训练集固定抽样 · {manifest['train_probe']['input_tokens'] / 1e6:.3f}M tokens", best_probe),
    ):
        y = [p[key] for p in points]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=2.8, label=label)
        ax.axhline(y[0], color=color, lw=.9, ls="--", alpha=.3)
        ax.scatter([best["input_tokens_b"]], [best[key]], color=color, s=45, zorder=5)
        if best["step"] != points[-1]["step"]:
            ax.annotate(f"最低 {best[key]:.4f}\n{best['input_tokens_b']:.3f}B",
                    (best["input_tokens_b"], best[key]), xytext=(23, -30),
                    textcoords="offset points", fontsize=10, color=color,
                    arrowprops={"arrowstyle": "-", "color": color},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": 2})
        ax.scatter([x[-1]], [y[-1]], color=color, s=36, zorder=5)
        ax.annotate(f"最新 {y[-1]:.4f}", (x[-1], y[-1]), xytext=(-9, 11),
                    textcoords="offset points", ha="right", fontsize=10, color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": 2})
    eval_every = manifest.get("arguments", {}).get("eval_every")
    eval_label = f"每 {eval_every:,} 步" if eval_every else "固定评估点"
    ax.set_title(f"固定样本评估：{eval_label}，native eager BF16，曲线未平滑", loc="left", pad=12)
    ax.set_ylabel("平均 token loss")
    ax.legend(loc="upper right", ncols=1, frameon=False, fontsize=10)
    ax = axes[1]
    ax.plot([p["input_tokens_b"] for p in means], [p["token_weighted_batch_loss"] for p in means],
            color=green, lw=1.8, marker="o", ms=2.5)
    ax.set_title(f"训练 batch loss：每 {args.batch_window:,} 步按预测目标数加权平均（区间终点）",
                 loc="left", pad=12)
    ax.set_ylabel("区间平均 token loss")
    ax.set_xlabel("累计训练输入 tokens（B，十亿）", labelpad=9)
    ax.text(.985, .965, f"最新区间 {means[-1]['token_weighted_batch_loss']:.4f}",
            transform=ax.transAxes, ha="right", va="top", color=green, fontsize=10)
    warmup_b = input_tokens(manifest["schedule"]["warmup_steps"], manifest) / 1e9
    displayed_b = max(x[-1], min(target_b, .5), 1e-9)
    for ax in axes:
        ax.grid(axis="y", color="#E4EAF0", lw=.8)
        ax.set_axisbelow(True)
        ax.axvline(warmup_b, color="#9BAAB8", ls=":", lw=1.2, alpha=.8)
        ax.set_xlim(-min(.04, displayed_b * .06), displayed_b + min(.07, displayed_b * .14))
        if x[-1] >= manifest["dataset_input_tokens"] / 1e9:
            ax.axvline(manifest["dataset_input_tokens"] / 1e9, color="#9BAAB8", ls=":", lw=1.2)
    axes[0].text(warmup_b + min(.025, displayed_b * .05), .97, "warmup 结束", transform=axes[0].get_xaxis_transform(),
                 va="top", fontsize=9, color=muted)
    fig.text(.09, .093,
             f"固定训练集抽样共 {manifest['train_probe']['blocks']} 块：截至图中末点，已训练到 {int(seen.sum())} 块，尚未训练到 {int((~seen).sum())} 块。",
             fontsize=10, color=muted)
    probe_note = "训练探针沿用扩展前的数据子集；" if manifest.get("training_extension") else ""
    fig.text(.09, .059, probe_note + "两组固定样本不同，loss 绝对差值不能直接视为过拟合程度；虚线为初始值。",
             fontsize=9.5, color=muted)
    fig.text(.09, .028, f"下图统计每次更新前的实际 batch loss；batch 持续变化且采用{execution_label}计算，与固定评估的口径不同。",
             fontsize=9.5, color=muted)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"train_validation_curve.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    summary = {"generated_utc": now.isoformat(), "generated_beijing": now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
               "run": str(run), "source_runs": [str(item[0]) for item in lineage], "common_evaluation_step": cutoff,
               "baseline": points[0], "latest": points[-1],
               "best_validation": best_val, "best_fixed_train_probe": best_probe,
               "train_probe_seen_blocks": int(seen.sum()), "train_probe_unseen_blocks": int((~seen).sum()),
               "batch_mean_window_updates": args.batch_window, "batch_mean_latest": means[-1],
               "batch_mean_weighting": "supervised-token weighted, non-overlapping complete windows, x at window end",
               "fixed_evaluation_point_count": len(points), "batch_mean_point_count": len(means),
               "observed_training_rows_through_cutoff": len(rows), "sources": sources,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "fixed_curves_smoothed": False, "training_unchanged": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("common_evaluation_step", "latest", "best_validation", "best_fixed_train_probe", "batch_mean_latest")}, indent=2))
    print(output / "train_validation_curve.png")


if __name__ == "__main__":
    main()
