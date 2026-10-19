"""Draw completed fixed seen/unseen evaluations without touching training.

CPU-only. Each group's change is measured against that same group's base
loss; group-specific difficulty therefore does not determine the change.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


GROUPS = ["seen_train", "unseen_train", "validation"]
ALL_GROUPS = [*GROUPS, "uniform_train"]
LABELS = {
    "seen_train": "早期已见训练块",
    "unseen_train": "晚期仍未见训练块",
    "validation": "固定验证集",
    "uniform_train": "训练集均匀抽样（辅助）",
}
COLORS = {
    "seen_train": "#2878A7",
    "unseen_train": "#C77B2D",
    "validation": "#64758B",
}


def load_snapshot(path):
    payload = path.read_bytes()
    return json.loads(payload), {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def dump_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = (args.output or run / "report").resolve()
    summary, summary_source = load_snapshot(run / "evaluation_summary.json")
    plan, plan_source = load_snapshot(run / "plan.json")
    if summary.get("status") != "complete":
        raise SystemExit("Evaluation summary is not complete; no report generated.")

    cases = ["base", f"step{plan['early_step']}", f"step{plan['late_step']}"]
    x_b = []
    losses = {group: [] for group in ALL_GROUPS}
    raw_rows = []
    for case in cases:
        item = summary["cases"][case]
        step = int(item["step"])
        tokens = int(item["trained_tokens"])
        expected = next(p for p in plan["cases"] if p["label"] == case)
        if step != expected["step"] or tokens != expected["trained_tokens"]:
            raise ValueError(f"Snapshot metadata differs from plan: {case}")
        x_b.append(tokens / 1e9)
        for group in ALL_GROUPS:
            metric = item["metrics"][group]
            loss = float(metric["loss"])
            if not math.isfinite(loss):
                raise ValueError(f"Nonfinite loss in {case}/{group}")
            losses[group].append(loss)
            raw_rows.append({
                "case": case, "step": step, "trained_input_tokens": tokens,
                "trained_input_tokens_b": tokens / 1e9,
                "group": group, "loss": loss,
                "input_tokens": metric.get("input_tokens", ""),
                "supervised_tokens": metric.get("supervised_tokens", ""),
                "loss_change_from_base": loss - losses[group][0],
            })

    deltas = {}
    delta_rows = []
    for group in ALL_GROUPS:
        base, early, late = losses[group]
        deltas[group] = {
            "base_to_early": early - base,
            "early_to_late": late - early,
            "base_to_late": late - base,
        }
        for name, value in deltas[group].items():
            expected = summary["loss_changes"][group][name]
            if not math.isclose(value, float(expected), rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Stored delta mismatch: {group}/{name}")
        delta_rows.append({"group": group, "base_loss": base,
                           "early_loss": early, "late_loss": late,
                           **deltas[group]})

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.is_file():
        raise FileNotFoundError(f"Required CJK font unavailable: {font}")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=font).get_name(),
        "font.size": 11,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 8.6), facecolor="white",
                             gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=.085, right=.965, top=.72, bottom=.39, wspace=.3)
    dark, muted = "#172E42", "#66788A"
    early_step, late_step = plan["early_step"], plan["late_step"]
    fig.text(.085, .95, "固定已见 / 未见样本的 loss 对比",
             fontsize=22, weight="bold", color=dark)
    fig.text(.085, .905,
             "Qwen2.5-Coder-1.5B  ·  全参数 / 峰值 LR 3e-6  ·  原生评估  ·  越低越好",
             fontsize=12, color=muted)
    fig.text(.085, .866,
             f"基座 → step {early_step:,}（{x_b[1]:.3f}B tokens）"
             f" → step {late_step:,}（{x_b[2]:.3f}B tokens）",
             fontsize=11.5, color=muted)
    fig.text(.085, .818,
             f"已见 / 未见各固定 {plan['probe_blocks']} 个 2048-token 块；"
             "逐组比较同一批样本，不比较组间原始难度。",
             fontsize=11, color=muted)

    ax = axes[0]
    for group in GROUPS:
        relative = [loss - losses[group][0] for loss in losses[group]]
        ax.plot(x_b, relative, color=COLORS[group], marker="o", ms=7,
                lw=2, ls="--", label=LABELS[group], zorder=3)
    ax.axhline(0, color="#7D8D9C", lw=1, zorder=1)
    ax.grid(axis="y", color="#DFE6EB", lw=.8, alpha=.85)
    ax.set_title("各组相对自身基座的变化", loc="left", fontsize=13,
                 weight="bold", pad=15, color=dark)
    ax.set_ylabel("loss − 该组基座 loss")
    ax.set_xlabel("已训练输入 token（B）", labelpad=11)
    ax.set_xticks(x_b)
    ax.set_xticklabels(["0\n基座", f"{x_b[1]:.3f}\n早期", f"{x_b[2]:.3f}\n晚期"])
    ax.set_xlim(-.12, x_b[-1] + .15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(.08, .775),
               ncol=3, frameon=False, fontsize=10.5, columnspacing=2.1)

    ax = axes[1]
    values = [deltas[group]["early_to_late"] for group in GROUPS]
    ax.bar(range(len(GROUPS)), values, color=[COLORS[g] for g in GROUPS],
           width=.58, zorder=3)
    ax.axhline(0, color="#7D8D9C", lw=1, zorder=1)
    ax.grid(axis="y", color="#DFE6EB", lw=.8, alpha=.85)
    ax.set_title("从早期到晚期的回升幅度", loc="left", fontsize=13,
                 weight="bold", pad=15, color=dark)
    ax.set_ylabel("晚期 loss − 早期 loss")
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(["早期已见", "晚期未见", "验证集"])
    spread = max(max(abs(value) for value in values), .005)
    low = min(0, min(values)) - spread * .12
    high = max(0, max(values)) + spread * .30
    ax.set_ylim(low, high)
    for i, value in enumerate(values):
        ax.annotate(f"{value:+.5f}", xy=(i, value),
                    xytext=(0, 7 if value >= 0 else -8),
                    textcoords="offset points", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=11,
                    weight="bold", color=COLORS[GROUPS[i]])

    # Show raw losses beneath the normalized plot, not on a second y-axis.
    table_y, row_height = .27, .032
    columns = [.085, .365, .535, .705, .87]
    for xpos, label in zip(columns, ["固定样本", "基座 loss", "早期 loss", "晚期 loss", "早期 → 晚期"]):
        fig.text(xpos, table_y, label, fontsize=10.5, color=muted,
                 ha="left" if xpos == columns[0] else "right")
    for i, group in enumerate(GROUPS, start=1):
        y = table_y - i * row_height
        fig.text(columns[0], y, LABELS[group], fontsize=11, color=COLORS[group])
        for xpos, loss in zip(columns[1:4], losses[group]):
            fig.text(xpos, y, f"{loss:.5f}", ha="right", fontsize=11, color=dark)
        fig.text(columns[4], y, f"{deltas[group]['early_to_late']:+.5f}",
                 ha="right", fontsize=11, color=COLORS[group], weight="bold")

    fig.text(.085, .118,
             "已见：早期 checkpoint 前已用于本次 CPT；未见：晚期 checkpoint 前尚未用于本次 CPT。",
             fontsize=10.2, color=muted)
    fig.text(.085, .085,
             "“见过”以精确 token 块为准；不排除相邻文档片段或基座预训练重叠。"
             "三次快照的连线仅用于对照，不能确定唯一原因。",
             fontsize=10, color=muted)
    fig.text(.085, .05,
             "训练集均匀抽样的辅助结果另存 CSV；本诊断只做评估，没有训练更新。",
             fontsize=10, color=muted)

    output.mkdir(parents=True, exist_ok=True)
    stem = "seen_unseen_comparison"
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(output / f"{stem}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    dump_csv(output / "fixed_evaluation.csv", raw_rows)
    dump_csv(output / "loss_changes.csv", delta_rows)
    report = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "sources": [summary_source, plan_source],
        "plotted_groups": GROUPS,
        "auxiliary_csv_group": "uniform_train",
        "cases": cases,
        "x_input_tokens_b": x_b,
        "losses": losses,
        "loss_changes": deltas,
        "outputs": [str(output / f"{stem}.{suffix}") for suffix in ["png", "svg", "pdf"]]
                   + [str(output / "fixed_evaluation.csv"), str(output / "loss_changes.csv")],
        "limits": plan["limits"],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "loss_changes": deltas}, ensure_ascii=False))


if __name__ == "__main__":
    main()
