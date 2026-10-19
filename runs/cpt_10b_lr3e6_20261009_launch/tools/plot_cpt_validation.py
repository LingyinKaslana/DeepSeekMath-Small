"""Plot comparable saved native validation measurements without using the GPU."""
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


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_tokens(step, manifest):
    """Exact order-prefix count, including the shuffled partial token block."""
    length = manifest["signature"]["sequence_length"]
    batch = manifest["signature"]["effective_batch"]
    budget = manifest["dataset_input_tokens"]
    blocks = math.ceil(budget / length)
    epoch_steps = math.ceil(blocks / batch)
    completed_epochs, local_steps = divmod(step, epoch_steps)
    total = completed_epochs * budget
    if local_steps:
        selected = min(local_steps * batch, blocks)
        total += selected * length
        tail = budget % length
        if tail and manifest["order_metadata"]["files"][completed_epochs]["tail_position"] < selected:
            total -= length - tail
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--failure-step", type=int, help="One-based failed update; not a validation point")
    p.add_argument("--rollback-step", type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, sources, duplicates = {}, [], 0
    reference = None
    for run in args.run:
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        comparable = {key: manifest["signature"][key] for key in (
            "backend_config_sha256", "data_manifest_sha256", "token_file_sha256",
            "sequence_length", "effective_batch", "epochs", "seed", "peak_lr")}
        if reference is None:
            reference = manifest
            reference_comparable = comparable
        elif comparable != reference_comparable or manifest["order_metadata"] != reference["order_metadata"]:
            raise ValueError(f"Incomparable validation data/order/config: {run}")
        for path in (manifest_path, run / "validation_before.json", run / "validation_history.json"):
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            sources.append({"path": str(path.resolve()), "sha256": sha(path)})
            if path == manifest_path:
                continue
            measurements = payload if isinstance(payload, list) else [payload]
            for item in measurements:
                assert item["loss_backend"] == "native eager"
                assert item["supervised_tokens"] == 2096128 and item["blocks"] == 1024
                assert math.isfinite(item["loss"]) and math.isfinite(item["perplexity"])
                step = item["step"]
                measurement = {key: item[key] for key in (
                    "step", "loss", "perplexity", "supervised_tokens", "input_tokens",
                    "blocks", "loss_backend", "attention_mask")}
                if step in rows:
                    if rows[step]["measurement"] != measurement:
                        raise ValueError(f"Conflicting validation measurements at step {step}")
                    duplicates += 1
                    rows[step]["sources"].append(str(path.resolve()))
                else:
                    rows[step] = {"measurement": measurement, "sources": [str(path.resolve())]}
    points = []
    for step, row in sorted(rows.items()):
        tokens = input_tokens(step, reference)
        points.append(dict(step=step, trained_input_tokens=tokens, trained_tokens_b=tokens / 1e9,
                           validation_loss=row["measurement"]["loss"],
                           perplexity=row["measurement"]["perplexity"], sources=row["sources"]))
    if not points or points[0]["step"] != 0:
        raise ValueError("Expected the original base-model validation at step zero")
    best = min(points, key=lambda row: row["validation_loss"])
    first, latest = points[0], points[-1]
    with (args.output / "validation_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "trained_input_tokens", "trained_tokens_b",
                                                   "validation_loss", "perplexity", "sources"))
        writer.writeheader()
        writer.writerows(dict(row, sources=";".join(row["sources"])) for row in points)

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.size": 12, "axes.unicode_minus": False, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12.6, 6.7), facecolor="white")
    fig.subplots_adjust(left=.09, right=.96, bottom=.19, top=.81)
    fig.text(.09, .935, "数学继续预训练：验证集 loss", fontsize=23, weight="bold", color="#14283f")
    fig.text(.09, .878, "Qwen2.5-Coder-1.5B  ·  全参数训练  ·  OpenWebMath  ·  越低越好",
             fontsize=12, color="#526579")
    x = [row["trained_tokens_b"] for row in points]
    y = [row["validation_loss"] for row in points]
    ax.plot(x, y, color="#2766a3", lw=2.2, marker="o", ms=4, label="验证 loss", zorder=3)
    ax.axhline(first["validation_loss"], color="#83909b", lw=1.2, ls="--",
               label=f"训练前基线 {first['validation_loss']:.4f}")
    ax.scatter(best["trained_tokens_b"], best["validation_loss"], s=65, color="#12805c", zorder=5)
    ax.annotate(f"最低 {best['validation_loss']:.4f}\n{best['trained_tokens_b']:.3f}B tokens / step {best['step']:,}",
                xy=(best["trained_tokens_b"], best["validation_loss"]),
                xytext=(.15, .14), textcoords="axes fraction", fontsize=11, color="#126a50",
                arrowprops={"arrowstyle": "-", "color": "#12805c", "lw": 1.1},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    ax.scatter(latest["trained_tokens_b"], latest["validation_loss"], s=63, marker="D",
               color="#dc8630", edgecolors="white", lw=.7, zorder=6)
    ax.annotate(f"最近 {latest['validation_loss']:.4f}\n{latest['trained_tokens_b']:.3f}B tokens / step {latest['step']:,}",
                xy=(latest["trained_tokens_b"], latest["validation_loss"]),
                xytext=(.63, .63), textcoords="axes fraction", fontsize=11, color="#97601f",
                arrowprops={"arrowstyle": "-", "color": "#b67c3b", "lw": 1.1},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    epoch_b = reference["dataset_input_tokens"] / 1e9
    failure_b = input_tokens(args.failure_step - 1, reference) / 1e9 if args.failure_step else None
    right = max(x[-1], failure_b or 0)
    ax.set_xlim(-.08, max(1, math.ceil(right)) + .1)
    span = max(max(y) - min(y), .025)
    low, high = min(y) - span * .15, max(y) + span * .25
    ax.set_ylim(low, high)
    if right >= epoch_b:
        ax.axvline(epoch_b, color="#c7cfd7", ls=":", lw=1.4)
        ax.text(epoch_b - .07, high - span * .06, "第二轮开始", fontsize=10,
                color="#647383", ha="right", va="top")
    if failure_b is not None:
        ax.axvline(failure_b, color="#bd6c70", ls="--", lw=1, alpha=.8)
        rollback_b = input_tokens(args.rollback_step, reference) / 1e9 if args.rollback_step else None
        text = f"异常中断于 {failure_b:.3f}B"
        if rollback_b is not None:
            text += f"\n回退至 {rollback_b:.3f}B 续训"
        ax.text(.985, .30, text, transform=ax.transAxes, fontsize=10, ha="right",
                color="#a34c52", bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    ax.set_xlabel("累计训练输入 tokens（B，十亿）", labelpad=10)
    ax.set_ylabel("验证集平均 token loss", labelpad=12)
    ax.grid(axis="y", color="#e5eaf0", lw=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.085), ncols=2,
              frameon=False, fontsize=10, handlelength=2.6)
    now = datetime.now(timezone.utc)
    fig.text(.09, .066, f"同一验证集：2,097,152 输入 tokens / 2,096,128 预测目标  ·  native eager BF16  ·  {len(points)} 个唯一评估点",
             fontsize=10, color="#647383")
    fig.text(.09, .030, "继承历史及恢复时的重复点已去重；连接线仅便于阅读，未平滑，也未推测中断处的 loss。",
             fontsize=9, color="#798591")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output / f"validation_curve.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    summary = {"generated_utc": now.isoformat(), "generated_beijing": now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
               "unique_points": len(points), "duplicate_measurements_removed": duplicates,
               "baseline": first, "best": best, "latest": latest,
               "latest_minus_baseline_loss": latest["validation_loss"] - first["validation_loss"],
               "loss_is_native_eager_and_unsmoothed": True, "sources": sources,
               "script_sha256": sha(Path(__file__)), "matplotlib_version": matplotlib.__version__,
               "points": points, "failure_step": args.failure_step, "failure_position_b": failure_b,
               "rollback_step": args.rollback_step}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("unique_points", "duplicate_measurements_removed", "latest_minus_baseline_loss")}))
    print(args.output / "validation_curve.png")


if __name__ == "__main__":
    main()
