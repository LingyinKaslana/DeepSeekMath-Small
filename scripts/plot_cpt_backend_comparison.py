"""Audit and plot a completed fast/native CPT comparison using only saved CPU artifacts."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


CASES = ("fast", "native")
LABELS = {"fast": "高速实现", "native": "标准原生"}
COLORS = {"fast": "#2766a3", "native": "#df872c"}
COMMON_PLAN_FIELDS = (
    "checkpoint", "checkpoint_signature", "checkpoint_manifest_sha256", "checkpoint_files",
    "checkpoint_training_state_file", "source_run", "source_directory", "source_sha256",
    "diagnostic_script_sha256", "data", "data_manifest_sha256", "data_artifact_sha256",
    "start_step", "end_step", "updates", "schedule", "order_metadata",
    "selected_block_indices_sha256", "selected_training_token_and_label_values_sha256",
    "selected_blocks", "partial_tail_global_steps", "start_prefix_counts", "end_prefix_counts",
    "new_input_tokens", "new_supervised_tokens", "batch", "fixed_train_probe", "validation",
    "evaluation_offsets", "precision", "optimizer_math", "versions",
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def assert_check(checks, name, condition, detail=None):
    row = {"name": name, "passed": bool(condition)}
    if detail is not None:
        row["detail"] = detail
    checks.append(row)
    if not condition:
        raise ValueError(f"Comparison audit failed: {name}")


def scheduled_lr(schedule, global_step):
    step = global_step - 1
    if step < schedule["warmup_steps"]:
        factor = (step + 1) / schedule["warmup_steps"]
    elif step < schedule["decay_80_step"]:
        factor = 1.0
    elif step < schedule["decay_90_step"]:
        factor = schedule["decay_80_factor"]
    else:
        factor = schedule["decay_90_factor"]
    return schedule["peak_lr"] * factor


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def prefix_inputs(plan, completed_steps, dataset_token_budget):
    """Recover exact saved-order input count, including partial blocks/batches."""
    blocks = plan["order_metadata"]["dataset_blocks"]
    batch = plan["batch"]["effective"]
    length = plan["batch"]["sequence_length"]
    epoch_steps = math.ceil(blocks / batch)
    budget = dataset_token_budget
    signature = plan["checkpoint_signature"]
    if blocks != math.ceil(budget / length):
        raise ValueError("Dataset token budget disagrees with the saved order")
    if signature["effective_batch"] != batch or signature["sequence_length"] != length:
        raise ValueError("Plan batch differs from source checkpoint signature")
    epoch, local = divmod(completed_steps, epoch_steps)
    total = epoch * budget
    if local:
        selected = min(local * batch, blocks)
        total += selected * length
        remainder = budget % length
        if remainder and plan["order_metadata"]["files"][epoch]["tail_position"] < selected:
            total -= length - remainder
    return total


def load_case(directory):
    filenames = ("plan.json", "result.json", "evaluation_history.json", "training.jsonl",
                 "initial_optimizer_audit.json", "initial_rng.json", "initial_finite_audit.json",
                 "final_finite_audit.json", "backend.json")
    artifacts = {}
    data = {}
    for name in filenames:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing completed-comparison artifact: {path}")
        artifacts[name] = {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}
        if name == "training.jsonl":
            data["training"] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            data[name.removesuffix(".json")] = read_json(path)
    adaptation = directory / "optimizer_implementation_adaptation.json"
    if adaptation.is_file():
        data["optimizer_implementation_adaptation"] = read_json(adaptation)
        artifacts[adaptation.name] = {"path": str(adaptation.resolve()), "sha256": sha(adaptation)}
    data["artifacts"] = artifacts
    data["failure_artifact_exists"] = (directory / "nonfinite_failure.json").exists()
    metadata_path = Path(data["plan"]["data"]) / "manifest.json"
    if sha(metadata_path) != data["plan"]["data_manifest_sha256"]:
        raise ValueError("Current small data manifest differs from the recorded plan")
    data["dataset_manifest"] = read_json(metadata_path)
    artifacts["data/manifest.json"] = {"path": str(metadata_path.resolve()), "sha256": sha(metadata_path)}
    return data


def audit_cases(data, expected_updates=1000):
    """Validate actual runner schemas; never infer a missing check as a pass."""
    checks = []
    fast, native = (data[name] for name in CASES)
    plans = {name: data[name]["plan"] for name in CASES}
    for key in COMMON_PLAN_FIELDS:
        assert_check(checks, f"common plan: {key}", key in plans["fast"] and key in plans["native"]
                     and plans["fast"][key] == plans["native"][key],
                     {name: canonical_sha(plans[name].get(key)) for name in CASES})
    assert_check(checks, "common initial Adam tensors and steps",
                 fast["initial_optimizer_audit"] == native["initial_optimizer_audit"])
    assert_check(checks, "common restored CPU/CUDA RNG", fast["initial_rng"] == native["initial_rng"])
    common_math_rows = []
    measurements = []
    for case in CASES:
        c = data[case]
        plan, result, logs = c["plan"], c["result"], c["training"]
        budget = c["dataset_manifest"]["splits"]["train"]["tokens"]
        assert_check(checks, f"{case}: completed {expected_updates} updates",
                     plan["case"] == case and plan["updates"] == expected_updates
                     and plan["end_step"] - plan["start_step"] == expected_updates
                     and result["status"] == "complete" and result["case"] == case
                     and result["optimizer_updates_applied"] == expected_updates
                     and result["start_step"] == plan["start_step"] and result["end_step"] == plan["end_step"]
                     and len(logs) == expected_updates)
        assert_check(checks, f"{case}: no failure or production mutation",
                     result["failures"] == [] and not c["failure_artifact_exists"]
                     and result["production_files_modified"] is False and plan["production_files_modified"] is False)
        cpu = plan["cpu_data_path_comparison"]
        assert_check(checks, f"{case}: all selected batches have matching CPU token/label values",
                     cpu["passed"] is True and cpu["updates"] == expected_updates
                     and all(cpu[k] is True for k in ("input_ids_equal", "labels_equal", "token_and_target_counts_equal")))
        assert_check(checks, f"{case}: recorded selected token hash matches result",
                     result["training_token_and_label_values_sha256"] == plan["selected_training_token_and_label_values_sha256"])
        assert_check(checks, f"{case}: original Adam state is finite FP32",
                     c["initial_optimizer_audit"]["all_finite"] is True
                     and c["initial_optimizer_audit"]["moments_dtype"] == "torch.float32"
                     and c["initial_optimizer_audit"]["step_values"] == [plan["start_prefix_counts"][2]])
        assert_check(checks, f"{case}: RNG result matches restored-state audit", result["initial_rng"] == c["initial_rng"])
        for filename, expected in (("initial_finite_audit", plan["start_prefix_counts"][2]),
                                   ("final_finite_audit", plan["end_prefix_counts"][2])):
            finite = c[filename]
            assert_check(checks, f"{case}: {filename} all weight/Adam tensors",
                         finite["passed"] is True and finite["expected_updates"] == expected
                         and finite["parameter_tensors"] == c["initial_optimizer_audit"]["tensor_count"]
                         and len(finite["rows"]) == finite["parameter_tensors"]
                         and all(row["passed"] is True and row["parameter_finite"] is True
                                 and row["parameter_dtype"] == "torch.float32"
                                 and row["gradient_dtype"] in (None, "torch.float32") and row["step"] == expected
                                 and all(row[key + "_finite"] is True and row[key + "_dtype"] == "torch.float32"
                                         for key in ("exp_avg", "exp_avg_sq")) for row in finite["rows"]))
        count, targets, math_rows = plan["start_prefix_counts"][0], 0, []
        for offset, record in enumerate(logs, start=1):
            step = plan["start_step"] + offset
            rate = scheduled_lr(plan["schedule"], step)
            ok = (record["offset"] == offset and record["step"] == step and record["optimizer_applied"] is True
                  and all(finite_number(record[k]) for k in ("loss", "grad_norm", "seconds", "scheduled_lr", "effective_fp32_lr"))
                  and record["seconds"] > 0 and record["grad_norm"] >= 0
                  and record["scheduled_lr"] == rate and record["effective_fp32_lr"] == fp32(rate))
            if not ok:
                assert_check(checks, f"{case}: valid finite update {offset}", False)
            count += record["input_tokens"]
            targets += record["supervised_tokens"]
            if record["cumulative_tokens"] != count or prefix_inputs(plan, step, budget) != count:
                assert_check(checks, f"{case}: exact input prefix at update {offset}", False)
            math_rows.append({key: record[key] for key in ("offset", "step", "scheduled_lr", "effective_fp32_lr",
                                                          "input_tokens", "supervised_tokens", "cumulative_tokens", "optimizer_applied")})
        assert_check(checks, f"{case}: all logged losses/norms finite and LR/counts match plan", True)
        assert_check(checks, f"{case}: exact total input/target counts",
                     count == plan["end_prefix_counts"][0] and count - plan["start_prefix_counts"][0] == plan["new_input_tokens"]
                     and targets == plan["new_supervised_tokens"] and result["new_input_tokens"] == plan["new_input_tokens"])
        common_math_rows.append(math_rows)
        assert_check(checks, f"{case}: evaluation offsets are complete",
                     [r["offset"] for r in c["evaluation_history"]] == plan["evaluation_offsets"]
                     and c["evaluation_history"] == result["evaluation_history"])
        for row in c["evaluation_history"]:
            assert_check(checks, f"{case}: evaluation position {row['offset']}",
                         row["global_step"] == plan["start_step"] + row["offset"]
                         and row["cumulative_input_tokens"] == prefix_inputs(plan, row["global_step"], budget))
            for probe, source in (("validation", plan["validation"]), ("train_probe", plan["fixed_train_probe"])):
                metric = row[probe]
                assert_check(checks, f"{case}: {probe} native evaluation {row['offset']}",
                             metric["backend"] == "unpatched HF native eager"
                             and all(metric[k] == source[k] for k in ("blocks", "input_tokens", "supervised_tokens"))
                             and finite_number(metric["loss"]) and finite_number(metric["perplexity"])
                             and math.isclose(metric["perplexity"], math.exp(metric["loss"]), rel_tol=1e-12))
            measurements.append({"case": case, "offset": row["offset"], "global_step": row["global_step"],
                "cumulative_input_tokens": row["cumulative_input_tokens"],
                "additional_input_tokens": row["cumulative_input_tokens"] - plan["start_prefix_counts"][0],
                "validation_loss": row["validation"]["loss"], "validation_perplexity": row["validation"]["perplexity"],
                "train_probe_loss": row["train_probe"]["loss"], "train_probe_perplexity": row["train_probe"]["perplexity"],
                "evaluation_seconds": row["seconds"]})
        duration = sum(row["seconds"] for row in logs)
        first = logs[0]
        steady = (plan["new_input_tokens"] - first["input_tokens"]) / (duration - first["seconds"])
        assert_check(checks, f"{case}: recorded training speeds match update timings",
                     math.isclose(duration, result["training_seconds_including_compile"], rel_tol=1e-10)
                     and math.isclose(plan["new_input_tokens"] / duration, result["tokens_per_second_including_compile"], rel_tol=1e-10)
                     and math.isclose(steady, result["steady_tokens_per_second"], rel_tol=1e-10))
    assert_check(checks, "all per-update token counts, order positions and Adam math agree",
                 common_math_rows[0] == common_math_rows[1], canonical_sha(common_math_rows[0]))
    for probe in ("validation", "train_probe"):
        a, b = (data[case]["evaluation_history"][0][probe] for case in CASES)
        assert_check(checks, f"exactly equal initial native {probe} losses", a == b,
                     {case: data[case]["evaluation_history"][0][probe]["loss"] for case in CASES})
    adaptation = native["optimizer_implementation_adaptation"]
    assert_check(checks, "native Adam execution adaptation preserves tensor values",
                 adaptation["tensor_values_unchanged"] is True
                 and adaptation["combined_tensor_sha256"] == native["initial_optimizer_audit"]["combined_tensor_sha256"])
    assert_check(checks, "native reference is unpatched and uncompiled", native["backend"]["compile"] is False
                 and native["backend"]["model"] == "unpatched HF AutoModelForCausalLM")
    return checks, measurements, canonical_sha(common_math_rows[0])


def summarize(data, checks, measurements, per_update_sha):
    comparisons, outcomes = [], {}
    for case in CASES:
        rows = [r for r in measurements if r["case"] == case]
        initial, final = rows[0], rows[-1]
        result = data[case]["result"]
        outcomes[case] = {"initial": initial, "final": final,
            "validation_loss_change": final["validation_loss"] - initial["validation_loss"],
            "train_probe_loss_change": final["train_probe_loss"] - initial["train_probe_loss"],
            **{key: result[key] for key in ("steady_tokens_per_second", "tokens_per_second_including_compile",
                    "training_seconds_including_compile", "total_seconds_including_load_eval_save", "peak_allocated_bytes", "peak_reserved_bytes")},
            "first_update_seconds_including_initialization": data[case]["training"][0]["seconds"]}
    for fast, native in zip(data["fast"]["evaluation_history"], data["native"]["evaluation_history"]):
        comparisons.append({"offset": fast["offset"],
            "validation_fast_minus_native": fast["validation"]["loss"] - native["validation"]["loss"],
            "train_probe_fast_minus_native": fast["train_probe"]["loss"] - native["train_probe"]["loss"]})
    plan = data["fast"]["plan"]
    now = datetime.now(timezone.utc)
    return {"status": "complete", "generated_utc": now.isoformat(),
        "generated_beijing": now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
        "strict_audit_passed": all(row["passed"] for row in checks), "checks": checks,
        "start_checkpoint": plan["checkpoint"], "start_step": plan["start_step"], "end_step": plan["end_step"],
        "updates_per_case": plan["updates"], "additional_input_tokens_per_case": plan["new_input_tokens"],
        "common_plan_fingerprints": {key: canonical_sha(plan[key]) for key in COMMON_PLAN_FIELDS},
        "per_update_data_positions_and_math_sha256": per_update_sha,
        "initial_optimizer_tensor_sha256": data["fast"]["initial_optimizer_audit"]["combined_tensor_sha256"],
        "initial_rng": data["fast"]["initial_rng"], "outcomes": outcomes, "differences_at_each_offset": comparisons,
        "fast_to_native_steady_speed_ratio": outcomes["fast"]["steady_tokens_per_second"] / outcomes["native"]["steady_tokens_per_second"],
        "measurements": measurements, "artifacts": {case: data[case]["artifacts"] for case in CASES},
        "script_sha256": sha(Path(__file__)), "matplotlib_version": matplotlib.__version__,
        "limits": [
            f"Only this checkpoint and a short {plan['updates']}-update window are compared; this does not establish the cause of the earlier long-run loss rise or outcomes from the base model.",
            "This compares implementation packages: model compilation, optimizer execution, batch construction and mask representation all differ. It does not isolate model compilation alone.",
            "FlashAttention training is nondeterministic. One trajectory per case is insufficient for a statistical confidence interval.",
            "Both validation and the fixed training-corpus probe use the same unpatched native eager evaluation. The probe is not a held-out validation split or benchmark score.",
            "Per-update timing includes data/H2D, finite-loss synchronization, backward, clipping, AdamW and GPU synchronization; excludes evaluation, JSON logging, loading and saving. Steady speed excludes the first update in each case.",
            "The report compares recorded checkpoint/data fingerprints and runner audits. It does not re-read the large checkpoint or full 5B token file. Per-update checks compare logged positions/counts/LR; token/label value parity was checked for every selected batch in the runner CPU preflight.",
        ]}


def plot(summary, output):
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.2), facecolor="white")
    fig.subplots_adjust(left=.085, right=.965, top=.735, bottom=.29, wspace=.28)
    fig.text(.085, .94, "高速实现与标准原生训练：短窗口对照", fontsize=22, weight="bold", color="#172d43")
    fig.text(.085, .883, f"相同 checkpoint（step {summary['start_step']:,}） · 每臂 {summary['updates_per_case']:,} 更新 / {summary['additional_input_tokens_per_case'] / 1e6:.3f}M tokens",
             fontsize=11, color="#596b7c")
    handles = []
    for ax, key, title in zip(axes, ("validation_loss", "train_probe_loss"),
                             ("完整验证集 · 原生 loss", "固定训练语料探针 · 原生 loss")):
        all_losses = []
        for case in CASES:
            rows = [r for r in summary["measurements"] if r["case"] == case]
            x = [r["additional_input_tokens"] / 1e6 for r in rows]
            y = [r[key] for r in rows]
            line, = ax.plot(x, y, lw=2.2, color=COLORS[case], marker="o" if case == "fast" else "s",
                           ms=5, label=LABELS[case], zorder=3)
            all_losses.extend(y)
            if ax is axes[0]:
                handles.append(line)
        baseline = summary["outcomes"]["fast"]["initial"][key]
        ax.axhline(baseline, color="#9aa6b2", ls="--", lw=1, zorder=1)
        span = max(max(all_losses) - min(all_losses), 1e-5)
        ax.set_ylim(min(all_losses) - .16 * span, max(all_losses) + .18 * span)
        ax.set_title(title, loc="left", fontsize=14, pad=13, color="#20384e")
        ax.set_xlabel("额外训练输入 tokens（M，百万）", labelpad=10)
        ax.set_ylabel("平均 token loss（越低越好）", labelpad=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f" if span < .001 else "%.4f"))
        ax.grid(axis="y", color="#e6ebf0", lw=.8)
        ax.set_axisbelow(True)
    fig.legend(handles=handles, labels=[LABELS[case] for case in CASES], loc="upper left",
               bbox_to_anchor=(.078, .84), frameon=False, ncols=2, handlelength=2.4)
    speed = " · ".join(f"{LABELS[case]} {summary['outcomes'][case]['steady_tokens_per_second'] / 1e4:.2f} 万 tokens/s" for case in CASES)
    fig.text(.085, .15, f"稳态吞吐：{speed}（均排除首步与评估、保存时间）", fontsize=10, color="#53687a")
    fig.text(.085, .094, "两臂均用相同原生前向评估；连线未平滑。仅比较当前 checkpoint 的短窗口，无法单独归因于编译。",
             fontsize=10, color="#687888")
    fig.text(.085, .046, "纵轴按观测范围展开；这是实现组合对照，不代表长期训练收益或数学基准成绩。",
             fontsize=9, color="#7a8793")
    for extension in ("png", "svg"):
        fig.savefig(output / f"backend_comparison.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1] / "runs/cpt_native_compare_20261009"
    parser.add_argument("--experiment-root", "--root", "--run", dest="root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-updates", type=int, default=1000)
    parser.add_argument("--check-only", action="store_true", help="Write audited CSV/JSON without rendering a figure")
    args = parser.parse_args()
    if args.expected_updates < 1:
        parser.error("--expected-updates must be positive")
    output = args.output or args.root / "report"
    data = {case: load_case(args.root / case) for case in CASES}
    checks, measurements, per_update_sha = audit_cases(data, args.expected_updates)
    summary = summarize(data, checks, measurements, per_update_sha)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "backend_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    if not args.check_only:
        plot(summary, output)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"strict_audit_passed": summary["strict_audit_passed"],
                     "fast_to_native_steady_speed_ratio": summary["fast_to_native_steady_speed_ratio"],
                     "final_fast_minus_native": summary["differences_at_each_offset"][-1],
                     "output": str(output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
