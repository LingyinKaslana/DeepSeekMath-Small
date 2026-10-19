"""Sequential read-only model evaluations alongside the independent CPT job."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

DIAG = Path(__file__).resolve().parents[1]
PLAN = json.loads((DIAG / "plan.json").read_text())
PROBES = json.loads((DIAG / "probes.json").read_text())
HISTORY = json.loads((Path(PLAN["production_run"]) / "train_probe_history.json").read_text())
uniform_expected = {row["step"]: row["loss"] for row in HISTORY}
summary = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "coordinator_pid": os.getpid(), "production_run": PLAN["production_run"],
           "training_updates": 0, "production_files_modified": False, "cases": {},
           "early_step": PLAN["early_step"], "late_step": PLAN["late_step"], "limits": PLAN["limits"]}


def save():
    path = DIAG / "evaluation_summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(path)


def record_failure(kind, error, traceback):
    summary.update(status="failed", error=str(error),
                   finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    sys.__excepthook__(kind, error, traceback)


sys.excepthook = record_failure
save()
for case in PLAN["cases"]:
    label = case["label"]
    command = [sys.executable, str(DIAG / "tools/evaluate_cpt_seen_unseen.py"),
               "--source", str(DIAG / "source"), "--data", PLAN["data"],
               "--probes", str(DIAG / "probes.json"), "--model", case["model"],
               "--label", label, "--expected-validation", str(case["expected_validation"]),
               "--output", str(DIAG / ("eval_" + label))]
    if case["revision"]:
        command += ["--revision", case["revision"]]
    with (DIAG / ("eval_" + label + ".log")).open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        summary.update(active_case=label, evaluator_pid=process.pid)
        save()
        code = process.wait()
    if code:
        summary.update(status="failed", failed_case=label, exit_code=code,
                       finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        sys.exit(code)
    row = json.loads((DIAG / ("eval_" + label) / "result.json").read_text())
    assert row["status"] == "complete" and row["training_updates"] == 0
    assert abs(row["metrics"]["uniform_train"]["loss"] - uniform_expected[case["step"]]) <= 1e-6
    summary["cases"][label] = {"step": case["step"], "trained_tokens": case["trained_tokens"],
                               "metrics": row["metrics"], "probes_sha256": row["probes_sha256"],
                               "result": str(DIAG / ("eval_" + label) / "result.json")}
    save()
summary.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
               active_case=None, evaluator_pid=None)
base, early, late = (summary["cases"][case["label"]] for case in PLAN["cases"])
summary["loss_changes"] = {
    name: {"base_to_early": early["metrics"][name]["loss"] - base["metrics"][name]["loss"],
           "early_to_late": late["metrics"][name]["loss"] - early["metrics"][name]["loss"],
           "base_to_late": late["metrics"][name]["loss"] - base["metrics"][name]["loss"]}
    for name in base["metrics"]}
save()
print(json.dumps({"status": "complete", "loss_changes": summary["loss_changes"]}), flush=True)
