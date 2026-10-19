"""Read a running CPT job's progress and estimate time from measured wall time."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time


def read_json(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text()) if path.exists() else None
        except (json.JSONDecodeError, FileNotFoundError):
            # Live writer may be between truncate and write.
            if attempt < 2:
                time.sleep(0.01)
    return None


def first_row(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        for line in stream:
            try:
                return json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return None


def recent_rows(path, byte_budget=1_000_000):
    if not path.exists():
        return []
    with path.open("rb") as stream:
        size = path.stat().st_size
        offset = max(0, size - byte_budget)
        stream.seek(offset)
        if offset:
            stream.readline()
        lines = stream.read().splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A currently buffered partial record is not completed progress.
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.run / "manifest.json")
    result = read_json(args.run / "result.json")
    progress = read_json(args.run / "progress.json")
    latest = read_json(args.run / "latest_checkpoint.json")
    if manifest is None:
        print(json.dumps({"status": "not_started", "run": str(args.run.resolve())}, indent=2))
        return
    rows = recent_rows(args.run / "training.jsonl")
    last = rows[-1] if rows else progress
    out = {"status": result.get("status") if result else "in_progress",
           "run": str(args.run.resolve()), "started_utc": manifest["started_utc"],
           "configured_input_tokens": manifest.get("configured_input_tokens"),
           "configured_steps": manifest.get("schedule", {}).get("total_steps"),
           "latest_checkpoint": latest,
           "note": "Read-only progress; in_progress does not by itself prove process liveness"}
    if result:
        out["result"] = {key: result.get(key) for key in (
            "status", "stop_reason", "completed_steps", "trained_tokens",
            "steady_tokens_per_second", "wall_seconds_this_invocation", "checkpoint", "error")}
    if last:
        out["progress"] = last
        regular = [row for row in rows[-2000:]
                   if not row.get("first_full_shape_update") and row.get("seconds", 0) > 0]
        if regular:
            out["recent_update_tokens_per_second"] = (
                sum(row["input_tokens"] for row in regular) / sum(row["seconds"] for row in regular))
            out["recent_update_timing_note"] = "Update timings; excludes evaluation, saves and some logging"
        start = datetime.fromisoformat(manifest["started_utc"])
        now = datetime.now(timezone.utc)
        elapsed = (result.get("wall_seconds_this_invocation", 0) if result
                   else (now - start).total_seconds())
        prefix = manifest.get("resumed_input_tokens", 0)
        if not prefix and manifest.get("resumed_completed_steps", 0):
            first = first_row(args.run / "training.jsonl")
            prefix = first.get("cumulative_tokens", 0) - first.get("input_tokens", 0) if first else 0
        consumed = last.get("cumulative_tokens", 0)
        this_invocation = consumed - prefix
        total = manifest.get("configured_input_tokens", 0)
        if result is None and regular:
            out["estimated_remaining_training_hours"] = (
                max(0, total - consumed) / out["recent_update_tokens_per_second"] / 3600)
            out["training_estimate_note"] = "Excludes future evaluations, checkpoints and restarts"
        if this_invocation > 0 and elapsed > 0:
            speed = this_invocation / elapsed
            out.update(wall_tokens_per_second_this_invocation=speed,
                       wall_hours_this_invocation=elapsed / 3600)
            if result is None:
                cadence = [manifest.get("arguments", {}).get(key)
                           for key in ("eval_every", "checkpoint_every")]
                cadence = [value for value in cadence if isinstance(value, int) and value > 0]
                observation_steps = max(1000, min(cadence) if cadence else 1000)
                steps_seen = last.get("step", 0) - manifest.get("resumed_completed_steps", 0)
                if steps_seen > observation_steps:
                    remaining = max(0, total - consumed) / speed
                    out.update(estimated_remaining_wall_hours=remaining / 3600,
                               estimated_finish_utc=(now + timedelta(seconds=remaining)).isoformat(),
                               estimate_note="Measured invocation wall rate includes startup, evaluations and saves")
                else:
                    out["estimate_note"] = "Wall estimate waits for a full evaluation/checkpoint interval; startup cost is fixed"
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
