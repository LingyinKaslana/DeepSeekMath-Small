"""Record this independent production run's actual exit and verified outcome."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--stage", required=True)
parser.add_argument("--exit-code", type=int, required=True)
args = parser.parse_args()
result = read_json(args.run / "result.json")
verification = read_json(args.setup / "final_checkpoint_verification.json")
budget_complete = (
    result.get("status") == "complete"
    and result.get("trained_tokens") == 10_000_000_000
    and result.get("completed_steps") == 305176
    and result.get("optimizer_updates") == 305176
    and result.get("all_dataset_tokens_consumed") is True
)
postflight_passed = (
    args.stage == "finished" and args.exit_code == 0
    and verification.get("status") == "passed"
)
status = (
    "complete_verified" if budget_complete and postflight_passed
    else "stopped_verified" if postflight_passed and result.get("status") == "stopped"
    else "postflight_failed" if result.get("status") in {"complete", "stopped"}
    else "failed"
)
record = {
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "stage": args.stage,
    "exit_code": args.exit_code,
    "run": str(args.run),
    "training_status": result.get("status"),
    "stop_reason": result.get("stop_reason"),
    "completed_steps": result.get("completed_steps"),
    "optimizer_updates": result.get("optimizer_updates"),
    "trained_tokens": result.get("trained_tokens"),
    "configured_input_tokens": 10_000_000_000,
    "production_budget_complete": budget_complete,
    "final_checkpoint_verified": verification.get("status") == "passed",
    "checkpoint": result.get("checkpoint"),
    "checkpoint_verification": str(args.setup / "final_checkpoint_verification.json"),
    "report": str(args.setup / "report"),
    "prior_production_resumed": True,
}
(args.setup / "run_status.json").write_text(json.dumps(record, indent=2) + "\n")
launch = read_json(args.setup / "launch.json")
launch.update(status=status, finished_utc=record["updated_utc"], outcome=record)
(args.setup / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
