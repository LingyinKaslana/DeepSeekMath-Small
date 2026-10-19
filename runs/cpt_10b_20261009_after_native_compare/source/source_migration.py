"""Strict resume signatures with an explicit, auditable source-only exception."""
from datetime import datetime, timezone
from pathlib import Path
import re


def without_source(signature):
    return {key: value for key, value in signature.items() if key != "source_sha256"}


def authorize_signature(saved, current, allow_source_migration=False):
    if saved == current:
        return False
    if (not allow_source_migration or without_source(saved) != without_source(current)
            or not isinstance(saved.get("source_sha256"), dict)
            or not isinstance(current.get("source_sha256"), dict)):
        raise ValueError("Resume configuration/data/LR/seed/source differs from the saved training state; "
                         "--allow-source-migration permits source_sha256 changes only")
    return True


def validate_lineage(state):
    """Return original signature and a continuous chain of source-only changes."""
    signature = state["signature"]
    original = state.get("original_signature", signature)
    migrations = state.get("source_migrations", [])
    if not isinstance(migrations, list) or not isinstance(original, dict):
        raise ValueError("Invalid source-migration audit")
    previous = original
    for entry in migrations:
        if (entry.get("from_signature") != previous or entry.get("changed_fields") != ["source_sha256"]
                or entry.get("authorized_by") != "--allow-source-migration"
                or not isinstance(entry.get("completed_steps"), int)
                or isinstance(entry.get("completed_steps"), bool) or entry["completed_steps"] < 0):
            raise ValueError("Invalid or discontinuous source-migration audit")
        following = entry.get("to_signature")
        if not isinstance(following, dict) or not authorize_signature(previous, following, True):
            raise ValueError("A source-migration audit must record an actual source-only change")
        for name in ("from_source_directory", "to_source_directory", "resumed_checkpoint", "saved_utc"):
            if not isinstance(entry.get(name), str) or not entry[name]:
                raise ValueError(f"Missing source-migration audit {name}")
        previous = following
    if previous != signature:
        raise ValueError("Original signature / migration chain differs from current signature")
    if migrations and any(a["completed_steps"] > b["completed_steps"] for a, b in zip(migrations, migrations[1:])):
        raise ValueError("Source-migration steps are not monotonic")
    if migrations and migrations[-1]["completed_steps"] > state.get("completed_steps", -1):
        raise ValueError("Source migration lies beyond the saved training prefix")
    return original, migrations


def apply_source_migration(state, current, checkpoint, output):
    original, migrations = validate_lineage(state)
    changed = authorize_signature(state["signature"], current, True)
    state["original_signature"] = original
    if changed:
        entry = {"authorized_by": "--allow-source-migration", "changed_fields": ["source_sha256"],
                 "completed_steps": state["completed_steps"], "from_signature": state["signature"],
                 "to_signature": current, "resumed_checkpoint": str(checkpoint.resolve()),
                 "from_source_directory": str((checkpoint.parent / "source").resolve()),
                 "to_source_directory": str((output / "source").resolve()),
                 "saved_utc": datetime.now(timezone.utc).isoformat()}
        migrations = [*migrations, entry]
    state["signature"], state["source_migrations"] = current, migrations
    validate_lineage(state)
    return migrations


def verify_lineage_snapshots(state, sha):
    original, migrations = validate_lineage(state)
    for entry in migrations:
        for side in ("from", "to"):
            directory = Path(entry[f"{side}_source_directory"])
            for name, digest in entry[f"{side}_signature"]["source_sha256"].items():
                if (Path(name).name != name or not isinstance(digest, str)
                        or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                    raise ValueError("Invalid source-migration snapshot hash")
                path = directory / name
                if not path.is_file() or sha(path) != digest:
                    raise ValueError(f"Source-migration historical snapshot mismatch: {path}")
    return {"original_signature": original, "source_migrations": migrations,
            "source_only_migration_count": len(migrations)}
