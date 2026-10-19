"""Freeze seen/unseen block samples for two immutable CPT checkpoints."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

DIAG = Path(__file__).resolve().parents[1]
PLAN = json.loads((DIAG / "plan.json").read_text())
RUN = Path(PLAN["production_run"])
DATA = Path(PLAN["data"])
manifest = json.loads((RUN / "manifest.json").read_text())
data_manifest = json.loads((DATA / "manifest.json").read_text())
early, late = PLAN["early_step"], PLAN["late_step"]
batch, length, blocks = PLAN["effective_training_batch"], PLAN["sequence_length"], PLAN["probe_blocks"]
assert early < late < manifest["epoch_steps"]
assert batch == manifest["signature"]["effective_batch"] and length == manifest["signature"]["sequence_length"]
order_path = RUN / "orders" / manifest["order_metadata"]["files"][0]["name"]
order_sha = hashlib.sha256(order_path.read_bytes()).hexdigest()
assert order_sha == manifest["order_metadata"]["files"][0]["sha256"]
order = np.load(order_path, mmap_mode="r")
tokens = np.load(DATA / "train_tokens.npy", mmap_mode="r")
full = len(tokens) // length
assert tokens.dtype == np.uint32 and order.dtype == np.int32
assert len(order) == full + 1 and len(tokens) == 5_000_000_000
assert np.array_equal(np.sort(order), np.arange(len(order), dtype=np.int32))
assert not np.any(order[:late * batch] == full), "Tail is not in this comparison interval"
positions = np.empty(len(order), dtype=np.int64)
positions[order] = np.arange(len(order), dtype=np.int64)
seen_pool = order[:early * batch]
unseen_pool = order[late * batch:]
unseen_pool = unseen_pool[unseen_pool < full]
rng = np.random.default_rng(PLAN["probe_seed"])
seen = np.sort(rng.choice(seen_pool, size=blocks, replace=False)).astype(np.int64)
unseen = np.sort(rng.choice(unseen_pool, size=blocks, replace=False)).astype(np.int64)
uniform = np.asarray(manifest["train_probe"]["indices"], dtype=np.int64)
assert np.intersect1d(seen, unseen).size == 0
assert np.all(positions[seen] < early * batch)
assert np.all(positions[unseen] >= late * batch)


def group(indices, sampling, seed):
    assert len(indices) == blocks and len(np.unique(indices)) == blocks
    assert np.all(indices[:-1] < indices[1:]) and np.all(indices < full)
    digest = hashlib.sha256()
    for index in indices:
        start = int(index) * length
        digest.update(memoryview(np.ascontiguousarray(tokens[start:start + length])).cast("B"))
    return {"indices": indices.tolist(), "blocks": blocks, "seed": seed,
            "sampling": sampling,
            "indices_sha256": hashlib.sha256(memoryview(indices).cast("B")).hexdigest(),
            "tokens_sha256": digest.hexdigest(),
            "input_tokens": blocks * length, "supervised_tokens": blocks * (length - 1),
            "epoch0_order_positions": positions[indices].tolist(),
            "seen_blocks_at_early_step": int(np.count_nonzero(positions[indices] < early * batch)),
            "seen_blocks_at_late_step": int(np.count_nonzero(positions[indices] < late * batch))}


groups = {
    "uniform_train": group(uniform, "Original immutable uniform training-corpus probe", manifest["train_probe"]["seed"]),
    "seen_train": group(seen, f"Uniform without replacement from order positions [0,{early * batch}); all seen by early snapshot", PLAN["probe_seed"]),
    "unseen_train": group(unseen, f"Uniform without replacement from order positions [{late * batch},{len(order)}); tail excluded; none seen by late snapshot", PLAN["probe_seed"]),
}
for key in ("indices_sha256", "tokens_sha256", "blocks", "input_tokens", "supervised_tokens"):
    assert groups["uniform_train"][key] == manifest["train_probe"][key]
assert groups["seen_train"]["seen_blocks_at_early_step"] == blocks
assert groups["seen_train"]["seen_blocks_at_late_step"] == blocks
assert groups["unseen_train"]["seen_blocks_at_early_step"] == 0
assert groups["unseen_train"]["seen_blocks_at_late_step"] == 0
result = {
    "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
    "production_run": str(RUN), "data": str(DATA), "early_step": early, "late_step": late,
    "probe_seed": PLAN["probe_seed"], "sequence_length": length, "effective_batch": batch,
    "data_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
    "train_token_file_sha256_from_manifest": data_manifest["artifacts"]["train_tokens.npy"]["sha256"],
    "full_5b_token_file_rehashed": False, "order_path": str(order_path), "order_sha256": order_sha,
    "seen_population_blocks": len(seen_pool), "unseen_population_full_blocks": len(unseen_pool),
    "groups": groups, "private_rng_does_not_touch_production_rng": True,
    "seen_and_unseen_disjoint": True,
    "limits": PLAN["limits"],
}
assert not (DIAG / "probes.json").exists()
(DIAG / "probes.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"groups": {name: {k: row[k] for k in ("blocks", "seen_blocks_at_early_step", "seen_blocks_at_late_step", "indices_sha256", "tokens_sha256")} for name, row in groups.items()}}))
