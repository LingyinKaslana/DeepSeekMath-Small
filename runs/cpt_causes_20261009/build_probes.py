"""Generate fixed train-block probes using only NumPy and CPU mmap reads."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("probes.json")
RUN = ROOT / "runs/cpt_native_from0_5h_20261009"
DATA = ROOT / "data/processed/openwebmath_5b_s42"
START, END, BATCH, LENGTH, BLOCKS = 20000, 21000, 16, 2048, 256
SEED = 20261010
UNSEEN_EXCLUSION_STEP = 30860


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


manifest = json.loads((RUN / "manifest.json").read_text())
data_manifest = json.loads((DATA / "manifest.json").read_text())
uniform_source = json.loads((RUN / "train_probe.json").read_text())
order_path = RUN / "orders/order_epoch_00000.npy"
order_sha = sha_file(order_path)
assert order_sha == manifest["order_metadata"]["files"][0]["sha256"]
order = np.load(order_path, mmap_mode="r")
tokens = np.load(DATA / "train_tokens.npy", mmap_mode="r")
full = len(tokens) // LENGTH
tail = full
assert len(tokens) == 5_000_000_000 and len(order) == full + 1
assert full == 2441406 and len(order) == 2441407
assert order.dtype == np.int32 and tokens.dtype == np.uint32
assert LENGTH == manifest["signature"]["sequence_length"] and BATCH == manifest["signature"]["effective_batch"]
assert np.array_equal(np.sort(np.asarray(order)), np.arange(len(order), dtype=np.int32))
start_position, end_position = START * BATCH, END * BATCH
unseen_exclusion_position = UNSEEN_EXCLUSION_STEP * BATCH
assert not np.any(order[:end_position] == tail)
seen_pool = np.asarray(order[:start_position])
future_order = np.asarray(order[unseen_exclusion_position:])
unseen_pool = future_order[future_order != tail]
# These private generator calls do not read or change NumPy's global RNG,
# any Torch RNG, or any training state.
rng = np.random.default_rng(SEED)
seen = np.sort(rng.choice(seen_pool, size=BLOCKS, replace=False)).astype(np.int64)
unseen = np.sort(rng.choice(unseen_pool, size=BLOCKS, replace=False)).astype(np.int64)
uniform = np.asarray(uniform_source["indices"], dtype=np.int64)
assert np.intersect1d(seen, unseen).size == 0
position_by_index = np.empty(len(order), dtype=np.int64)
position_by_index[order] = np.arange(len(order), dtype=np.int64)
assert np.all(position_by_index[seen] < start_position)
assert np.all(position_by_index[unseen] >= unseen_exclusion_position)
assert np.all(unseen < full) and np.all(seen < full)


def group(indices, name, seed, population, draw):
    indices = np.asarray(indices, dtype=np.int64)
    assert len(indices) == BLOCKS and len(np.unique(indices)) == BLOCKS
    assert np.all(indices[:-1] < indices[1:]) and np.all(indices < full)
    digest = hashlib.sha256()
    for block_index in indices:
        start = int(block_index) * LENGTH
        digest.update(memoryview(np.ascontiguousarray(tokens[start:start + LENGTH])).cast("B"))
    positions = position_by_index[indices]
    return {"schema_version": 1, "name": name, "seed": seed, "blocks": BLOCKS,
            "indices": indices.tolist(),
            "indices_sha256": hashlib.sha256(memoryview(indices).cast("B")).hexdigest(),
            "tokens_sha256": digest.hexdigest(),
            "input_tokens": BLOCKS * LENGTH, "supervised_tokens": BLOCKS * (LENGTH - 1),
            "sampling": population, "private_rng_draw": draw,
            "epoch_0_order_positions": positions.tolist(),
            "order_position_min": int(positions.min()), "order_position_max": int(positions.max()),
            "seen_blocks_at_start_step": int(np.count_nonzero(positions < start_position)),
            "seen_blocks_at_end_step": int(np.count_nonzero(positions < end_position)),
            "seen_blocks_at_native_5h_final_step_30860": int(np.count_nonzero(positions < 30860 * BATCH))}


groups = {
    "uniform_train": group(uniform, "uniform_train", uniform_source["seed"],
                           "Exact original fixed uniform full-corpus probe retained, seed 20261009", "original run, unchanged"),
    "seen_train": group(seen, "seen_train", SEED,
                        "Uniform without replacement from epoch-0 order positions [0,320000), all already trained at step 20000", "first choice using private default_rng(20261010)"),
    "unseen_train": group(unseen, "unseen_train", SEED,
                          "Uniform without replacement from epoch-0 order positions [493760,2441407), tail excluded; no blocks trained through step 30860", "second choice from the same private generator"),
}
for field in ["indices_sha256", "tokens_sha256", "input_tokens", "supervised_tokens", "blocks"]:
    assert groups["uniform_train"][field] == uniform_source[field]
assert groups["seen_train"]["seen_blocks_at_start_step"] == BLOCKS
assert groups["seen_train"]["seen_blocks_at_end_step"] == BLOCKS
assert groups["unseen_train"]["seen_blocks_at_start_step"] == 0
assert groups["unseen_train"]["seen_blocks_at_end_step"] == 0
assert groups["unseen_train"]["seen_blocks_at_native_5h_final_step_30860"] == 0
report = {
    "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Fixed train-corpus probes for matched native steps 20000 to 21000; diagnostic, not production data changes",
    "gpu_execution": False, "torch_imported_or_rng_modified": False,
    "start_step": START, "end_step": END, "effective_batch": BATCH,
    "unseen_exclusion_through_step": UNSEEN_EXCLUSION_STEP,
    "sequence_length": LENGTH, "epoch": 0, "groups": groups,
    "data": str(DATA), "run": str(RUN),
    "data_manifest_sha256": sha_file(DATA / "manifest.json"),
    "train_token_file_sha256_from_existing_manifest": data_manifest["artifacts"]["train_tokens.npy"]["sha256"],
    "full_train_token_file_rehashed": False,
    "order_path": str(order_path), "order_sha256": order_sha,
    "hash_encoding": {"indices": "Sorted native little-endian int64 bytes, C contiguous",
                      "tokens": "Concatenate native little-endian uint32 full blocks in sorted index order; SHA256 raw bytes, no NPY header"},
    "cursor_proof": {"start_consumed_order_positions": [0, start_position],
                     "branch_train_order_positions": [start_position, end_position],
                     "branch_updates": END - START,
                     "branch_input_tokens": (END - START) * BATCH * LENGTH,
                     "branch_supervised_tokens": (END - START) * BATCH * (LENGTH - 1),
                     "start_trained_input_tokens": START * BATCH * LENGTH,
                     "end_trained_input_tokens": END * BATCH * LENGTH,
                     "seen_population_blocks": len(seen_pool), "unseen_population_full_blocks": len(unseen_pool),
                     "tail_index": tail, "tail_epoch0_order_position": int(position_by_index[tail]),
                     "order_is_exact_permutation": True,
                     "seen_and_unseen_disjoint": True,
                     "tail_absent_from_training_window": True,
                     "seen_probe_within_start_prefix": True,
                     "unseen_probe_absent_through_end_prefix": True},
    "read_only_evaluation_recommendation": {
        "models": ["Pinned base", "Native checkpoint_step_00020000", "Native stopped_checkpoint step 30860"],
        "mode": "Ordinary unpatched HF, FP32 parameters, BF16 autocast, sdpa, 2D all-ones mask, labels shifted once by HF; eval mode, no gradients/optimizer, same token-weighted loss as original run",
        "micro_batch": 4,
        "groups": ["uniform_train", "seen_train", "unseen_train"],
        "interpretation": "Seen/unseen status is exact by block training history through step 30860: all seen_train blocks trained before step 20000; no unseen_train block trained even at step 30860. Similar documents elsewhere in the corpus are not ruled out.",
        "within_group_comparison": "Compare each fixed group across model snapshots; differences between groups alone may reflect sample difficulty and do not measure overfitting.",
        "independent_validation": "Keep original validation loss alongside these training-corpus diagnostic groups; do not replace validation."},
}
tmp = OUT.with_suffix(".json.tmp")
tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
tmp.replace(OUT)
print(json.dumps({"output": str(OUT), "output_sha256": sha_file(OUT),
                  "groups": {k: {key: value[key] for key in ["indices_sha256", "tokens_sha256", "seen_blocks_at_start_step", "seen_blocks_at_end_step", "seen_blocks_at_native_5h_final_step_30860", "order_position_min", "order_position_max"]} for k, value in groups.items()}}, ensure_ascii=False))
