"""Build restartable, deduplicated OpenWebMath token budgets without loading corpus texts.

Workers batch-tokenize deterministic per-shard samples into bounded disk caches.
The coordinator merges in a fixed shuffled shard order, with SQLite enforcing
exact normalized-text and URL uniqueness across both splits. Every old pilot
validation document is reserved, and its token file is preserved as a prefix.
New validation uses a fixed hash threshold independent of the requested budget.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
import pyarrow.parquet as pq

SCHEMA = 1
PACKING = "document text + EOS, concatenated within each split; no cross-document attention mask"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def deterministic_seed(seed, name):
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16], 16)


def load_tokenizer(model, revision, threads):
    # Explicitly override activate.sh's serial tokenizer setting in CPU workers.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError("A fast local tokenizer with EOS is required")
    # Documents are concatenated and packed later; tokenization is not a
    # model forward and can exceed the model context length without truncation.
    tokenizer.model_max_length = 1 << 60
    return tokenizer


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def worker(job):
    directory = Path(job["cache"])
    directory.mkdir(parents=True, exist_ok=True)
    # An orphaned worker may finish its durable cache after a coordinator dies.
    # A resumed coordinator waits for that same shard instead of racing it.
    cache_lock = (directory / ".worker.lock").open("a+")
    fcntl.flock(cache_lock.fileno(), fcntl.LOCK_EX)
    progress_path = directory / "progress.json"
    tokenizer = load_tokenizer(job["model"], job["revision"], job["threads"])
    source_path = Path(job["source_path"])
    # Source digest is checked before ever using a shard, not just trusted from manifest.
    if source_path.stat().st_size != job["source_bytes"] or sha(source_path) != job["source_sha256"]:
        raise ValueError(f"Source file changed: {source_path}")
    pilot_records = list(read_jsonl(job["pilot_jsonl"]))  # only the fixed 334-document pilot validation
    reserved_hashes = {row["text_sha256"] for row in pilot_records}
    reserved_urls = {row["url"] for row in pilot_records if row["url"]}
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
        if state["signature"] != job["signature"]:
            raise ValueError("Worker cache signature changed")
        if state["status"] == "complete" and all(state["tokens"][split] >= job["candidate_tokens"][split]
                                                   for split in ("train", "validation")):
            return state
        state["status"] = "building"
    else:
        state = {"signature": job["signature"], "shard": job["name"], "status": "building",
                 "group_cursor": 0, "row_cursor": 0, "tokens": {"train": 0, "validation": 0},
                 "documents": {"train": 0, "validation": 0},
                 "metadata_bytes": {"train": 0, "validation": 0},
                 "scanned_documents": 0, "rejected": {}, "started_utc": stamp(),
                 "source_sha256": job["source_sha256"]}
    handles = {}
    for split in ("train", "validation"):
        for suffix, count in (("tokens.raw", state["tokens"][split] * 4),
                              ("jsonl", state["metadata_bytes"][split])):
            path = directory / f"{split}.{suffix}"
            handle = path.open("r+b" if path.exists() else "w+b")
            if handle.seek(0, 2) < count:
                raise ValueError("Committed worker data was truncated")
            handle.truncate(count)
            handle.seek(count)
            handles[split, suffix] = handle
    reader = pq.ParquetFile(source_path)
    groups = list(range(reader.num_row_groups))
    random.Random(deterministic_seed(job["seed"], job["name"])).shuffle(groups)
    rejected = Counter(state["rejected"])
    begin = time.monotonic()
    added_tokens = 0
    try:
        for group_cursor in range(state["group_cursor"], len(groups)):
            group = groups[group_cursor]
            rows = reader.read_row_group(group, columns=["text", "url"]).to_pylist()
            order = list(range(len(rows)))
            random.Random(deterministic_seed(job["seed"], f"{job['name']}:{group}")).shuffle(order)
            start_row = state["row_cursor"] if group_cursor == state["group_cursor"] else 0
            for batch_start in range(start_row, len(order), job["batch_documents"]):
                batch_end = min(batch_start + job["batch_documents"], len(order))
                candidates = []
                for row_cursor in range(batch_start, batch_end):
                    row_index = order[row_cursor]
                    row = rows[row_index]
                    text, url = row["text"], row["url"]
                    state["scanned_documents"] += 1
                    if not isinstance(text, str) or not text.strip():
                        rejected["empty"] += 1
                        continue
                    if len(text) > job["maximum_characters"]:
                        rejected["over_maximum_characters"] += 1
                        continue
                    digest = text_hash(text)
                    if digest in reserved_hashes or (url and url in reserved_urls):
                        rejected["reserved_pilot_validation"] += 1
                        continue
                    split = "validation" if int(digest[:16], 16) < job["validation_threshold"] else "train"
                    if state["tokens"][split] >= job["candidate_tokens"][split]:
                        continue
                    candidates.append((text, url, digest, split, row_index))
                if candidates:
                    encoded = tokenizer([row[0] for row in candidates], add_special_tokens=False,
                                        return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                    for (text, url, digest, split, row_index), ids in zip(candidates, encoded, strict=True):
                        if state["tokens"][split] >= job["candidate_tokens"][split]:
                            continue
                        if len(ids) < job["minimum_document_tokens"]:
                            rejected["under_minimum_document_tokens"] += 1
                            continue
                        ids.append(tokenizer.eos_token_id)
                        record = {"url": url, "text_sha256": digest, "source_shard": job["name"],
                                  "row_group": group, "row_in_group": row_index,
                                  "document_tokens_with_eos": len(ids),
                                  "candidate_token_offset": state["tokens"][split]}
                        if split == "validation":
                            record["text"] = text
                        handles[split, "tokens.raw"].write(np.asarray(ids, dtype=np.uint32).tobytes())
                        handles[split, "jsonl"].write((json.dumps(record, ensure_ascii=False) + "\n").encode())
                        state["tokens"][split] += len(ids)
                        state["documents"][split] += 1
                        added_tokens += len(ids)
                state["group_cursor"], state["row_cursor"] = group_cursor, batch_end
                if all(state["tokens"][split] >= job["candidate_tokens"][split] for split in ("train", "validation")):
                    break
            # One row group (approximately 1,000 rows) is the restart unit.
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
            state["metadata_bytes"] = {split: handles[split, "jsonl"].tell() for split in ("train", "validation")}
            state["rejected"] = dict(rejected)
            state["updated_utc"] = stamp()
            state["last_session_seconds"] = time.monotonic() - begin
            state["last_session_tokens_per_second"] = added_tokens / max(1e-9, state["last_session_seconds"])
            if all(state["tokens"][split] >= job["candidate_tokens"][split] for split in ("train", "validation")):
                state["status"] = "complete"
                atomic_json(progress_path, state)
                return state
            state["group_cursor"], state["row_cursor"] = group_cursor + 1, 0
            atomic_json(progress_path, state)
        state["status"] = "source_exhausted"
        atomic_json(progress_path, state)
        return state
    finally:
        for handle in handles.values():
            handle.close()


def initialize_database(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA cache_size=-131072")
    connection.execute("""CREATE TABLE IF NOT EXISTS selected (hash TEXT PRIMARY KEY, url TEXT UNIQUE, split TEXT NOT NULL,
                        token_offset INTEGER NOT NULL, used_tokens INTEGER NOT NULL, shard TEXT NOT NULL)""")
    connection.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
    connection.commit()
    return connection


def save_coordinator(connection, state, arrays, handles, output):
    for array in arrays.values():
        array.flush()
    for split in arrays:
        descriptor = os.open(output / f"{split}_tokens.npy", os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        handles[split].flush()
        os.fsync(handles[split].fileno())
        state["metadata_bytes"][split] = handles[split].tell()
    state["updated_utc"] = stamp()
    connection.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (json.dumps(state),))
    connection.commit()  # commit is authoritative after every merged shard
    atomic_json(output / "progress.json", state)


def insert_record(connection, arrays, handles, split, record, tokens, remaining, state):
    if remaining <= 0:
        return 0
    used = min(len(tokens), remaining)
    offset = state["tokens"][split]
    try:
        connection.execute("INSERT INTO selected VALUES (?, ?, ?, ?, ?, ?)",
                           (record["text_sha256"], record.get("url") or None, split, offset, used, record["source_shard"]))
    except sqlite3.IntegrityError:
        state["rejected_global_text_or_url_duplicates"] += 1
        return 0
    arrays[split][offset:offset + used] = tokens[:used]
    result = {key: value for key, value in record.items() if key != "candidate_token_offset"}
    result.update(used_tokens=used, token_offset=offset, prefix_truncated=used < record["document_tokens_with_eos"])
    handles[split].write((json.dumps(result, ensure_ascii=False) + "\n").encode())
    state["tokens"][split] += used
    state["documents"][split] += 1
    return used


def source_document(source, record, cache):
    key = record["source_shard"], record["row_group"]
    if key != cache.get("key"):
        reader = pq.ParquetFile(Path(source["snapshot"]) / key[0])
        cache["rows"] = reader.read_row_group(key[1], columns=["text", "url"]).to_pylist()
        cache["key"] = key
    return cache["rows"][record["row_in_group"]]


def verify(output, source, tokenizer, connection, totals, block_length, pilot, files):
    checks, samples = {}, {}
    # Every metadata record is checked against the committed uniqueness database.
    for split, expected in totals.items():
        array = np.load(output / f"{split}_tokens.npy", mmap_mode="r")
        if array.dtype != np.uint32 or array.shape != (expected,):
            raise ValueError("Token array layout does not match target")
        for start in range(0, expected, 8_388_608):
            if int(array[start:start + 8_388_608].max()) >= len(tokenizer):
                raise ValueError("Out-of-vocabulary token found")
        offset = documents = truncated = 0
        sample_records = {}
        for record in read_jsonl(output / f"{split}.jsonl"):
            used = record["used_tokens"]
            if record["token_offset"] != offset or not 0 < used <= record["document_tokens_with_eos"]:
                raise ValueError("Document token offsets are not contiguous")
            expected_database_row = (record.get("url") or None, split, offset, used, record["source_shard"])
            actual = connection.execute("SELECT url,split,token_offset,used_tokens,shard FROM selected WHERE hash=?",
                                        (record["text_sha256"],)).fetchone()
            if actual != expected_database_row:
                raise ValueError("Metadata/database provenance mismatch")
            shard = record["source_shard"]
            bucket = sample_records.setdefault(shard, {"first": record, "middle": record, "last": record, "count": 0})
            bucket["last"], bucket["count"] = record, bucket["count"] + 1
            # Keep an additional reproducible reservoir sample without storing documents.
            if deterministic_seed(42, record["text_sha256"]) % bucket["count"] == 0:
                bucket["middle"] = record
            if split == "validation":
                ids = tokenizer.encode(record["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
                if text_hash(record["text"]) != record["text_sha256"] or len(ids) != record["document_tokens_with_eos"]:
                    raise ValueError("Validation source text or tokenizer changed")
                np.testing.assert_array_equal(array[offset:offset + used], np.asarray(ids[:used], dtype=np.uint32))
            offset += used
            documents += 1
            truncated += int(record["prefix_truncated"])
        if offset != expected or documents != connection.execute("SELECT count(*) FROM selected WHERE split=?", (split,)).fetchone()[0]:
            raise ValueError("Metadata totals do not match arrays/database")
        if split == "train" and set(sample_records) != {file["name"] for file in files}:
            raise ValueError("Not all requested source shards contributed training tokens")
        cache, roundtrips = {}, 0
        for bucket in sample_records.values():
            for record in {row["text_sha256"]: row for row in (bucket["first"], bucket["middle"], bucket["last"])}.values():
                row = source_document(source, record, cache)
                if text_hash(row["text"]) != record["text_sha256"] or row["url"] != record["url"]:
                    raise ValueError("Source record provenance does not roundtrip")
                ids = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
                if len(ids) != record["document_tokens_with_eos"]:
                    raise ValueError("Source token length does not roundtrip")
                start, take = record["token_offset"], record["used_tokens"]
                np.testing.assert_array_equal(array[start:start + take], np.asarray(ids[:take], dtype=np.uint32))
                roundtrips += 1
        checks[split] = {"tokens": expected, "documents": documents, "full_blocks": expected // block_length,
                         "tokens_in_full_blocks": expected // block_length * block_length,
                         "remainder_tokens": expected % block_length, "prefix_truncated_documents": truncated,
                         "source_shards": len(sample_records), "source_token_roundtrip_samples": roundtrips,
                         "all_metadata_records_audited": True, "all_token_ids_range_checked": True,
                         "all_document_token_roundtrip": split == "validation"}
    pilot_array = np.load(pilot / "validation_tokens.npy", mmap_mode="r")
    new_array = np.load(output / "validation_tokens.npy", mmap_mode="r")
    np.testing.assert_array_equal(new_array[:len(pilot_array)], pilot_array)
    for record in read_jsonl(pilot / "validation.jsonl"):
        entry = connection.execute("SELECT split FROM selected WHERE hash=?", (record["text_sha256"],)).fetchone()
        if entry != ("validation",):
            raise ValueError("Old validation text was lost or contaminated training")
        if record["url"]:
            if connection.execute("SELECT split FROM selected WHERE url=?", (record["url"],)).fetchone() != ("validation",):
                raise ValueError("Old validation URL was lost or contaminated training")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("Uniqueness index failed its integrity check")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pilot-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--model-revision", default="df3ce67c0e24480f20468b6ef2894622d69eb73b")
    parser.add_argument("--train-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--validation-tokens", type=int, default=2_097_152)
    parser.add_argument("--block-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tokenizer-threads", type=int, default=4)
    parser.add_argument("--batch-documents", type=int, default=256)
    parser.add_argument("--candidate-reserve-factor", type=float, default=1.20)
    parser.add_argument("--minimum-document-tokens", type=int, default=64)
    parser.add_argument("--maximum-characters", type=int, default=200_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-shards", type=int)
    parser.add_argument("--max-shards", type=int, help="Smoke tests only; default processes all source shards")
    args = parser.parse_args()
    if min(args.train_tokens, args.validation_tokens, args.block_length, args.workers, args.tokenizer_threads, args.batch_documents) <= 0:
        raise ValueError("Budgets and resource settings must be positive")
    if args.workers * args.tokenizer_threads > (os.cpu_count() or 1) or not 0 < args.validation_fraction < 1:
        raise ValueError("Tokenizer thread total exceeds CPU count or split fraction invalid")
    if args.candidate_reserve_factor < 1 or args.minimum_document_tokens < 1:
        raise ValueError("Invalid reserve factor or minimum document length")
    source = json.loads(args.source_manifest.read_text())
    if source["status"] != "verified":
        raise ValueError("Source manifest must be verified")
    pilot_manifest = json.loads((args.pilot_data / "manifest.json").read_text())
    if (pilot_manifest["tokenizer"], pilot_manifest["tokenizer_revision"]) != (args.model, args.model_revision):
        raise ValueError("Pilot tokenizer does not match selected tokenizer")
    for name in ("validation.jsonl", "validation_tokens.npy"):
        if sha(args.pilot_data / name) != pilot_manifest["artifacts"][name]["sha256"]:
            raise ValueError("Reserved pilot validation artifact changed")
    pilot_tokens = pilot_manifest["splits"]["validation"]["tokens"]
    if args.validation_tokens < pilot_tokens:
        raise ValueError("New validation budget cannot drop old pilot validation")
    files = sorted(source["files"], key=lambda file: file["name"])
    random.Random(args.seed).shuffle(files)
    if args.max_shards:
        files = files[:args.max_shards]
    totals = {"train": args.train_tokens, "validation": args.validation_tokens}
    settings = {"schema": SCHEMA, "source_manifest_sha256": sha(args.source_manifest),
                "script_sha256": sha(Path(__file__)), "model": args.model, "revision": args.model_revision,
                "seed": args.seed, "totals": totals, "block_length": args.block_length,
                "validation_threshold": int(args.validation_fraction * 2**64),
                "pilot_validation_jsonl_sha256": sha(args.pilot_data / "validation.jsonl"),
                "pilot_validation_tokens_sha256": sha(args.pilot_data / "validation_tokens.npy"),
                "batch_documents": args.batch_documents, "reserve_factor": args.candidate_reserve_factor,
                "minimum_document_tokens": args.minimum_document_tokens, "maximum_characters": args.maximum_characters,
                "shard_order": [file["name"] for file in files]}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    elif not args.output.exists():
        raise ValueError("Resume output does not exist")
    lock_handle = (args.output / ".prepare.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another preparation process owns this output") from error
    if args.resume:
        if not args.output.exists():
            raise ValueError("Resume output does not exist")
        saved_settings = json.loads((args.output / "settings.json").read_text())
        if settings != saved_settings:
            raise ValueError("Preparation settings/source/script changed; refusing resume")
        if (args.output / "manifest.json").exists():
            ready = json.loads((args.output / "manifest.json").read_text())
            if ready["status"] == "ready":
                print(json.dumps({"status": "already_ready", "output": str(args.output)}), flush=True)
                return
    else:
        atomic_json(args.output / "settings.json", settings)
    connection = initialize_database(args.output / "selected.sqlite")
    committed = connection.execute("SELECT payload FROM state WHERE id=1").fetchone()
    state = json.loads(committed[0]) if committed else {"signature": signature, "status": "building",
            "shards_done": 0, "tokens": {"train": 0, "validation": 0}, "documents": {"train": 0, "validation": 0},
            "metadata_bytes": {"train": 0, "validation": 0}, "shard_sampling": [],
            "rejected_global_text_or_url_duplicates": 0, "started_utc": stamp()}
    arrays, handles = {}, {}
    for split, budget in totals.items():
        path = args.output / f"{split}_tokens.npy"
        arrays[split] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.uint32, shape=(budget,))
        if arrays[split].shape != (budget,) or arrays[split].dtype != np.uint32:
            raise ValueError("Existing array shape/dtype changed")
        metadata = args.output / f"{split}.jsonl"
        handles[split] = metadata.open("r+b" if metadata.exists() else "w+b")
        if handles[split].seek(0, 2) < state["metadata_bytes"][split]:
            raise ValueError("Committed merged metadata truncated")
        handles[split].truncate(state["metadata_bytes"][split])
        handles[split].seek(state["metadata_bytes"][split])
    tokenizer = load_tokenizer(args.model, args.model_revision, 1)
    if committed is None:
        pilot_array = np.load(args.pilot_data / "validation_tokens.npy", mmap_mode="r")
        for record in read_jsonl(args.pilot_data / "validation.jsonl"):
            offset, used = record["token_offset"], record["used_tokens"]
            # Preserve the old used prefix, including its original document truncation.
            inserted = insert_record(connection, arrays, handles, "validation", record,
                                     pilot_array[offset:offset + used], used, state)
            if inserted != used:
                raise ValueError("Pilot validation has duplicates or inconsistent records")
        save_coordinator(connection, state, arrays, handles, args.output)
    jobs = []
    for index, file in enumerate(files):
        quotas = {"train": args.train_tokens // len(files) + int(index < args.train_tokens % len(files)),
                  "validation": (args.validation_tokens - pilot_tokens) // len(files) +
                  int(index < (args.validation_tokens - pilot_tokens) % len(files))}
        candidate_tokens = {split: math.ceil(quota * args.candidate_reserve_factor) for split, quota in quotas.items()}
        jobs.append({"name": file["name"], "source_path": str(Path(source["snapshot"]) / file["name"]),
                     "source_bytes": file["size"], "source_sha256": file["sha256"],
                     "cache": str(args.output / "candidate_cache" / f"shard_{index:03d}"),
                     "pilot_jsonl": str(args.pilot_data / "validation.jsonl"), "signature": signature,
                     "model": args.model, "revision": args.model_revision, "seed": args.seed,
                     "validation_threshold": settings["validation_threshold"], "candidate_tokens": candidate_tokens,
                     "quotas": quotas, "threads": args.tokenizer_threads, "batch_documents": args.batch_documents,
                     "minimum_document_tokens": args.minimum_document_tokens, "maximum_characters": args.maximum_characters})
    print(json.dumps({"status": "building", "output": str(args.output), "workers": args.workers,
                      "tokenizer_threads": args.tokenizer_threads, "total_shards": len(files),
                      "shards_done": state["shards_done"], "targets": totals}), flush=True)
    # Spawn avoids inheriting the coordinator tokenizer's initialized Rust thread pool.
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
    futures = {index: executor.submit(worker, job) for index, job in enumerate(jobs) if index >= state["shards_done"]}
    stopped = False
    try:
        for index in range(state["shards_done"], len(jobs)):
            job = jobs[index]
            while not futures[index].done():
                time.sleep(5)
                completed = sum(future.done() for future in futures.values())
                progress = {"phase": "tokenizing", "merged_shards": state["shards_done"],
                            "candidate_shards_complete": completed, "total_shards": len(files), "updated_utc": stamp()}
                atomic_json(args.output / "worker_progress.json", progress)
            result = futures[index].result()
            cache = Path(job["cache"])
            usage, counts = {"train": 0, "validation": 0}, {"train": 0, "validation": 0}
            for split in ("validation", "train"):
                metadata_cursor = 0
                while usage[split] < job["quotas"][split]:
                    raw_path = cache / f"{split}.tokens.raw"
                    candidates = np.memmap(raw_path, dtype=np.uint32, mode="r") if raw_path.stat().st_size else np.empty(0, dtype=np.uint32)
                    with (cache / f"{split}.jsonl").open("rb") as records:
                        records.seek(metadata_cursor)
                        for line in records:
                            record = json.loads(line)
                            remaining = job["quotas"][split] - usage[split]
                            start, length = record["candidate_token_offset"], record["document_tokens_with_eos"]
                            if start < 0 or start + length > len(candidates):
                                raise ValueError("Candidate tokens do not match metadata")
                            inserted = insert_record(connection, arrays, handles, split, record,
                                                     candidates[start:start + length], remaining, state)
                            usage[split] += inserted
                            counts[split] += int(inserted > 0)
                            metadata_cursor = records.tell()
                            if usage[split] >= job["quotas"][split]:
                                break
                    del candidates
                    if usage[split] >= job["quotas"][split]:
                        break
                    if result["status"] == "source_exhausted":
                        raise ValueError(f"Insufficient unique {split} tokens after exhausting {job['name']}: "
                                         f"{usage[split]} / {job['quotas'][split]}")
                    # Cross-shard deduplication can consume the reserve. Continue
                    # the same deterministic row cursor instead of restarting.
                    deficit = job["quotas"][split] - usage[split]
                    job["candidate_tokens"][split] = result["tokens"][split] + max(deficit * 2, job["quotas"][split] // 10)
                    print(json.dumps({"phase": "extending_candidate_cache", "shard": job["name"],
                                      "split": split, "deficit": deficit}), flush=True)
                    result = executor.submit(worker, job).result()
            state["shard_sampling"].append({"shard": job["name"], "tokens": usage, "documents": counts,
                                            "candidate_tokens": result["tokens"], "candidate_documents": result["documents"],
                                            "scanned_documents": result["scanned_documents"],
                                            "source_sha256": result["source_sha256"]})
            state["shards_done"] = index + 1
            save_coordinator(connection, state, arrays, handles, args.output)
            print(json.dumps({"shards_done": state["shards_done"], "total_shards": len(files),
                              "tokens": state["tokens"], "documents": state["documents"],
                              "worker_tokens_per_second": result.get("last_session_tokens_per_second")}), flush=True)
            if args.stop_after_shards and state["shards_done"] >= args.stop_after_shards:
                stopped = True
                for future in futures.values():
                    future.cancel()
                break
    finally:
        # Running shard tasks finish durable caches; interruptions never invalidate committed merge state.
        executor.shutdown(wait=True, cancel_futures=True)
        for handle in handles.values():
            handle.close()
    if stopped:
        print(json.dumps({"status": "stopped", "shards_done": state["shards_done"]}), flush=True)
        connection.close()
        return
    if state["tokens"] != totals:
        raise ValueError("Final token budget mismatch")
    state["status"] = "verifying"
    atomic_json(args.output / "progress.json", state)
    checks = verify(args.output, source, tokenizer, connection, totals, args.block_length, args.pilot_data, files)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    artifacts = {name: {"bytes": (args.output / name).stat().st_size, "sha256": sha(args.output / name)}
                 for name in ("train_tokens.npy", "validation_tokens.npy", "train.jsonl", "validation.jsonl", "selected.sqlite", "settings.json")}
    manifest = {"status": "ready", "purpose": "OpenWebMath-only full-parameter CPT training dataset",
                "source_repo": source["repo"], "source_revision": source["revision"],
                "source_manifest": str(args.source_manifest.resolve()), "source_manifest_sha256": sha(args.source_manifest),
                "tokenizer": args.model, "tokenizer_revision": args.model_revision, "seed": args.seed,
                "sampling": "equal train and new-validation token quotas across fixed-seed shuffled source shards; shuffled row groups and rows",
                "split": "fixed normalized-text SHA256 threshold independent of budgets; all pilot validation text hashes/URLs reserved first",
                "validation_fraction": args.validation_fraction, "validation_hash_threshold": settings["validation_threshold"],
                "packing": PACKING, "block_length": args.block_length, "dtype": "uint32", "splits": checks,
                "train_validation_text_hash_overlap": 0, "train_validation_url_overlap": 0,
                "exact_global_selected_text_and_url_deduplication": True,
                "preserved_pilot_validation_tokens": pilot_tokens,
                "preserved_pilot_validation_documents": pilot_manifest["splits"]["validation"]["documents"],
                "pilot_validation_jsonl_sha256": settings["pilot_validation_jsonl_sha256"],
                "pilot_validation_tokens_sha256": settings["pilot_validation_tokens_sha256"],
                "all_used_source_files_sha256_verified": True,
                "train_metadata_has_original_text": False,
                "train_metadata_source_resolution": "source_shard + row_group + row_in_group resolves original text in pinned source parquet",
                "rejected_global_text_or_url_duplicates": state["rejected_global_text_or_url_duplicates"],
                "artifacts": artifacts, "shard_sampling": state["shard_sampling"],
                "preparation_signature": signature, "script_sha256": settings["script_sha256"],
                "benchmark_decontamination": "not performed; held-out text loss only, no benchmark generalization claims",
                "license": source["license"], "source_url": source["source_url"],
                "started_utc": state["started_utc"], "completed_utc": stamp()}
    atomic_json(args.output / "manifest.json", manifest)
    state["status"] = "ready"
    atomic_json(args.output / "progress.json", state)
    print(json.dumps({"status": "ready", "splits": checks, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
