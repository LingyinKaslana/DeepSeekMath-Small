"""Retokenize a pinned OpenWebMath document subset for a DeepSeek Coder pilot.

Training follows the already selected Qwen documents in each source shard,
with equal NEW-token quotas across all shards. Validation keeps every original
held-out document, but restores its complete text before retokenization. Old
Qwen token arrays are never used. Restartable worker caches are committed per
shard; an interrupted merge is rebuilt from those immutable caches on resume.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
PACKING = "document text + EOS, concatenated within each split; no cross-document attention mask"
ORIGIN_KEYS = ("url", "text_sha256", "source_shard", "row_group", "row_in_group")
OLD_TOKEN_KEYS = ("document_tokens_with_eos", "used_tokens", "token_offset", "prefix_truncated")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def text_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def load_tokenizer(model, revision, threads):
    # Each spawned worker initializes its own Rust pool; no inherited tokenizer.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError("A locally cached fast tokenizer with an EOS token is required")
    tokenizer.model_max_length = 1 << 60
    return tokenizer


def index_reference_training(path, expected_documents):
    """Record contiguous shard byte ranges without retaining 1 GB of metadata."""
    entries, completed = [], set()
    previous, documents = None, 0
    with Path(path).open("rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            record = json.loads(line)
            shard = record["source_shard"]
            if shard != previous:
                if shard in completed:
                    raise ValueError("Reference training shard metadata is not contiguous")
                if entries:
                    entries[-1]["metadata_end"] = start
                    completed.add(previous)
                entries.append({"name": shard, "metadata_start": start,
                                "reference_documents": 0})
                previous = shard
            entries[-1]["reference_documents"] += 1
            documents += 1
        if entries:
            entries[-1]["metadata_end"] = handle.tell()
    if documents != expected_documents or not entries:
        raise ValueError("Reference training document count does not match its manifest")
    return entries


class SourceRows:
    """Bounded caches prevent one Parquet read per selected document."""
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot)
        self.readers, self.groups = OrderedDict(), OrderedDict()

    def document(self, record):
        shard = record["source_shard"]
        reader = self.readers.pop(shard, None)
        if reader is None:
            reader = pq.ParquetFile(self.snapshot / shard)
        self.readers[shard] = reader
        while len(self.readers) > 8:
            self.readers.popitem(last=False)
        key = shard, record["row_group"]
        rows = self.groups.pop(key, None)
        if rows is None:
            rows = reader.read_row_group(key[1], columns=["text", "url"]).to_pylist()
        self.groups[key] = rows
        while len(self.groups) > 4:
            self.groups.popitem(last=False)
        row = rows[record["row_in_group"]]
        if (not isinstance(row["text"], str) or text_hash(row["text"]) != record["text_sha256"]
                or row["url"] != record["url"]):
            raise ValueError("Pinned source text/URL differs from reference provenance")
        if "text" in record and row["text"] != record["text"]:
            raise ValueError("Original validation text differs from its pinned source row")
        return row


def new_record(original, length, used, offset, byte_offset=None):
    result = {key: original[key] for key in ORIGIN_KEYS}
    if "text" in original:
        result["text"] = original["text"]
    result.update(document_tokens_with_eos=length, used_tokens=used,
                  token_offset=offset, prefix_truncated=used < length,
                  reference_tokens={key: original[key] for key in OLD_TOKEN_KEYS})
    if byte_offset is not None:
        result["reference_metadata_byte_offset"] = byte_offset
    return result


def cached_result(directory, signature):
    path = directory / "complete.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result["signature"] != signature:
        raise ValueError("Worker cache signature differs from the requested preparation")
    for name, artifact in result["artifacts"].items():
        actual = directory / name
        if actual.stat().st_size != artifact["bytes"] or sha(actual) != artifact["sha256"]:
            raise ValueError("Committed worker cache was changed or truncated")
    return result


def commit_cache(directory, result):
    result["artifacts"] = {
        name: {"bytes": (directory / name).stat().st_size, "sha256": sha(directory / name)}
        for name in ("tokens.raw", "metadata.jsonl")}
    result["status"] = "complete"
    result["completed_utc"] = stamp()
    atomic_json(directory / "complete.json", result)
    return result


def train_worker(job):
    directory = Path(job["cache"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".worker.lock").open("a+") as lock:
        # A crashed coordinator can leave a worker finishing this same cache.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        source_path = Path(job["snapshot"]) / job["name"]
        if source_path.stat().st_size != job["source_bytes"] or sha(source_path) != job["source_sha256"]:
            raise ValueError(f"Pinned source shard changed: {source_path}")
        existing = cached_result(directory, job["signature"])
        if existing is not None:
            return existing
        started = time.perf_counter()
        tokenizer = load_tokenizer(job["model"], job["revision"], job["threads"])
        source_rows = SourceRows(job["snapshot"])
        tokens = documents = 0
        result = {"signature": job["signature"], "shard": job["name"],
                  "source_sha256": job["source_sha256"], "source_bytes": job["source_bytes"],
                  "reference_documents": job["reference_documents"], "quota": job["quota"],
                  "started_utc": stamp()}
        atomic_json(directory / "progress.json", {**result, "status": "building"})
        with Path(job["reference_train"]).open("rb") as reference, \
                (directory / "tokens.raw").open("wb") as raw, \
                (directory / "metadata.jsonl").open("w") as metadata:
            reference.seek(job["metadata_start"])
            while tokens < job["quota"]:
                records, texts = [], []
                for _ in range(job["batch_documents"]):
                    byte_offset = reference.tell()
                    if byte_offset >= job["metadata_end"]:
                        break
                    record = json.loads(reference.readline())
                    if record["source_shard"] != job["name"]:
                        raise ValueError("Reference byte range crossed its selected source shard")
                    records.append((record, byte_offset))
                    texts.append(source_rows.document(record)["text"])
                if not records:
                    raise ValueError(f"Insufficient reference-selected documents for {job['name']}")
                encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                                    return_token_type_ids=False)["input_ids"]
                for (record, byte_offset), ids in zip(records, encoded, strict=True):
                    if tokens >= job["quota"]:
                        break
                    ids.append(tokenizer.eos_token_id)
                    used = min(len(ids), job["quota"] - tokens)
                    raw.write(np.asarray(ids[:used], dtype=np.uint32).tobytes())
                    metadata.write(json.dumps(new_record(record, len(ids), used, tokens, byte_offset),
                                              ensure_ascii=False) + "\n")
                    tokens += used
                    documents += 1
                atomic_json(directory / "progress.json", {
                    **result, "status": "building", "tokens": tokens,
                    "documents": documents, "updated_utc": stamp()})
            raw.flush()
            metadata.flush()
            os.fsync(raw.fileno())
            os.fsync(metadata.fileno())
        result.update(tokens=tokens, documents=documents,
                      seconds=time.perf_counter() - started)
        return commit_cache(directory, result)


def build_validation(args, signature, source):
    directory = args.output / "worker_cache" / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".worker.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = cached_result(directory, signature)
        if existing is not None:
            return existing
        tokenizer = load_tokenizer(args.model, args.revision, args.tokenizer_threads)
        records = list(read_jsonl(args.reference_data / "validation.jsonl"))
        source_rows = SourceRows(source["snapshot"])
        tokens = 0
        with (directory / "tokens.raw").open("wb") as raw, \
                (directory / "metadata.jsonl").open("w") as metadata:
            for start in range(0, len(records), args.batch_documents):
                batch = records[start:start + args.batch_documents]
                texts = [source_rows.document(record)["text"] for record in batch]
                encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                                    return_token_type_ids=False)["input_ids"]
                for record, ids in zip(batch, encoded, strict=True):
                    ids.append(tokenizer.eos_token_id)
                    raw.write(np.asarray(ids, dtype=np.uint32).tobytes())
                    metadata.write(json.dumps(new_record(record, len(ids), len(ids), tokens),
                                              ensure_ascii=False) + "\n")
                    tokens += len(ids)
            raw.flush()
            metadata.flush()
            os.fsync(raw.fileno())
            os.fsync(metadata.fileno())
        return commit_cache(directory, {"signature": signature, "tokens": tokens,
                                       "documents": len(records), "started_utc": stamp()})


def copy_cache_into_array(directory, array, offset):
    raw = np.memmap(directory / "tokens.raw", dtype=np.uint32, mode="r")
    for start in range(0, len(raw), 8_388_608):
        chunk = raw[start:start + 8_388_608]
        array[offset + start:offset + start + len(chunk)] = chunk
    return len(raw)


def verify(output, reference, source, tokenizer, expected, block_length):
    checks, hashes, urls = {}, {}, {}
    validation_reference = list(read_jsonl(reference / "validation.jsonl"))
    source_rows = SourceRows(source["snapshot"])
    with (reference / "train.jsonl").open("rb") as original_train:
        for split, expected_tokens in expected.items():
            array = np.load(output / f"{split}_tokens.npy", mmap_mode="r")
            if array.dtype != np.uint32 or array.shape != (expected_tokens,):
                raise ValueError("Prepared token layout does not match its final counters")
            for start in range(0, len(array), 8_388_608):
                if int(array[start:start + 8_388_608].max()) >= len(tokenizer):
                    raise ValueError("New token array contains an out-of-vocabulary ID")
            hashes[split], urls[split] = set(), set()
            offset = documents = truncated = 0
            samples = {}
            for record in read_jsonl(output / f"{split}.jsonl"):
                if split == "train":
                    original_train.seek(record["reference_metadata_byte_offset"])
                    original = json.loads(original_train.readline())
                else:
                    original = validation_reference[documents]
                if (any(record[key] != original[key] for key in ORIGIN_KEYS)
                        or record["reference_tokens"] != {key: original[key] for key in OLD_TOKEN_KEYS}):
                    raise ValueError("A selected document moved split or changed reference provenance")
                used, length = record["used_tokens"], record["document_tokens_with_eos"]
                if (record["token_offset"] != offset or not 0 < used <= length
                        or record["prefix_truncated"] != (used < length)):
                    raise ValueError("New token metadata offsets/lengths do not match")
                digest, url = record["text_sha256"], record["url"]
                if digest in hashes[split] or (url and url in urls[split]):
                    raise ValueError("Selected split contains repeated text or URL")
                hashes[split].add(digest)
                if url:
                    urls[split].add(url)
                bucket = samples.setdefault(record["source_shard"], {
                    "first": record, "middle": record, "last": record, "count": 0})
                bucket["count"] += 1
                bucket["last"] = record
                if int(digest[:16], 16) % bucket["count"] == 0:
                    bucket["middle"] = record
                if split == "validation":
                    if record["text"] != original["text"] or used != length:
                        raise ValueError("Validation must retain complete, unchanged reference texts")
                    ids = tokenizer.encode(record["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
                    if len(ids) != length:
                        raise ValueError("Validation tokenizer roundtrip length differs")
                    np.testing.assert_array_equal(array[offset:offset + used], np.asarray(ids, dtype=np.uint32))
                offset += used
                documents += 1
                truncated += int(record["prefix_truncated"])
            if offset != expected_tokens or (split == "validation" and documents != len(validation_reference)):
                raise ValueError("Prepared document counts or token totals do not match")
            roundtrips = 0
            for bucket in samples.values():
                records = {record["text_sha256"]: record
                           for record in (bucket["first"], bucket["middle"], bucket["last"])}
                for record in records.values():
                    text = source_rows.document(record)["text"]
                    ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                    start, used = record["token_offset"], record["used_tokens"]
                    if len(ids) != record["document_tokens_with_eos"]:
                        raise ValueError("Source/tokenizer sampled roundtrip length differs")
                    np.testing.assert_array_equal(array[start:start + used], np.asarray(ids[:used], dtype=np.uint32))
                    roundtrips += 1
            checks[split] = {
                "tokens": expected_tokens, "documents": documents,
                "full_blocks": expected_tokens // block_length,
                "tokens_in_full_blocks": expected_tokens // block_length * block_length,
                "remainder_tokens": expected_tokens % block_length,
                "prefix_truncated_documents": truncated, "source_shards": len(samples),
                "all_metadata_records_audited": True, "all_selected_documents_match_reference_split": True,
                "all_token_ids_range_checked": True, "source_token_roundtrip_samples": roundtrips,
                "all_document_token_roundtrip": split == "validation"}
    if hashes["train"] & hashes["validation"] or urls["train"] & urls["validation"]:
        raise ValueError("Train and held-out validation overlap by normalized text or URL")
    return checks


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-manifest", type=Path,
                        default=ROOT / "data/raw/openwebmath_20261008/manifest.json")
    result.add_argument("--reference-data", type=Path,
                        default=ROOT / "data/processed/openwebmath_5b_s42")
    result.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-base")
    result.add_argument("--revision", default="c919139c3a9b4070729c8b2cca4847ab29ca8d94")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--train-tokens", type=int, default=512_000_000)
    result.add_argument("--block-length", type=int, default=2048)
    result.add_argument("--workers", type=int, default=12)
    result.add_argument("--tokenizer-threads", type=int, default=2)
    result.add_argument("--batch-documents", type=int, default=256)
    result.add_argument("--resume", action="store_true")
    return result


def main():
    args = parser().parse_args()
    started = stamp()
    if min(args.train_tokens, args.block_length, args.workers, args.tokenizer_threads, args.batch_documents) < 1:
        raise ValueError("Token budgets and resource settings must be positive")
    if args.workers * args.tokenizer_threads > (os.cpu_count() or 1):
        raise ValueError("Worker tokenizer thread total exceeds available CPUs")
    source = json.loads(args.source_manifest.read_text())
    reference = json.loads((args.reference_data / "manifest.json").read_text())
    if source["status"] != "verified" or reference["status"] != "ready":
        raise ValueError("Pinned raw sources and reference selection must already be verified/ready")
    if (reference["source_repo"], reference["source_revision"]) != (source["repo"], source["revision"]):
        raise ValueError("Reference selection and raw source revision differ")
    source_hash = sha(args.source_manifest)
    if source_hash != reference["source_manifest_sha256"]:
        raise ValueError("Pinned source manifest changed since reference selection")
    reference_hashes = {}
    for name in ("train.jsonl", "validation.jsonl"):
        reference_hashes[name] = sha(args.reference_data / name)
        if reference_hashes[name] != reference["artifacts"][name]["sha256"]:
            raise ValueError("Reference document selection metadata changed")
    shards = index_reference_training(args.reference_data / "train.jsonl",
                                      reference["splits"]["train"]["documents"])
    files = {file["name"]: file for file in source["files"]}
    if set(entry["name"] for entry in shards) != set(files) or len(shards) != len(files):
        raise ValueError("Reference training selection must cover every pinned source shard")
    if args.train_tokens < len(shards):
        raise ValueError("Train budget cannot provide a positive quota to every source shard")
    settings = {
        "schema_version": 1, "model": args.model, "revision": args.revision,
        "source_manifest_sha256": source_hash,
        "reference_manifest_sha256": sha(args.reference_data / "manifest.json"),
        "reference_metadata_sha256": reference_hashes,
        "script_sha256": sha(Path(__file__)), "train_tokens": args.train_tokens,
        "block_length": args.block_length, "batch_documents": args.batch_documents,
        "shard_order": [entry["name"] for entry in shards],
        "validation_policy": "all reference validation documents, complete original text; old prefixes restored"}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if args.resume:
        if not args.output.is_dir():
            raise ValueError("Resume output does not exist")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / ".prepare.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another preparation coordinator owns this output") from error
        if args.resume:
            if json.loads((args.output / "settings.json").read_text()) != settings:
                raise ValueError("Resume source, script, reference selection or token settings changed")
            if (args.output / "manifest.json").exists():
                manifest = json.loads((args.output / "manifest.json").read_text())
                if manifest.get("status") == "ready":
                    for name, artifact in manifest["artifacts"].items():
                        if sha(args.output / name) != artifact["sha256"]:
                            raise ValueError("Previously ready preparation artifact changed")
                    print(json.dumps({"status": "already_ready", "output": str(args.output)}), flush=True)
                    return
        else:
            atomic_json(args.output / "settings.json", settings)
        # Fail on missing tokenizer before spawning jobs or creating large arrays.
        tokenizer = load_tokenizer(args.model, args.revision, 1)
        del tokenizer
        jobs = []
        for index, entry in enumerate(shards):
            file = files[entry["name"]]
            jobs.append({**entry, "signature": signature, "snapshot": source["snapshot"],
                         "source_bytes": file["size"], "source_sha256": file["sha256"],
                         "reference_train": str((args.reference_data / "train.jsonl").resolve()),
                         "cache": str((args.output / "worker_cache" / f"shard_{index:03d}").resolve()),
                         "quota": args.train_tokens // len(shards) + int(index < args.train_tokens % len(shards)),
                         "model": args.model, "revision": args.revision,
                         "threads": args.tokenizer_threads, "batch_documents": args.batch_documents})
        progress = {"status": "building", "phase": "train_tokenization", "started_utc": started,
                    "train_token_target": args.train_tokens, "total_shards": len(jobs),
                    "shards_merged": 0, "tokens_merged": 0}
        atomic_json(args.output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        train_array = np.lib.format.open_memmap(args.output / "train_tokens.npy", mode="w+",
                                               dtype=np.uint32, shape=(args.train_tokens,))
        results, offset = [], 0
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor, \
                (args.output / "train.jsonl").open("w") as metadata:
            futures = [executor.submit(train_worker, job) for job in jobs]
            for index, (job, future) in enumerate(zip(jobs, futures, strict=True)):
                result = future.result()
                directory = Path(job["cache"])
                copied = copy_cache_into_array(directory, train_array, offset)
                if copied != job["quota"] or copied != result["tokens"]:
                    raise ValueError("Worker token quota differs from its cache")
                for record in read_jsonl(directory / "metadata.jsonl"):
                    record["token_offset"] += offset
                    metadata.write(json.dumps(record, ensure_ascii=False) + "\n")
                offset += copied
                results.append(result)
                progress.update(shards_merged=index + 1, tokens_merged=offset, updated_utc=stamp())
                atomic_json(args.output / "progress.json", progress)
                print(json.dumps({"shards_merged": index + 1, "train_tokens": offset,
                                  "worker_documents": result["documents"]}), flush=True)
            metadata.flush()
            os.fsync(metadata.fileno())
        if offset != args.train_tokens:
            raise ValueError("Final training token count differs from the requested budget")
        train_array.flush()
        del train_array
        progress.update(phase="validation_tokenization", updated_utc=stamp())
        atomic_json(args.output / "progress.json", progress)
        validation = build_validation(args, signature, source)
        validation_array = np.lib.format.open_memmap(args.output / "validation_tokens.npy", mode="w+",
                                                    dtype=np.uint32, shape=(validation["tokens"],))
        cache = args.output / "worker_cache" / "validation"
        if copy_cache_into_array(cache, validation_array, 0) != validation["tokens"]:
            raise ValueError("Validation cache byte length differs from its token count")
        validation_array.flush()
        del validation_array
        with (args.output / "validation.jsonl").open("w") as metadata:
            for record in read_jsonl(cache / "metadata.jsonl"):
                metadata.write(json.dumps(record, ensure_ascii=False) + "\n")
            metadata.flush()
            os.fsync(metadata.fileno())
        for split in ("train", "validation"):
            with (args.output / f"{split}_tokens.npy").open("rb") as handle:
                os.fsync(handle.fileno())
        progress.update(phase="verification", validation_tokens=validation["tokens"], updated_utc=stamp())
        atomic_json(args.output / "progress.json", progress)
        tokenizer = load_tokenizer(args.model, args.revision, 1)
        checks = verify(args.output, args.reference_data, source, tokenizer,
                        {"train": args.train_tokens, "validation": validation["tokens"]}, args.block_length)
        if checks["train"]["source_shards"] != len(shards):
            raise ValueError("Prepared training subset lost a source shard")
        names = ("train_tokens.npy", "validation_tokens.npy", "train.jsonl", "validation.jsonl", "settings.json")
        artifacts = {name: {"bytes": (args.output / name).stat().st_size,
                            "sha256": sha(args.output / name)} for name in names}
        manifest = {
            "status": "ready", "purpose": "Fresh DeepSeek-Coder mathematical CPT pilot",
            "source_repo": source["repo"], "source_revision": source["revision"],
            "source_manifest": str(args.source_manifest.resolve()), "source_manifest_sha256": source_hash,
            "reference_data": str(args.reference_data.resolve()),
            "reference_manifest_sha256": settings["reference_manifest_sha256"],
            "reference_metadata_sha256": reference_hashes,
            "tokenizer": args.model, "tokenizer_revision": args.revision,
            "tokenizer_eos_token_id": tokenizer.eos_token_id, "tokenizer_vocabulary_size": len(tokenizer),
            "seed": reference.get("seed"), "packing": PACKING,
            "sampling": "equal NEW-token quota per source shard; prefix of existing reference-selected document order",
            "split": "existing reference train/validation document membership; no fresh split or document resampling",
            "block_length": args.block_length, "dtype": "uint32", "splits": checks,
            "train_validation_text_hash_overlap": 0, "train_validation_url_overlap": 0,
            "exact_selected_text_and_url_deduplication": True,
            "all_used_source_files_sha256_verified": True,
            "train_metadata_has_original_text": False,
            "train_metadata_source_resolution": "source_shard + row_group + row_in_group resolves original text in pinned source parquet",
            "validation_policy": settings["validation_policy"],
            "reference_validation_documents": reference["splits"]["validation"]["documents"],
            "reference_validation_prefix_truncated_documents": reference["splits"]["validation"]["prefix_truncated_documents"],
            "validation_loss_comparison": "New tokenizer and restored full held-out text; absolute loss is not directly comparable to the Qwen run",
            "shard_sampling": [{key: result[key] for key in (
                "shard", "tokens", "documents", "reference_documents", "quota", "source_sha256")}
                for result in results],
            "old_token_arrays_reused": False, "preparation_signature": signature,
            "script_sha256": settings["script_sha256"], "artifacts": artifacts,
            "benchmark_decontamination": reference.get("benchmark_decontamination", "not performed"),
            "license": source.get("license"), "source_url": source.get("source_url"),
            "started_utc": started, "completed_utc": stamp()}
        atomic_json(args.output / "manifest.json", manifest)
        progress.update(status="ready", phase="complete", splits=checks, updated_utc=stamp())
        atomic_json(args.output / "progress.json", progress)
        print(json.dumps({"status": "ready", "splits": checks}), flush=True)


if __name__ == "__main__":
    main()
