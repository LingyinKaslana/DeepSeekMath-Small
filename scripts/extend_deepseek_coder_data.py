"""Extend pinned DeepSeek OpenWebMath arrays without changing their logical prefix.

Old DeepSeek tokens and metadata are copied verbatim; new documents start after
its last selected reference document in each shard. Incomplete old document
tails are intentionally skipped. No Qwen token array is used. Worker caches,
source hashes, held-out membership and complete prefix/probe checks are audited.
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
import shutil

import numpy as np
import pyarrow.parquet as pq


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


def copy_cache_into_array(directory, array, offset):
    raw = np.memmap(directory / "tokens.raw", dtype=np.uint32, mode="r")
    for start in range(0, len(raw), 8_388_608):
        chunk = raw[start:start + 8_388_608]
        array[offset + start:offset + start + len(chunk)] = chunk
    return len(raw)


def verify(output, reference, source, tokenizer, expected, block_length, prefix_tokens=0):
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
                sample_key = (record["source_shard"], "prefix" if split == "train" and offset < prefix_tokens else "extension")
                bucket = samples.setdefault(sample_key, {
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
                if not record["prefix_truncated"] and int(array[offset + used - 1]) != tokenizer.eos_token_id:
                    raise ValueError("Complete document is missing the tokenizer EOS")
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
                "prefix_truncated_documents": truncated, "source_shards": len({key[0] for key in samples}),
                "roundtrip_sample_regions": {region: sum(1 for key in samples if key[1] == region) for region in ("prefix", "extension")},
                "all_complete_documents_end_with_eos": True,
                "all_metadata_records_audited": True, "all_selected_documents_match_reference_split": True,
                "all_token_ids_range_checked": True, "source_token_roundtrip_samples": roundtrips,
                "all_document_token_roundtrip": split == "validation"}
    if hashes["train"] & hashes["validation"] or urls["train"] & urls["validation"]:
        raise ValueError("Train and held-out validation overlap by normalized text or URL")
    return checks


def checked_artifacts(directory, manifest, names=None):
    """Return immutable hashes, refusing missing or modified input artifacts."""
    result = {}
    for name in names or manifest["artifacts"]:
        spec = manifest["artifacts"][name]
        path = directory / name
        digest = sha(path)
        if path.stat().st_size != spec["bytes"] or digest != spec["sha256"]:
            raise ValueError(f"Pinned preparation artifact changed: {path}")
        result[name] = digest
    return result


def copy_token_prefix(source, destination):
    digest = hashlib.sha256()
    for start in range(0, len(source), 8_388_608):
        chunk = source[start:start + 8_388_608]
        destination[start:start + len(chunk)] = chunk
        digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
    return digest.hexdigest()


def prefix_proof(old, new, expected_digest):
    digest = hashlib.sha256()
    for start in range(0, len(old), 8_388_608):
        original = old[start:start + 8_388_608]
        actual = new[start:start + len(original)]
        np.testing.assert_array_equal(original, actual)
        digest.update(memoryview(np.ascontiguousarray(actual)).cast("B"))
    if digest.hexdigest() != expected_digest:
        raise ValueError("Preserved DeepSeek logical token prefix checksum changed")
    return digest.hexdigest()


def file_prefix_sha(path, count):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        remaining = count
        while remaining:
            chunk = handle.read(min(remaining, 8_388_608))
            if not chunk:
                raise ValueError("Metadata prefix is shorter than its input")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def probe_proof(old, new, block_length, probe=None):
    """Keep saved probe indices, rather than resampling against a larger corpus."""
    if probe is None:
        count = min(64, len(old) // block_length)
        indices = np.sort(np.random.default_rng(20261009).choice(
            len(old) // block_length, size=count, replace=False)).astype(np.int64)
        expected_index_sha = expected_token_sha = None
    else:
        indices = np.asarray(probe["indices"], dtype=np.int64)
        if len(indices) != probe["blocks"]:
            raise ValueError("Saved probe index count differs")
        expected_index_sha = probe["indices_sha256"]
        expected_token_sha = probe["tokens_sha256"]
    if (np.any(indices < 0) or np.any(indices >= len(old) // block_length)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("Probe must select distinct complete old-prefix blocks")
    old_sha, new_sha = hashlib.sha256(), hashlib.sha256()
    for index in indices:
        start = int(index) * block_length
        before, after = old[start:start + block_length], new[start:start + block_length]
        np.testing.assert_array_equal(before, after)
        old_sha.update(memoryview(np.ascontiguousarray(before)).cast("B"))
        new_sha.update(memoryview(np.ascontiguousarray(after)).cast("B"))
    index_sha = hashlib.sha256(memoryview(indices).cast("B")).hexdigest()
    if (old_sha.hexdigest() != new_sha.hexdigest()
            or (expected_index_sha is not None and expected_index_sha != index_sha)
            or (expected_token_sha is not None and expected_token_sha != old_sha.hexdigest())):
        raise ValueError("Old fixed training probe checksum changed")
    return {"blocks": len(indices), "indices": indices.tolist(),
            "indices_sha256": index_sha, "old_tokens_sha256": old_sha.hexdigest(),
            "new_tokens_sha256": new_sha.hexdigest(), "preserved": True,
            "saved_training_probe": probe is not None}


def pilot_boundaries(directory, manifest, shards, reference_path):
    """Ensure old per-shard selections are exact contiguous reference prefixes."""
    entries = {entry["name"]: entry for entry in shards}
    counts, last, next_offsets = {}, {}, {}
    offset = documents = 0
    previous, closed = None, set()
    with reference_path.open("rb") as reference:
        for record in read_jsonl(directory / "train.jsonl"):
            name = record["source_shard"]
            if name not in entries:
                raise ValueError("Pilot selected an unknown reference source shard")
            if name != previous:
                if name in closed:
                    raise ValueError("Pilot source shard metadata is not contiguous")
                if previous is not None:
                    closed.add(previous)
                previous = name
                next_offsets[name] = entries[name]["metadata_start"]
            position = record["reference_metadata_byte_offset"]
            if position != next_offsets[name]:
                raise ValueError("Pilot documents are not the exact reference shard prefix")
            reference.seek(position)
            original = json.loads(reference.readline())
            if (any(record[key] != original[key] for key in ORIGIN_KEYS)
                    or record["reference_tokens"] != {key: original[key] for key in OLD_TOKEN_KEYS}
                    or record["token_offset"] != offset):
                raise ValueError("Pilot metadata changed reference provenance or token offsets")
            used, length = record["used_tokens"], record["document_tokens_with_eos"]
            if not 0 < used <= length or record["prefix_truncated"] != (used < length):
                raise ValueError("Pilot document used/full token lengths differ")
            next_offsets[name] = reference.tell()
            last[name] = record
            counts[name] = counts.get(name, 0) + 1
            offset += used
            documents += 1
    if (list(counts) != [entry["name"] for entry in shards]
            or offset != manifest["splits"]["train"]["tokens"]
            or documents != manifest["splits"]["train"]["documents"]):
        raise ValueError("Pilot shard coverage/counters differ from its manifest")
    result = {}
    for name, record in last.items():
        result[name] = {"metadata_start": next_offsets[name],
                        "old_documents": counts[name],
                        "skipped_partial_tail_tokens": record["document_tokens_with_eos"] - record["used_tokens"],
                        "old_last_reference_metadata_byte_offset": record["reference_metadata_byte_offset"],
                        "old_last_text_sha256": record["text_sha256"]}
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--reference-data", type=Path, required=True)
    result.add_argument("--pilot-data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-base")
    result.add_argument("--revision", default="c919139c3a9b4070729c8b2cca4847ab29ca8d94")
    result.add_argument("--new-total-tokens", type=int, default=5_000_000_000)
    result.add_argument("--block-length", type=int, default=2048)
    result.add_argument("--workers", type=int, default=12)
    result.add_argument("--tokenizer-threads", type=int, default=2)
    result.add_argument("--batch-documents", type=int, default=256)
    result.add_argument("--probe-json", type=Path)
    result.add_argument("--resume", action="store_true")
    return result


def main():
    args = parser().parse_args()
    started = stamp()
    for name in ("source_manifest", "reference_data", "pilot_data", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if min(args.new_total_tokens, args.block_length, args.workers,
           args.tokenizer_threads, args.batch_documents) < 1:
        raise ValueError("Token budgets and resource settings must be positive")
    if args.workers * args.tokenizer_threads > (os.cpu_count() or 1):
        raise ValueError("Tokenizer worker threads exceed available CPUs")
    if any(args.output.is_relative_to(parent) for parent in (args.pilot_data, args.reference_data)):
        raise ValueError("Output must not mutate any pinned input preparation")
    source = json.loads(args.source_manifest.read_text())
    reference = json.loads((args.reference_data / "manifest.json").read_text())
    pilot = json.loads((args.pilot_data / "manifest.json").read_text())
    if source["status"] != "verified" or any(value["status"] != "ready" for value in (reference, pilot)):
        raise ValueError("Raw source, reference selection and pilot must be verified/ready")
    source_hash = sha(args.source_manifest)
    reference_manifest_hash = sha(args.reference_data / "manifest.json")
    pilot_manifest_hash = sha(args.pilot_data / "manifest.json")
    for value in (reference, pilot):
        if ((value["source_repo"], value["source_revision"]) != (source["repo"], source["revision"])
                or value["source_manifest_sha256"] != source_hash):
            raise ValueError("Pinned source revision or source manifest differs")
    if (pilot["reference_manifest_sha256"] != reference_manifest_hash
            or (pilot["tokenizer"], pilot["tokenizer_revision"]) != (args.model, args.revision)
            or pilot["block_length"] != args.block_length
            or pilot["packing"] != PACKING or pilot["dtype"] != "uint32"):
        raise ValueError("Pilot tokenizer/reference/packing/block layout differs")
    reference_hashes = checked_artifacts(args.reference_data, reference, ("train.jsonl", "validation.jsonl"))
    if pilot["reference_metadata_sha256"] != reference_hashes:
        raise ValueError("Pilot reference document selection differs")
    pilot_hashes = checked_artifacts(args.pilot_data, pilot)
    old_tokens = pilot["splits"]["train"]["tokens"]
    append_tokens = args.new_total_tokens - old_tokens
    shards = index_reference_training(args.reference_data / "train.jsonl", reference["splits"]["train"]["documents"])
    files = {item["name"]: item for item in source["files"]}
    if set(entry["name"] for entry in shards) != set(files) or append_tokens < len(shards):
        raise ValueError("Extension must append positive quotas from every pinned source shard")
    boundaries = pilot_boundaries(args.pilot_data, pilot, shards, args.reference_data / "train.jsonl")
    probe = None if args.probe_json is None else json.loads(args.probe_json.read_text())
    settings = {"schema_version": 1, "model": args.model, "revision": args.revision,
                "source_manifest_sha256": source_hash,
                "reference_manifest_sha256": reference_manifest_hash,
                "reference_metadata_sha256": reference_hashes,
                "pilot_manifest_sha256": pilot_manifest_hash, "pilot_artifact_sha256": pilot_hashes,
                "script_sha256": sha(Path(__file__)), "old_train_tokens": old_tokens,
                "append_train_tokens": append_tokens, "new_total_tokens": args.new_total_tokens,
                "block_length": args.block_length, "batch_documents": args.batch_documents,
                "probe_json_sha256": None if args.probe_json is None else sha(args.probe_json),
                "shard_order": [entry["name"] for entry in shards], "append_boundaries": boundaries,
                "validation_policy": "byte-identical complete pilot held-out documents and DeepSeek tokens"}
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
                raise ValueError("Resume source, script, pilot or settings changed")
            if (args.output / "manifest.json").exists():
                manifest = json.loads((args.output / "manifest.json").read_text())
                if manifest.get("status") == "ready":
                    checked_artifacts(args.output, manifest)
                    print(json.dumps({"status": "already_ready", "output": str(args.output)}), flush=True)
                    return
        else:
            atomic_json(args.output / "settings.json", settings)
        tokenizer = load_tokenizer(args.model, args.revision, 1)
        if (tokenizer.eos_token_id != pilot["tokenizer_eos_token_id"]
                or len(tokenizer) != pilot["tokenizer_vocabulary_size"]):
            raise ValueError("Pinned tokenizer EOS or vocabulary size changed")
        del tokenizer
        old = np.load(args.pilot_data / "train_tokens.npy", mmap_mode="r", allow_pickle=False)
        if old.dtype != np.uint32 or old.shape != (old_tokens,):
            raise ValueError("Pilot train token shape/dtype differs")
        progress = {"status": "building", "phase": "extension_tokenization", "started_utc": started,
                    "train_token_target": args.new_total_tokens, "preserved_tokens": old_tokens,
                    "append_token_target": append_tokens, "total_shards": len(shards),
                    "shards_merged": 0, "tokens_merged": old_tokens}
        atomic_json(args.output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        train = np.lib.format.open_memmap(args.output / "train_tokens.npy", mode="w+",
                                         dtype=np.uint32, shape=(args.new_total_tokens,))
        jobs = []
        for index, entry in enumerate(shards):
            file = files[entry["name"]]
            start = boundaries[entry["name"]]["metadata_start"]
            if start >= entry["metadata_end"]:
                raise ValueError(f"No unused reference documents remain for {entry['name']}")
            jobs.append({**entry, "metadata_start": start, "signature": signature,
                         "snapshot": source["snapshot"], "source_bytes": file["size"],
                         "source_sha256": file["sha256"],
                         "reference_train": str(args.reference_data / "train.jsonl"),
                         "cache": str(args.output / "worker_cache" / f"shard_{index:03d}"),
                         "quota": append_tokens // len(shards) + int(index < append_tokens % len(shards)),
                         "model": args.model, "revision": args.revision,
                         "threads": args.tokenizer_threads, "batch_documents": args.batch_documents})
        results, offset = [], old_tokens
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor, \
                (args.output / "train.jsonl").open("wb") as metadata:
            futures = [executor.submit(train_worker, job) for job in jobs]
            prefix_digest = copy_token_prefix(old, train)
            with (args.pilot_data / "train.jsonl").open("rb") as original:
                shutil.copyfileobj(original, metadata, 8_388_608)
            for index, (job, future) in enumerate(zip(jobs, futures, strict=True)):
                result = future.result()
                directory = Path(job["cache"])
                copied = copy_cache_into_array(directory, train, offset)
                if copied != job["quota"] or copied != result["tokens"]:
                    raise ValueError("Worker extension quota differs from its verified cache")
                for record in read_jsonl(directory / "metadata.jsonl"):
                    record["token_offset"] += offset
                    metadata.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
                offset += copied
                results.append(result)
                progress.update(shards_merged=index + 1, tokens_merged=offset, updated_utc=stamp())
                atomic_json(args.output / "progress.json", progress)
                print(json.dumps({"shards_merged": index + 1, "train_tokens": offset,
                                  "new_worker_documents": result["documents"]}), flush=True)
            metadata.flush()
            os.fsync(metadata.fileno())
        if offset != args.new_total_tokens:
            raise ValueError("Final expanded token count differs from its requested budget")
        train.flush()
        del train
        for name in ("validation_tokens.npy", "validation.jsonl"):
            shutil.copyfile(args.pilot_data / name, args.output / name)
            with (args.output / name).open("rb") as handle:
                os.fsync(handle.fileno())
            if sha(args.output / name) != pilot_hashes[name]:
                raise ValueError("Copied held-out validation artifact changed")
        progress.update(phase="verification", updated_utc=stamp())
        atomic_json(args.output / "progress.json", progress)
        tokenizer = load_tokenizer(args.model, args.revision, 1)
        new = np.load(args.output / "train_tokens.npy", mmap_mode="r", allow_pickle=False)
        new_prefix_digest = prefix_proof(old, new, prefix_digest)
        old_metadata_bytes = pilot["artifacts"]["train.jsonl"]["bytes"]
        metadata_prefix_digest = file_prefix_sha(args.output / "train.jsonl", old_metadata_bytes)
        if metadata_prefix_digest != pilot_hashes["train.jsonl"]:
            raise ValueError("Pilot training metadata byte prefix changed")
        probe_evidence = probe_proof(old, new, args.block_length, probe)
        checks = verify(args.output, args.reference_data, source, tokenizer,
                        {"train": args.new_total_tokens,
                         "validation": pilot["splits"]["validation"]["tokens"]},
                        args.block_length, prefix_tokens=old_tokens)
        if (checks["train"]["source_shards"] != len(shards)
                or checks["train"]["roundtrip_sample_regions"]["extension"] != len(shards)
                or checks["validation"]["documents"] != pilot["splits"]["validation"]["documents"]):
            raise ValueError("Expanded source coverage or validation membership changed")
        # Recheck immutable inputs after workers/verification before committing ready.
        if (sha(args.pilot_data / "manifest.json") != pilot_manifest_hash
                or sha(args.reference_data / "manifest.json") != reference_manifest_hash
                or sha(args.source_manifest) != source_hash):
            raise ValueError("Input manifest changed during preparation")
        checked_artifacts(args.pilot_data, pilot)
        checked_artifacts(args.reference_data, reference, ("train.jsonl", "validation.jsonl"))
        extension = {"policy": "preserve exact DeepSeek logical token/metadata prefix; append only previously unselected reference documents",
                     "pilot_data": str(args.pilot_data), "pilot_manifest_sha256": pilot_manifest_hash,
                     "pilot_artifact_sha256": pilot_hashes, "old_train_tokens": old_tokens,
                     "append_train_tokens": append_tokens, "new_total_tokens": args.new_total_tokens,
                     "old_logical_prefix_sha256": prefix_digest, "new_logical_prefix_sha256": new_prefix_digest,
                     "logical_prefix_sha256": prefix_digest, "prefix_all_tokens_equal": True,
                     "old_prefix_full_blocks": old_tokens // args.block_length,
                     "old_prefix_block_aligned": old_tokens % args.block_length == 0,
                     "train_metadata_prefix_bytes": old_metadata_bytes,
                     "train_metadata_prefix_sha256": metadata_prefix_digest,
                     "validation_artifacts_byte_identical": True,
                     "validation_verification": checks["validation"],
                     "preserved_train_probe": probe_evidence,
                     "old_partial_document_tails_policy": "skip; next document begins each appended shard, avoiding duplicate selected text/URL",
                     "skipped_old_partial_tail_tokens": sum(item["skipped_partial_tail_tokens"] for item in boundaries.values()),
                     "appended_documents": sum(item["documents"] for item in results),
                     "all_appended_shard_quotas_actually_met": True,
                     "append_boundaries": boundaries}
        atomic_json(args.output / "prefix_verification.json", extension)
        names = ("train_tokens.npy", "validation_tokens.npy", "train.jsonl", "validation.jsonl",
                 "settings.json", "prefix_verification.json")
        artifacts = {name: {"bytes": (args.output / name).stat().st_size,
                            "sha256": sha(args.output / name)} for name in names}
        manifest = {"status": "ready", "purpose": "Expanded DeepSeek-Coder mathematical CPT corpus",
                    "source_repo": source["repo"], "source_revision": source["revision"],
                    "source_manifest": str(args.source_manifest), "source_manifest_sha256": source_hash,
                    "reference_data": str(args.reference_data), "reference_manifest_sha256": reference_manifest_hash,
                    "reference_metadata_sha256": reference_hashes,
                    "tokenizer": args.model, "tokenizer_revision": args.revision,
                    "tokenizer_eos_token_id": tokenizer.eos_token_id,
                    "tokenizer_vocabulary_size": len(tokenizer), "seed": reference.get("seed"),
                    "packing": PACKING, "dtype": "uint32", "block_length": args.block_length,
                    "sampling": "exact old DeepSeek token prefix, followed by equal NEW-token quotas per source shard from unused reference-selected documents",
                    "split": "unchanged reference train/validation document membership; no fresh split or document resampling",
                    # Preserve held-out split metadata as well as its files;
                    # expanded audit descriptions belong to data_extension.
                    "splits": {"train": checks["train"], "validation": pilot["splits"]["validation"]},
                    "data_extension": extension,
                    "train_validation_text_hash_overlap": 0, "train_validation_url_overlap": 0,
                    "exact_selected_text_and_url_deduplication": True,
                    "all_used_source_files_sha256_verified": True,
                    "train_metadata_has_original_text": False,
                    "train_metadata_source_resolution": "source_shard + row_group + row_in_group resolves original text in pinned source parquet",
                    "validation_policy": settings["validation_policy"],
                    "reference_validation_documents": reference["splits"]["validation"]["documents"],
                    "reference_validation_prefix_truncated_documents": reference["splits"]["validation"]["prefix_truncated_documents"],
                    "qwen_token_arrays_reused": False, "deepseek_pilot_token_arrays_reused": True,
                    "preparation_signature": signature, "script_sha256": settings["script_sha256"],
                    "shard_sampling": [{key: result[key] for key in (
                        "shard", "tokens", "documents", "reference_documents", "quota", "source_sha256")}
                        for result in results],
                    "artifacts": artifacts, "benchmark_decontamination": reference.get("benchmark_decontamination", "not performed"),
                    "license": source.get("license"), "source_url": source.get("source_url"),
                    "started_utc": started, "completed_utc": stamp()}
        atomic_json(args.output / "manifest.json", manifest)
        progress.update(status="ready", phase="complete", splits=checks, updated_utc=stamp())
        atomic_json(args.output / "progress.json", progress)
        print(json.dumps({"status": "ready", "output": str(args.output), "splits": checks,
                          "prefix_sha256": prefix_digest}), flush=True)


if __name__ == "__main__":
    main()

