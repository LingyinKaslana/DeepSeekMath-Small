"""Build a deterministic token-budgeted subset from every OpenWebMath shard.

Sample shuffled row groups and rows, with equal token quotas per source shard.
Split by normalized text hash; reject repeated selected texts and URLs across
both splits. Output flat uint32 token arrays, original text JSONL, and provenance.
This is stratified document sampling, not uniform sampling of corpus tokens.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    p.add_argument("--model-revision", default="df3ce67c0e24480f20468b6ef2894622d69eb73b")
    p.add_argument("--train-tokens", type=int, default=10000000)
    p.add_argument("--validation-tokens", type=int, default=262144)
    p.add_argument("--block-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    assert min(a.train_tokens, a.validation_tokens, a.block_length) > 0
    source = json.loads(a.source_manifest.read_text())
    assert source["status"] == "verified"
    a.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision, local_files_only=True)
    assert tokenizer.eos_token_id is not None
    rng = random.Random(a.seed)
    files = sorted(source["files"], key=lambda f: f["name"])
    totals = {"train": a.train_tokens, "validation": a.validation_tokens}
    arrays = {s: np.lib.format.open_memmap(a.output / f"{s}_tokens.npy", mode="w+",
                                         dtype=np.uint32, shape=(n,)) for s, n in totals.items()}
    handles = {s: (a.output / f"{s}.jsonl").open("x") for s in totals}
    offsets = dict.fromkeys(totals, 0)
    documents = dict.fromkeys(totals, 0)
    seen_hashes, seen_urls = set(), set()
    rejects = Counter()
    shard_stats = []
    validation_fraction = a.validation_tokens / (a.train_tokens + a.validation_tokens)
    try:
        for index, file in enumerate(files):
            quotas = {s: n // len(files) + (index < n % len(files)) for s, n in totals.items()}
            used = dict.fromkeys(totals, 0)
            counts = dict.fromkeys(totals, 0)
            reader = pq.ParquetFile(Path(source["snapshot"]) / file["name"])
            groups = list(range(reader.num_row_groups))
            rng.shuffle(groups)
            done = False
            for group in groups:
                rows = reader.read_row_group(group, columns=["text", "url"]).to_pylist()
                order = list(range(len(rows)))
                rng.shuffle(order)
                for row_index in order:
                    row = rows[row_index]
                    text, url = row["text"], row["url"]
                    if not isinstance(text, str) or not text.strip():
                        rejects["empty"] += 1
                        continue
                    # Bound CPU tokenization cost for this small pilot.
                    if len(text) > 200000:
                        rejects["over_200000_characters"] += 1
                        continue
                    doc_hash = normalized_hash(text)
                    split_value = int(doc_hash[:16], 16) / 2**64
                    split = "validation" if split_value < validation_fraction else "train"
                    if used[split] >= quotas[split]:
                        continue
                    if doc_hash in seen_hashes or (url and url in seen_urls):
                        rejects["duplicate_selected_text_or_url"] += 1
                        continue
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if len(ids) < 64:
                        rejects["under_64_tokens"] += 1
                        continue
                    ids.append(tokenizer.eos_token_id)
                    take = min(len(ids), quotas[split] - used[split])
                    start = offsets[split]
                    arrays[split][start:start + take] = ids[:take]
                    record = {"text": text, "url": url, "text_sha256": doc_hash,
                              "source_shard": file["name"], "row_group": group, "row_in_group": row_index,
                              "document_tokens_with_eos": len(ids), "used_tokens": take,
                              "token_offset": start, "prefix_truncated": take < len(ids)}
                    handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    offsets[split] += take
                    used[split] += take
                    counts[split] += 1
                    documents[split] += 1
                    seen_hashes.add(doc_hash)
                    if url:
                        seen_urls.add(url)
                    if used == quotas:
                        done = True
                        break
                if done:
                    break
            assert used == quotas, (file["name"], used, quotas)
            shard_stats.append({"shard": file["name"], "tokens": used, "documents": counts})
            if (index + 1) % 10 == 0 or index + 1 == len(files):
                print(json.dumps({"shards_done": index + 1, "tokens": offsets, "documents": documents}), flush=True)
        for array in arrays.values():
            array.flush()
    finally:
        for handle in handles.values():
            handle.close()

    # Independent disk round-trip and split checks before declaring readiness.
    checks = {}
    split_hashes, split_urls = {}, {}
    for split, expected_tokens in totals.items():
        array = np.load(a.output / f"{split}_tokens.npy", mmap_mode="r")
        assert array.dtype == np.uint32 and len(array) == expected_tokens
        assert int(array.max()) < len(tokenizer)
        rows = [json.loads(line) for line in (a.output / f"{split}.jsonl").open()]
        assert sum(r["used_tokens"] for r in rows) == expected_tokens
        assert [r["token_offset"] for r in rows] == list(np.cumsum([0] + [r["used_tokens"] for r in rows[:-1]]))
        split_hashes[split] = {r["text_sha256"] for r in rows}
        split_urls[split] = {r["url"] for r in rows if r["url"]}
        assert len(split_hashes[split]) == len(rows)
        # Check token prefixes for the first, middle and last document in each split.
        for i in {0, len(rows) // 2, len(rows) - 1}:
            row = rows[i]
            ids = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
            np.testing.assert_array_equal(array[row["token_offset"]:row["token_offset"] + row["used_tokens"]],
                                          np.asarray(ids[:row["used_tokens"]], dtype=np.uint32))
        checks[split] = {"tokens": len(array), "documents": len(rows),
                         "full_blocks": len(array) // a.block_length,
                         "tokens_in_full_blocks": len(array) // a.block_length * a.block_length,
                         "remainder_tokens": len(array) % a.block_length,
                         "prefix_truncated_documents": sum(r["prefix_truncated"] for r in rows)}
    assert split_hashes["train"].isdisjoint(split_hashes["validation"])
    assert split_urls["train"].isdisjoint(split_urls["validation"])
    artifacts = {f.name: {"bytes": f.stat().st_size, "sha256": digest_file(f)}
                 for f in a.output.iterdir() if f.is_file()}
    manifest = {
        "status": "ready", "purpose": "Small OpenWebMath-only CPT pilot and held-out text loss evaluation",
        "source_repo": source["repo"], "source_revision": source["revision"],
        "source_manifest": str(a.source_manifest.resolve()), "source_manifest_sha256": digest_file(a.source_manifest),
        "tokenizer": a.model, "tokenizer_revision": a.model_revision, "seed": a.seed,
        "sampling": "equal token quotas across all shards; shuffled row groups and rows",
        "split": "normalized-text SHA256 threshold, before tokenization; selected text/URL deduplication",
        "packing": "document text + EOS, concatenated within each split; no cross-document attention mask",
        "block_length": a.block_length, "dtype": "uint32", "splits": checks,
        "train_validation_text_hash_overlap": 0, "train_validation_url_overlap": 0,
        "rejected": dict(rejects), "artifacts": artifacts, "shard_sampling": shard_stats,
        "benchmark_decontamination": "not performed; held-out text loss only, no benchmark generalization claims",
        "license": source["license"], "source_url": source["source_url"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest_file(Path(__file__))}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "splits": checks}), flush=True)


if __name__ == "__main__":
    main()
