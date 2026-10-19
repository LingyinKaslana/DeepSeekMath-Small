"""Download a pinned OpenWebMath snapshot and verify all Parquet shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
import pyarrow.parquet as pq


def verify(item):
    path, expected = item
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    assert path.stat().st_size == expected["size"], path
    assert digest == expected["sha256"], path
    metadata = pq.read_metadata(path)
    return {**expected, "rows": metadata.num_rows, "row_groups": metadata.num_row_groups}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--revision", default="fde8ef8de2300f5e778f56261843dab89f230815")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    repo = "open-web-math/open-web-math"
    info = HfApi().dataset_info(repo, revision=a.revision, files_metadata=True)
    expected = [{"name": f.rfilename, "size": f.size, "sha256": f.lfs.sha256}
                for f in info.siblings if f.rfilename.endswith(".parquet")]
    manifest = {"repo": repo, "revision": info.sha, "status": "downloading",
                "expected_files": expected, "expected_bytes": sum(f["size"] for f in expected)}
    target = a.output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloading {len(expected)} shards, {manifest['expected_bytes']} bytes", flush=True)
    snapshot = Path(snapshot_download(repo, repo_type="dataset", revision=info.sha,
                                     allow_patterns=["data/*.parquet", "README.md", "LICENSE*"],
                                     max_workers=8))
    print("Download finished; verifying SHA256 and Parquet metadata", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify, [(snapshot / f["name"], f) for f in expected]))
    manifest.update(status="verified", snapshot=str(snapshot.resolve()), files=verified,
                    total_rows=sum(f["rows"] for f in verified),
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    license="ODC-By-1.0; underlying content licenses retained; see source README",
                    source_url=f"https://huggingface.co/datasets/{repo}",
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["status", "revision", "expected_bytes", "total_rows", "snapshot"]}), flush=True)


if __name__ == "__main__":
    main()
