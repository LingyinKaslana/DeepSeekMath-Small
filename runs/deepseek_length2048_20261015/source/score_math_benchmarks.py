"""Score complete GSM8K/MATH generations using official CoT evaluators.

Generation text is never executed as a Python program. Official numerical and
symbolic answer comparison runs in persistent CPU processes; a hard per-item
deadline kills and replaces a worker, even if a vendor function catches errors.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import importlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import sys
import time
import types

from math_benchmark_protocol import EXPECTED_COUNTS, QUESTION_FIELDS, evaluation_directory, load_benchmarks

_OFFICIAL = None


def load_official_scorer(vendor: Path):
    global _OFFICIAL
    evaluation = evaluation_directory(vendor)
    if _OFFICIAL is not None and _OFFICIAL[0] == evaluation:
        return _OFFICIAL[1:]
    # Namespace packages avoid the prompt __init__, vLLM, PAL and code execution.
    for name in ("data_processing", "eval"):
        for old in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
            del sys.modules[old]
        package = types.ModuleType(name)
        package.__path__ = [str(evaluation / name)]
        sys.modules[name] = package
    extraction = importlib.import_module("data_processing.answer_extraction")
    scoring = importlib.import_module("eval.eval_script")
    _OFFICIAL = (evaluation, extraction, scoring)
    return extraction, scoring


def official_gold(benchmark: str, item: dict, extraction):
    if benchmark == "gsm8k":
        return item["answer"].replace(",", "")
    return extraction.extract_math_answer(item["problem"], item["solution"], task="cot")


def empty_prediction(prediction) -> bool:
    if isinstance(prediction, list):
        return not prediction or all(empty_prediction(value) for value in prediction)
    return prediction is None or str(prediction).strip() in ("", "[invalid]")


def score_completion(benchmark: str, item: dict, completion: str, vendor: Path) -> dict:
    """CPU oracle used by workers and small correctness checks."""
    extraction, scoring = load_official_scorer(vendor)
    question = item[QUESTION_FIELDS[benchmark]]
    gold = official_gold(benchmark, item, extraction)
    if benchmark == "gsm8k":
        prediction = extraction.extract_gsm_few_shot_cot_answer(question, completion, task="cot")
        sample = {"prediction": prediction, "answer": gold}
        correct = bool(scoring.eval_last_single_answer(sample))
        # Clearly secondary: fixing thousands separators changes official parsing.
        diagnostic_text = re.sub(r"(?<=\d),(?=\d)", "", completion)
        secondary = extraction.extract_gsm_few_shot_cot_answer(question, diagnostic_text, task="cot")
        secondary_correct = bool(scoring.eval_last_single_answer({"prediction": secondary, "answer": gold}))
        diagnostic = {"gsm8k_comma_normalized_prediction": secondary,
                      "gsm8k_comma_normalized_correct": secondary_correct}
    elif benchmark == "math":
        prediction = extraction.extract_math_few_shot_cot_answer(question, completion, task="cot")
        sample = {"prediction": prediction, "answer": gold}
        correct = bool(scoring.eval_math(sample))
        diagnostic = {}
    else:
        raise ValueError(benchmark)
    return {"gold": gold, "prediction": prediction,
            "official_evaluated_prediction": sample["prediction"], "correct": correct,
            "empty_prediction": empty_prediction(prediction), "timeout": False,
            "error": None, **diagnostic}


def _worker(connection, vendor: str):
    try:
        load_official_scorer(Path(vendor))
        connection.send({"ready": True})
    except Exception as exc:
        connection.send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    while True:
        try:
            job = connection.recv()
        except EOFError:
            break
        if job is None:
            break
        try:
            result = score_completion(job["benchmark"], job["gold_row"], job["completion"], Path(vendor))
        except Exception as exc:
            result = {"correct": False, "timeout": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        connection.send(result)
    connection.close()


def _start_worker(context, vendor: Path):
    parent, child = context.Pipe()
    process = context.Process(target=_worker, args=(child, str(vendor)), daemon=True)
    process.start()
    child.close()
    if not parent.poll(30):
        process.kill()
        process.join()
        parent.close()
        raise RuntimeError("Official CPU scorer failed to initialize within 30 seconds")
    status = parent.recv()
    if not status.get("ready"):
        process.join()
        parent.close()
        raise RuntimeError(f"Official scorer initialization failed: {status}")
    return {"process": process, "connection": parent, "job": None, "deadline": None}


def _stop_worker(worker):
    if worker["process"].is_alive():
        worker["process"].kill()
    worker["process"].join()
    worker["connection"].close()


def parallel_score(jobs: list[dict], vendor: Path, workers: int, timeout: float):
    context = mp.get_context("spawn")
    pool = []
    pending = deque(jobs)
    finished = 0
    try:
        for _ in range(min(workers, len(jobs))):
            pool.append(_start_worker(context, vendor))
        while pending or any(worker["job"] is not None for worker in pool):
            for worker in pool:
                if worker["job"] is None and pending:
                    job = pending.popleft()
                    worker["connection"].send(job)
                    worker["job"] = job
                    worker["deadline"] = time.monotonic() + timeout
            active = [worker for worker in pool if worker["job"] is not None]
            remaining = max(0., min(worker["deadline"] for worker in active) - time.monotonic())
            readable = set(wait([worker["connection"] for worker in active], timeout=min(.1, remaining)))
            for number, worker in enumerate(pool):
                if worker["job"] is None:
                    continue
                result = None
                if worker["connection"] in readable:
                    try:
                        result = worker["connection"].recv()
                    except (EOFError, ConnectionResetError):
                        result = {"correct": False, "timeout": False, "error": "Scoring worker exited"}
                elif time.monotonic() >= worker["deadline"]:
                    result = {"correct": False, "timeout": True,
                              "error": f"Official scoring exceeded {timeout:g} seconds"}
                if result is not None:
                    job = worker["job"]
                    worker["job"] = None
                    finished += 1
                    if result.get("timeout") or not worker["process"].is_alive():
                        _stop_worker(worker)
                        pool[number] = _start_worker(context, vendor)
                    yield job, result
                    if finished % 100 == 0 or finished == len(jobs):
                        print(json.dumps({"event": "scoring_progress", "scored": finished,
                                          "total": len(jobs)}), flush=True)
    finally:
        for worker in pool:
            _stop_worker(worker)


def _read_predictions(run: Path, benchmark: str, gold_rows: list[dict], allow_partial: bool):
    path = run / f"{benchmark}_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    joined = {}
    for row in rows:
        index = row.get("dataset_index")
        if type(index) is not int or not 0 <= index < len(gold_rows) or index in joined:
            raise ValueError(f"{path}: invalid or duplicate dataset_index {index!r}")
        gold = gold_rows[index]
        if row.get("benchmark") != benchmark:
            raise ValueError(f"{path}: wrong benchmark at {index}")
        if str(row.get("dataset_id")) != str(gold["id"]):
            raise ValueError(f"{path}: dataset_id mismatch at {index}")
        if row.get("question") != gold[QUESTION_FIELDS[benchmark]]:
            raise ValueError(f"{path}: question mismatch at {index}")
        if not isinstance(row.get("completion"), str):
            raise TypeError(f"{path}: completion is not text at {index}")
        joined[index] = row
    if not allow_partial and len(joined) != len(gold_rows):
        raise ValueError(f"{path}: expected all {len(gold_rows)} predictions, got {len(joined)}")
    return joined


def _metrics(rows: list[dict]) -> dict:
    correct = sum(bool(row["correct"]) for row in rows)
    count = len(rows)
    return {"total": count, "correct": correct, "accuracy": correct / count if count else None,
            "accuracy_percent": correct / count * 100 if count else None,
            "timeouts": sum(bool(row.get("timeout")) for row in rows),
            "errors": sum(bool(row.get("error")) for row in rows),
            "empty_predictions": sum(bool(row.get("empty_prediction")) for row in rows),
            "truncated_generations": sum(bool(row.get("truncated")) for row in rows),
            "missing_generations": sum(bool(row.get("missing_generation")) for row in rows)}


def _write_json(path: Path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    args.run = args.run.resolve()
    args.vendor = args.vendor.resolve()
    datasets = load_benchmarks(args.vendor)
    extraction, _ = load_official_scorer(args.vendor)
    # Fail before scoring if symbolic equivalence would silently lack LaTeX support.
    from sympy.parsing.latex import parse_latex
    if parse_latex(r"\frac{1}{2}") != parse_latex("0.5"):
        from sympy import simplify
        if simplify(parse_latex(r"\frac{1}{2}") - parse_latex("0.5")) != 0:
            raise RuntimeError("SymPy LaTeX equivalence smoke failed")
    joined = {benchmark: _read_predictions(args.run, benchmark, rows, args.allow_partial)
              for benchmark, rows in datasets.items()}
    output = {benchmark: [None] * len(rows) for benchmark, rows in datasets.items()}
    jobs = []
    for benchmark, gold_rows in datasets.items():
        for index, gold in enumerate(gold_rows):
            row = joined[benchmark].get(index)
            if row is None:
                output[benchmark][index] = {
                    "benchmark": benchmark, "dataset_index": index, "dataset_id": gold["id"],
                    "question": gold[QUESTION_FIELDS[benchmark]], "completion": "",
                    "gold": official_gold(benchmark, gold, extraction), "prediction": None,
                    "correct": False, "timeout": False, "error": "Missing generation",
                    "empty_prediction": True, "missing_generation": True, "truncated": False,
                }
            else:
                jobs.append({"benchmark": benchmark, "dataset_index": index,
                             "gold_row": gold, "completion": row["completion"]})
    for job, result in parallel_score(jobs, args.vendor, args.workers, args.timeout):
        benchmark, index = job["benchmark"], job["dataset_index"]
        original = joined[benchmark][index]
        gold = datasets[benchmark][index]
        scored = {**original, "gold": official_gold(benchmark, gold, extraction),
                  "prediction": None, "empty_prediction": not original["completion"].strip(),
                  **result, "missing_generation": False,
                  "truncated": original.get("finish_reason") in ("length", "max_tokens", "max_new_tokens")}
        if benchmark == "math":
            scored.update({key: gold[key] for key in ("type", "level", "category")})
        output[benchmark][index] = scored
    scores = {"protocol": "Official DeepSeekMath base-model few-shot CoT, greedy single completion",
              "status": "complete" if not args.allow_partial else "complete_scoring_partial_generation_allowed",
              "run": str(args.run), "vendor": str(args.vendor), "workers": args.workers,
              "per_item_timeout_seconds": args.timeout,
              "denominator": "All official test rows; timeouts/errors/missing count as incorrect",
              "generated_python_execution": False, "benchmarks": {}}
    for benchmark, rows in output.items():
        if any(row is None for row in rows):
            raise RuntimeError(f"Incomplete internal scoring results for {benchmark}")
        destination = args.run / f"{benchmark}_scored.jsonl"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
        metrics = _metrics(rows)
        metrics["scored_file"] = str(destination)
        metrics["prediction_sha256"] = hashlib.sha256(
            (args.run / f"{benchmark}_predictions.jsonl").read_bytes()).hexdigest()
        if benchmark == "math":
            for key in ("type", "level", "category"):
                groups = defaultdict(list)
                for row in rows:
                    value = row.get(key, datasets[benchmark][row["dataset_index"]][key])
                    groups[value].append(row)
                metrics[f"by_{key}"] = {value: _metrics(group) for value, group in sorted(groups.items())}
        else:
            count = sum(bool(row.get("gsm8k_comma_normalized_correct")) for row in rows)
            metrics["secondary_comma_normalized"] = {
                "label": "Diagnostic only; not official main score", "correct": count,
                "total": len(rows), "accuracy": count / len(rows),
                "changed_predictions": sum(row.get("gsm8k_comma_normalized_prediction") != row.get("prediction")
                                           for row in rows if "gsm8k_comma_normalized_prediction" in row)}
        scores["benchmarks"][benchmark] = metrics
    _write_json(args.run / "scores.json", scores)
    print(json.dumps(scores, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
