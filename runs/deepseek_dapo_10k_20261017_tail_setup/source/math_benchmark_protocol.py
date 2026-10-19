"""Pinned DeepSeekMath base-model GSM8K/MATH few-shot CoT protocol.

Only the two CoT prompt files and their abstract base are imported. Calling
load_benchmarks configures the vendor used by build_prompt and stop_strings.
Rows retain their original order: MATH's original IDs are not unique.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

EXPECTED_COUNTS = {"gsm8k": 1319, "math": 5000}
QUESTION_FIELDS = {"gsm8k": "question", "math": "problem"}
_PROMPTS: dict[str, object] = {}
_VENDOR: Path | None = None


def evaluation_directory(vendor: Path) -> Path:
    vendor = Path(vendor).resolve()
    result = vendor / "evaluation" if (vendor / "evaluation").is_dir() else vendor
    if not (result / "datasets").is_dir():
        raise FileNotFoundError(f"No official evaluation/datasets under {vendor}")
    return result


def _load_prompts(vendor: Path) -> None:
    global _VENDOR, _PROMPTS
    evaluation = evaluation_directory(vendor)
    if _VENDOR == evaluation and _PROMPTS:
        return
    name = "_deepseekmath_benchmark_cot_prompts"
    package = types.ModuleType(name)
    package.__path__ = [str(evaluation / "few_shot_prompts")]
    sys.modules[name] = package
    modules = {}
    for filename in ("few_shot_prompting", "cot_gsm_8_shot", "cot_minerva_math_4_shot"):
        qualified = f"{name}.{filename}"
        spec = importlib.util.spec_from_file_location(
            qualified, evaluation / "few_shot_prompts" / f"{filename}.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(qualified)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        modules[filename] = module
    _PROMPTS = {
        "gsm8k": modules["cot_gsm_8_shot"].CoTGSMPrompt(),
        "math": modules["cot_minerva_math_4_shot"].MinervaMathPrompt(),
    }
    _VENDOR = evaluation


def load_benchmarks(vendor: Path) -> dict[str, list[dict]]:
    """Load complete original official test rows, never deduplicate their IDs."""
    evaluation = evaluation_directory(vendor)
    datasets = {}
    for benchmark, count in EXPECTED_COUNTS.items():
        source = evaluation / "datasets" / benchmark / "test.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(rows) != count:
            raise ValueError(f"{benchmark}: expected {count} complete test rows, got {len(rows)}")
        required = {"id", QUESTION_FIELDS[benchmark]}
        required |= {"answer", "cot"} if benchmark == "gsm8k" else {"solution", "type", "level", "category"}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"{benchmark}[{index}] missing official fields {required}")
            if not isinstance(row[QUESTION_FIELDS[benchmark]], str):
                raise TypeError(f"{benchmark}[{index}] question is not text")
        datasets[benchmark] = rows
    _load_prompts(vendor)
    return datasets


def _prompt(benchmark: str):
    if benchmark not in EXPECTED_COUNTS:
        raise ValueError(f"Unsupported benchmark {benchmark!r}")
    if not _PROMPTS:
        raise RuntimeError("Call load_benchmarks(vendor) before formatting prompts")
    return _PROMPTS[benchmark]


def build_prompt(benchmark: str, item: dict) -> str:
    """Official base prompt: empty completion, no chat template or markup."""
    return _prompt(benchmark).format_prompt(item[QUESTION_FIELDS[benchmark]], "").lstrip()


def stop_strings(benchmark: str) -> list[str]:
    return list(_prompt(benchmark).stop_words())
