"""Compare the selected CPT backend to native Flash SDPA at production shape.

Both paths use the same untouched base weights, real packed-token microbatch,
BF16 forward autocast, and FP32 gradients with backward outside autocast.
No optimizer update is performed. Thresholds match benchmark_backends.py.
Checks use deterministic Flash backward by default to isolate backend rounding
from the nondeterministic backward reductions used in normal training.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer

from benchmark_backends import PROBES, compare_gradients
import cpt
from model_backends import configure_model, eager_native
from packed_data import PackedTokens
from token_batches import PackedTokenBatches
from train_cpt import check_model_safety, load_backend, verify_data


LOSS_TOLERANCE = .003
GRADIENT_TOLERANCE = .05
CONTROL_TOLERANCE = 1e-6
SOURCE_FILES = ("verify_backend_numerics.py", "benchmark_backends.py", "model_backends.py",
                "token_batches.py", "packed_data.py", "cpt.py", "train_cpt.py", "compiled_adamw.py", "native_ops.py", "native_chunked_ce.py")


def strict_json(path, value):
    """Serialize numerical failures as null plus explicit status, never NaN JSON."""
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(value) for value in item]
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def native_control(model, values):
    # Keep training mode identical to the reference; SDPA dropout is zero for
    # this selected model. The explicit context bypasses local model patches.
    with eager_native(model), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**values, use_cache=False).loss
    torch.cuda.synchronize()
    return float(loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-nondeterministic-flash", action="store_true",
                        help="Diagnostic only: allow ordinary nondeterministic Flash backward")
    args = parser.parse_args()
    started = time.perf_counter()
    # Set this before the first CUDA availability/device/tensor operation.
    # Importing Torch above does not initialize the CUDA context. cuBLAS reads
    # its workspace setting during handle initialization, not at this assignment.
    if not args.allow_nondeterministic_flash:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(not args.allow_nondeterministic_flash)
    config = load_backend(args.backend_config)
    _, token_hashes = verify_data(args.data, config)
    args.output.mkdir(parents=True, exist_ok=False)
    script_directory = Path(__file__).resolve().parent
    manifest = {
        "schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend_config_path": str(args.backend_config.resolve()),
        "backend_config_sha256": cpt.sha(args.backend_config), "backend_config": config,
        "data_path": str(args.data.resolve()), "data_manifest_sha256": cpt.sha(args.data / "manifest.json"),
        "token_file_sha256": token_hashes,
        "source_sha256": {name: cpt.sha(script_directory / name) for name in SOURCE_FILES},
        "model": config["model"], "model_revision": config["revision"],
        "shape": [config["micro_batch"], config["sequence_length"]],
        "sampling": "First microbatch from the same seed-42 one-epoch shuffled block order; includes tail in the shuffle",
        "thresholds": {"absolute_loss_difference": LOSS_TOLERANCE,
                       "global_gradient_relative_l2": GRADIENT_TOLERANCE,
                       "each_existing_probe_relative_l2": GRADIENT_TOLERANCE,
                       "native_restore_control_absolute_loss_difference": CONTROL_TOLERANCE},
        "precision": "BF16 forward autocast; FP32 master weights/gradients; backward outside autocast",
        "attention": "FLASH_ATTENTION exclusively for both forward/backward paths; no fallback",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_nondeterministic_flash": args.allow_nondeterministic_flash,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "determinism_note": "Deterministic checks isolate compiler rounding from production nondeterministic Flash backward; this script does not change training settings",
        "optimizer_updates": 0,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "liger-kernel", "numpy")},
    }
    strict_json(args.output / "manifest.json", manifest)
    telemetry = cpt.Telemetry(args.output)
    telemetry.thread.start()
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This verifier requires CUDA with BF16 support")
        torch.set_num_threads(8)
        torch.manual_seed(42)
        model = cpt.load_model(argparse.Namespace(model=config["model"], revision=config["revision"]))
        check_model_safety(model)
        model.train()
        if model.config.attention_dropout != 0:
            raise ValueError("The numerical comparison requires deterministic zero-dropout attention")
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"], local_files_only=True)
        dataset = PackedTokens(args.data, sequence_length=config["sequence_length"], include_tail=True,
                               pad_token_id=tokenizer.eos_token_id)
        dataset.explicit_mask = True
        if len(dataset) < config["micro_batch"] + 1:
            raise ValueError("Need a complete production microbatch plus one independent control block")
        provider = PackedTokenBatches()
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
        indices = order[:config["micro_batch"]]
        values, supervised_tokens, input_tokens = provider(dataset, indices)
        control_index = order[config["micro_batch"]]
        control_values, _, _ = provider(dataset, [control_index])
        manifest.update(
            device={"name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                    "index": torch.cuda.current_device(), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "torch_cuda_runtime": torch.version.cuda},
            block_indices=indices, control_block_index=control_index,
            input_tokens=input_tokens, supervised_tokens=supervised_tokens,
            parameter_tensors=len(list(model.named_parameters())),
        )
        strict_json(args.output / "manifest.json", manifest)
        telemetry.phase = "native_flash_reference"
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        native_started = time.perf_counter()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**values, use_cache=False).loss
            loss.backward()
        torch.cuda.synchronize()
        native_seconds = time.perf_counter() - native_started
        native_loss = float(loss.detach())
        if not math.isfinite(native_loss):
            raise RuntimeError("Native Flash reference loss is nonfinite")
        if any(parameter.grad is None or parameter.grad.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("Native reference must produce all FP32 parameter gradients")
        reference = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        del loss
        control_expected = native_control(model, control_values)
        forward, metadata = configure_model(model, config["options"])
        if config["mask_policy"] == "causal":
            values["attention_mask"] = {"full_attention": None}
        telemetry.phase = "selected_backend_flash"
        torch.cuda.synchronize()
        candidate_started = time.perf_counter()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forward(**values, use_cache=False).loss
            loss.backward()
        torch.cuda.synchronize()
        candidate_seconds = time.perf_counter() - candidate_started
        candidate_loss = float(loss.detach())
        if any(parameter.grad is None or parameter.grad.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("Selected backend must produce all FP32 parameter gradients")
        gradient_error, probes = compare_gradients(model, reference)
        loss_error = abs(candidate_loss - native_loss)
        missing_probes = sorted(set(PROBES) - probes.keys())
        finite = all(math.isfinite(value) for value in (candidate_loss, loss_error, gradient_error, *probes.values()))
        numerical_passed = finite and not missing_probes and loss_error < LOSS_TOLERANCE and gradient_error < GRADIENT_TOLERANCE and all(value < GRADIENT_TOLERANCE for value in probes.values())
        del reference, loss
        model.zero_grad(set_to_none=True)
        gc.collect()
        telemetry.phase = "native_restoration_control"
        control_actual = native_control(model, control_values)
        control_error = abs(control_actual - control_expected)
        control_passed = math.isfinite(control_error) and control_error <= CONTROL_TOLERANCE
        result = {
            "schema_version": 1, "status": "passed" if numerical_passed and control_passed else "rejected_numerical_check",
            "passed": numerical_passed and control_passed, "numerical_passed": numerical_passed, "finite": finite,
            "native_loss": native_loss, "backend_loss": candidate_loss,
            "absolute_loss_difference": loss_error, "global_gradient_relative_l2": gradient_error,
            "probe_gradient_relative_l2": probes, "missing_existing_probes": missing_probes,
            "native_restore_control": {"passed": control_passed, "block_index": control_index,
                                       "native_loss_before_configuration": control_expected,
                                       "native_loss_after_configuration": control_actual,
                                       "absolute_loss_difference": control_error},
            "backend_metadata": metadata, "attention": "flash_only", "shape": manifest["shape"],
            "deterministic_algorithms": manifest["deterministic_algorithms"],
            "allow_nondeterministic_flash": args.allow_nondeterministic_flash,
            "cublas_workspace_config": manifest["cublas_workspace_config"],
            "determinism_note": manifest["determinism_note"],
            "block_indices": indices, "input_tokens": input_tokens, "supervised_tokens": supervised_tokens,
            "compared_fp32_gradient_tensors": len(list(model.named_parameters())),
            "native_seconds": native_seconds, "backend_seconds_including_compile": candidate_seconds,
            "diagnostic_wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "thresholds": manifest["thresholds"], "optimizer_updates": 0,
            "model": manifest["model"], "model_revision": manifest["model_revision"],
            "device": manifest["device"], "backend_config_path": manifest["backend_config_path"],
            "backend_config_sha256": manifest["backend_config_sha256"], "data_path": manifest["data_path"],
            "data_manifest_sha256": manifest["data_manifest_sha256"],
            "token_file_sha256": token_hashes, "source_sha256": manifest["source_sha256"],
        }
        strict_json(args.output / "result.json", result)
        print(json.dumps({"event": "production_flash_numerical_check", "status": result["status"],
                          "absolute_loss_difference": loss_error if finite else None,
                          "global_gradient_relative_l2": gradient_error if finite else None,
                          "native_restore_control_passed": control_passed}, allow_nan=False), flush=True)
    except Exception as error:
        strict_json(args.output / "result.json", {"schema_version": 1, "status": "failed",
                                                "error": str(error), "traceback": traceback.format_exc(),
                                                "backend_config_sha256": manifest["backend_config_sha256"],
                                                "data_manifest_sha256": manifest["data_manifest_sha256"],
                                                "deterministic_algorithms": manifest["deterministic_algorithms"],
                                                "cublas_workspace_config": manifest["cublas_workspace_config"],
                                                "source_sha256": manifest["source_sha256"]})
        raise
    finally:
        telemetry.stop.set()
        telemetry.thread.join(timeout=6)


if __name__ == "__main__":
    main()
