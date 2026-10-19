"""Run disposable batch-16 CPT with NvPerf device-wide periodic counters."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

import cpt
from packed_data import PackedTokens


def csv_position(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 4096))
        lines = stream.read().decode().splitlines()
    for line in reversed(lines):
        fields = line.split(',')
        if fields[0].isdigit() and len(fields) > 2:
            return float(fields[1])
    return None


def measured_update(model, optimizer, data, indices):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    marks = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    marks[0].record()
    values, n, tokens = cpt.batch(data, indices)
    marks[1].record()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(**values, use_cache=False).loss
    weighted = loss * 1.0
    marks[2].record()
    weighted.backward()
    marks[3].record()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss')
    marks[4].record()
    optimizer.step()
    marks[5].record()
    return loss.detach(), norm.detach(), tokens, marks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    workspace = root.parent
    tool = workspace / 'nvidia-perfsdk/Samples/NvPerfUtility/tools/build/bin/Monitoring'
    preset = workspace / 'nvidia-perfsdk/configs/gb110-llm-extended.yaml'
    shutil.copy2(preset, args.output / preset.name)
    shutil.copy2(Path(__file__), args.output / 'source.py')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    model = cpt.load_model(argparse.Namespace(model='Qwen/Qwen2.5-Coder-1.5B',
        revision='df3ce67c0e24480f20468b6ef2894622d69eb73b'))
    data = PackedTokens(root / 'data/processed/openwebmath_10m_s42')
    data.explicit_mask = True
    optimizer = cpt.make_optimizer(model, 1e-5, False)
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(42)).tolist()
    step = 0
    def run(count):
        nonlocal step
        rows = []
        for _ in range(count):
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, norm, tokens, events = measured_update(model, optimizer, data, order[step * 16:(step + 1) * 16])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rows.append(dict(step=step, seconds=elapsed, tokens=tokens, loss=float(loss), norm=float(norm),
                             cuda_span_ms={name: events[i].elapsed_time(events[i + 1]) for i, name in enumerate(
                                 ('input', 'forward_loss', 'backward', 'clip_check', 'optimizer'))}))
            step += 1
        return rows
    run(3)
    before = run(12)
    csv_path = args.output / 'training.csv'
    command = ['sudo', '-n', str(tool), '--device', '0', '--yaml', str(args.output / preset.name),
               '-n', '0.02', '--max-buffer-duration', '60', '--csv', '--output-csv', str(csv_path),
               '--csv-duration', '24']
    with (args.output / 'monitor.log').open('w') as log:
        process = subprocess.Popen(command, cwd=args.output, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 10
        while csv_position(csv_path) is None:
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Monitoring failed to start; inspect monitor.log')
            time.sleep(0.02)
        time.sleep(1)
        boundary_start = csv_position(csv_path)
        torch.cuda.reset_peak_memory_stats()
        sampled = run(32)
        # Flush decode up to the last completed GPU update before recording end.
        time.sleep(0.05)
        boundary_end = csv_position(csv_path)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        process.wait(timeout=30)
        if process.returncode != 0:
            raise RuntimeError(f'Monitoring exit code {process.returncode}')
    after = run(12)
    monitor_log = (args.output / 'monitor.log').read_text()
    if 'overflow' in monitor_log or 'NVPERF|ERR|' in monitor_log:
        raise RuntimeError('Invalid counter capture: see monitor.log')
    records = list(csv.DictReader(csv_path.open()))
    positions = [float(row['elapsed_sec']) for row in records]
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise RuntimeError('Counter timestamps must increase monotonically')
    # Endpoints map by reading the latest decoded sample (20ms periodicity).
    # Trim 0.55s from both boundaries, excluding startup/boundary uncertainty.
    selected = [row for row in records if boundary_start + 0.55 <= float(row['elapsed_sec']) <= boundary_end - 0.55]
    interval = next(key for key in records[0] if key.startswith('Sample interval'))
    weights = np.array([float(row[interval]) for row in selected])
    metrics = {}
    for key in records[0]:
        if key in ('sample_id', 'elapsed_sec') or key.startswith('sys_mem') or key == interval:
            continue
        values = np.array([float(row[key]) for row in selected])
        metrics[key] = dict(mean=float(np.average(values, weights=weights)),
                            p10=float(np.percentile(values, 10)), p50=float(np.median(values)),
                            p90=float(np.percentile(values, 90)), max=float(values.max()))
    def throughput(rows):
        return sum(row['tokens'] for row in rows) / sum(row['seconds'] for row in rows)
    result = dict(measured_utc=datetime.now(timezone.utc).isoformat(), mode='disposable full-parameter CPT',
                  model='Qwen/Qwen2.5-Coder-1.5B', sequence_length=2048, batch=16,
                  precision='BF16 autocast; FP32 parameters, gradients, AdamW state',
                  tool_command=command, preset_sha256=cpt.sha(preset), tool_sha256=cpt.sha(tool),
                  source_sha256=cpt.sha(Path(__file__)), counters_device_wide=True,
                  csv_start_boundary=boundary_start, csv_end_boundary=boundary_end,
                  steady_samples=len(selected), steady_duration_seconds=float(weights.sum() / 1000),
                  monitored_tokens_per_second=throughput(sampled),
                  unmonitored_before_tokens_per_second=throughput(before),
                  unmonitored_after_tokens_per_second=throughput(after),
                  peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                  cuda_span_mean_ms={key: float(np.mean([row['cuda_span_ms'][key] for row in sampled]))
                                     for key in sampled[0]['cuda_span_ms']},
                  cuda_span_caution='Event-to-event spans include GPU work and host launch/synchronization gaps; not kernel self time',
                  metrics=metrics, before=before, sampled=sampled, after=after)
    cpt.write(args.output / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('before', 'sampled', 'after')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
