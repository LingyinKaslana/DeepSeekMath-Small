"""Short disposable CPT measurements; never changes the completed checkpoint."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

import cpt
from packed_data import PackedTokens


def process_io():
    fields = dict(line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())
    return {k: int(fields[k]) for k in ('read_bytes', 'write_bytes')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    config = argparse.Namespace(model='Qwen/Qwen2.5-Coder-1.5B',
        revision='df3ce67c0e24480f20468b6ef2894622d69eb73b')
    data_path = Path(__file__).resolve().parents[1] / 'data/processed/openwebmath_10m_s42'
    dataset = PackedTokens(data_path)
    dataset.explicit_mask = True
    model = cpt.load_model(config)
    reference = {k: v.detach().clone() for k, v in model.state_dict().items()}
    reference_bytes = sum(t.numel() * t.element_size() for t in reference.values())
    native_batch = cpt.batch
    # Upload uint32 storage as int32: every token ID fits signed int32. Build
    # int64 input batches on device, and share labels with unmodified input IDs.
    # These diagnostic batches contain only full blocks, with no tail padding.
    tokens = torch.from_numpy(np.array(dataset.tokens, dtype=np.int32)).pin_memory().to('cuda', non_blocking=True)
    blocks = tokens[:len(dataset) * dataset.sequence_length].reshape(-1, dataset.sequence_length)
    mask = torch.ones((16, dataset.sequence_length), device='cuda', dtype=torch.int64)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    torch.cuda.synchronize()

    def resident_batch(ds, indices):
        index = torch.tensor(list(indices), device='cuda', dtype=torch.int64)
        ids = blocks.index_select(0, index).long()
        return dict(input_ids=ids, labels=ids, attention_mask=mask[:len(indices)]), len(indices) * (ds.sequence_length - 1), ids.numel()

    # Check transferred values and labels before using the resident diagnostic.
    old, n, count = native_batch(dataset, order[:16])
    new, n2, count2 = resident_batch(dataset, order[:16])
    assert (n, count) == (n2, count2)
    assert all(torch.equal(old[k], new[k]) for k in old)
    del old, new
    results = []
    # Alternate cases to expose temperature or clock drift; each starts from
    # the same weights, optimizer, warmup and ordered batches.
    for round_number, kind in enumerate(('cpu_mmap', 'gpu_resident', 'gpu_resident', 'cpu_mmap')):
        cpt.batch = native_batch if kind == 'cpu_mmap' else resident_batch
        model.load_state_dict(reference)
        optimizer = cpt.make_optimizer(model, 1e-5, False)
        for step in range(3):
            cpt.update(model, optimizer, dataset, order[step * 16:(step + 1) * 16], 16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        io_before = process_io()
        faults_before = resource.getrusage(resource.RUSAGE_SELF)
        rows = []
        for step in range(3, 15):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            loss, norm, count = cpt.update(model, optimizer, dataset, order[step * 16:(step + 1) * 16], 16)
            torch.cuda.synchronize()
            rows.append(dict(seconds=time.perf_counter() - begin, tokens=count,
                             loss=float(loss), grad_norm=float(norm)))
        io_after = process_io()
        faults_after = resource.getrusage(resource.RUSAGE_SELF)
        result = dict(kind=kind, round=round_number, steps=len(rows), rows=rows,
                      tokens_per_second=sum(r['tokens'] for r in rows) / sum(r['seconds'] for r in rows),
                      seconds_per_step=sum(r['seconds'] for r in rows) / len(rows),
                      disk_read_bytes=io_after['read_bytes'] - io_before['read_bytes'],
                      major_faults=faults_after.ru_majflt - faults_before.ru_majflt,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        results.append(result)
        print(json.dumps(result), flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # CPU preparation and transfer costs without model computation. The input
    # submission time includes mmap lookup, conversions, stacking and pinning.
    cpt.batch = native_batch
    submit, complete = [], []
    for start in range(0, 3200, 16):
        torch.cuda.synchronize()
        began = time.perf_counter()
        values, _, _ = native_batch(dataset, order[start:start + 16])
        submit.append(time.perf_counter() - began)
        torch.cuda.synchronize()
        complete.append(time.perf_counter() - began)
        del values

    # Profile the FINAL micro-batch size, not the earlier calibration micro=2.
    del reference, tokens, blocks, mask
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    optimizer = cpt.make_optimizer(model, 1e-5, False)
    for step in range(3):
        cpt.update(model, optimizer, dataset, order[step * 16:(step + 1) * 16], 16)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as profile:
        cpt.update(model, optimizer, dataset, order[48:64], 16)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(args.output / 'batch16_trace.json'))
    (args.output / 'batch16_profile.txt').write_text(profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
    events = json.loads((args.output / 'batch16_trace.json').read_text())['traceEvents']
    copies = defaultdict(lambda: dict(calls=0, bytes=0, duration_us=0.0))
    for e in events:
        if e.get('cat') == 'gpu_memcpy':
            row = copies[e['name']]
            row['calls'] += 1
            row['bytes'] += e.get('args', {}).get('bytes', 0)
            row['duration_us'] += e.get('dur', 0)
    kernel_time_us = sum(e.get('dur', 0) for e in events if e.get('cat') == 'kernel')
    aggregate = {}
    for kind in ('cpu_mmap', 'gpu_resident'):
        rows = [row for result in results if result['kind'] == kind for row in result['rows']]
        elapsed = sum(row['seconds'] for row in rows)
        aggregate[kind] = dict(tokens_per_second=sum(row['tokens'] for row in rows) / elapsed,
                               seconds_per_step=elapsed / len(rows), measured_steps=len(rows))
    summary = dict(measured_utc=datetime.now(timezone.utc).isoformat(),
        model=config.model, sequence_length=2048, micro_batch=16, effective_batch=16,
        precision='BF16 autocast; FP32 parameters/gradients/AdamW moments',
        checkpoint_untouched=True, comparisons=results, aggregate=aggregate,
        diagnostic_extra_reference_bytes=reference_bytes,
        diagnostic_resident_token_bytes=40000000,
        cpu_batch_submit_median_ms=float(np.median(submit) * 1000),
        cpu_batch_complete_median_ms=float(np.median(complete) * 1000),
        cuda_profile_status='captured' if kernel_time_us else 'unavailable: trace contains CPU events only',
        cuda_memcpy_profile=dict(copies) if kernel_time_us else None,
        cuda_kernel_total_profile_us=kernel_time_us if kernel_time_us else None,
        profile_caution='Profiler adds CPU launch overhead; operator durations are attribution, not unprofiled utilization',
        script_sha256=cpt.sha(Path(__file__)))
    cpt.write(args.output / 'result.json', summary)
    print(json.dumps({'event': 'diagnostic_complete', **{k: v for k, v in summary.items() if k != 'comparisons'}}), flush=True)


if __name__ == '__main__':
    main()
