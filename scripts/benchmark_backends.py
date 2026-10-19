"""Compare disposable CPT backends under fixed data, batch, precision and LR."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
import traceback

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

import cpt
from packed_data import PackedTokens
from token_batches import PackedTokenBatches


PROBES = ('model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.down_proj.weight',
          'model.layers.0.input_layernorm.weight', 'model.layers.27.self_attn.q_proj.weight',
          'model.layers.27.mlp.down_proj.weight', 'model.layers.27.input_layernorm.weight',
          'model.norm.weight', 'model.embed_tokens.weight')


def compare_gradients(model, reference):
    differences, magnitudes, probes = [], [], {}
    for name, p in model.named_parameters():
        old = reference[name]
        difference = (p.grad - old).square().sum()
        magnitude = old.square().sum()
        differences.append(difference)
        magnitudes.append(magnitude)
        if name in PROBES:
            probes[name] = float((difference / magnitude.clamp_min(1e-24)).sqrt())
    global_error = float((torch.stack(differences).sum() / torch.stack(magnitudes).sum()).sqrt())
    return global_error, probes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kernels', default='native')
    p.add_argument('--loss', default='native')
    p.add_argument('--chunk-mem-const', type=int, default=1)
    p.add_argument('--compile-scope', choices=['none', 'layers', 'model', 'mlp'], default='none')
    p.add_argument('--compile-mode', default='default')
    p.add_argument('--compile-label-alias', choices=['any', 'same_as_input'], default='any')
    p.add_argument('--allow-graph-breaks', action='store_true')
    p.add_argument('--rms-in-place', action='store_true')
    p.add_argument('--emulate-precision-casts', action='store_true')
    p.add_argument('--preserve-silu', action='store_true')
    p.add_argument('--ce-chunk-rows', type=int, default=4096)
    p.add_argument('--legacy-data', action='store_true')
    p.add_argument('--mask-policy', choices=['ones', 'causal'], default='causal')
    p.add_argument('--micro-batch', type=int, default=16)
    p.add_argument('--effective-batch', type=int, default=16)
    p.add_argument('--steps', type=int, default=24)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--fused-optimizer', action='store_true')
    p.add_argument('--compiled-optimizer', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    data_path = root / 'data/processed/openwebmath_10m_s42'
    source_names = ['benchmark_backends.py', 'model_backends.py', 'token_batches.py', 'cpt.py', 'packed_data.py', 'compiled_adamw.py', 'native_ops.py', 'native_chunked_ce.py']
    cpt.write(args.output / 'manifest.json', dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        started_utc=datetime.now(timezone.utc).isoformat(), data_manifest_sha256=cpt.sha(data_path / 'manifest.json'),
        source_sha256={n: cpt.sha(Path(__file__).parent / n) for n in source_names}))
    telemetry = cpt.Telemetry(args.output)
    telemetry.thread.start()
    try:
        from model_backends import configure_model, eager_native
        model = cpt.load_model(argparse.Namespace(model='Qwen/Qwen2.5-Coder-1.5B', revision='df3ce67c0e24480f20468b6ef2894622d69eb73b'))
        model.train()
        assert not getattr(model.config, 'use_sliding_window', False)
        dataset = PackedTokens(data_path)
        dataset.explicit_mask = args.mask_policy == 'ones'
        provider = cpt.batch if args.legacy_data else PackedTokenBatches()
        options = dict(kernels=args.kernels, loss=args.loss, chunk_mem_const=args.chunk_mem_const,
                       compile_scope=args.compile_scope, compile_mode=args.compile_mode,
                       compile_shape=[args.micro_batch, 2048], compile_fullgraph=not args.allow_graph_breaks,
                       compile_label_alias=args.compile_label_alias,
                       rms_in_place=args.rms_in_place, emulate_precision_casts=args.emulate_precision_casts,
                       preserve_silu=args.preserve_silu, ce_chunk_rows=args.ce_chunk_rows)
        # Collect the reference BEFORE any instance-local patch is applied.
        numerical_started = time.perf_counter()
        values, _, _ = cpt.batch(dataset, [0])
        values = {k: v[:, :512] for k, v in values.items()}
        values['attention_mask'] = torch.ones_like(values['input_ids'])
        with sdpa_kernel(SDPBackend.MATH):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**values).loss
            loss.backward()
        reference = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        expected_loss = float(loss.detach())
        model.zero_grad(set_to_none=True)
        del loss
        forward, metadata = configure_model(model, options)
        if args.mask_policy == 'causal':
            values['attention_mask'] = {'full_attention': None}
        with sdpa_kernel(SDPBackend.MATH):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                force = {'_cpt_force_compile': True} if args.compile_scope != 'none' else {}
                loss = forward(**values, **force).loss
            loss.backward()
        gradient_error, probes = compare_gradients(model, reference)
        loss_error = abs(float(loss.detach()) - expected_loss)
        passed = loss_error < .003 and gradient_error < .05 and max(probes.values()) < .05
        numerical = dict(passed=passed, native_loss=expected_loss, backend_loss=float(loss.detach()),
                         absolute_loss_difference=loss_error, global_gradient_relative_l2=gradient_error,
                         probe_gradient_relative_l2=probes, attention='math', sequence_length=512,
                         compiled_check='compiled small-shape math forward/backward' if args.compile_scope != 'none' else 'eager backend')
        torch.cuda.synchronize()
        numerical['seconds_including_small_shape_compile'] = time.perf_counter() - numerical_started
        cpt.write(args.output / 'numerical_check.json', numerical)
        print(json.dumps({'event': 'numerical_check', **numerical}), flush=True)
        del reference, values, loss
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        if not passed:
            cpt.write(args.output / 'result.json', dict(status='rejected_numerical_check', options=options, numerical=numerical))
            return
        optimizer_check, optimizer_metadata = None, None
        if args.compiled_optimizer:
            from compiled_adamw import make_compiled_adamw, verify_compiled_adamw
            optimizer_check = verify_compiled_adamw()
            cpt.write(args.output / 'optimizer_check.json', optimizer_check)
            if optimizer_check['status'] != 'passed':
                raise RuntimeError('Compiled optimizer failed its parameter/moment checks')
            optimizer, optimizer_step, optimizer_metadata = make_compiled_adamw(model.parameters(), 1e-5)
        else:
            optimizer = cpt.make_optimizer(model, 1e-5, args.fused_optimizer)
            optimizer_step = optimizer.step
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()

        def update(indices):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batches = [provider(dataset, indices[i:i + args.micro_batch]) for i in range(0, len(indices), args.micro_batch)]
            total = sum(n for _, n, _ in batches)
            loss_sum = torch.zeros((), device='cuda')
            for values, n, _ in batches:
                if args.mask_policy == 'causal':
                    values['attention_mask'] = {'full_attention': None}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = forward(**values, use_cache=False).loss
                weighted = loss * (n / total)
                weighted.backward()
                loss_sum += weighted.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(loss_sum):
                raise RuntimeError('Nonfinite loss')
            optimizer_step()
            return loss_sum.detach(), norm.detach(), sum(n for _, _, n in batches)

        records = []
        first_update_seconds = None
        for step in range(args.warmup + args.steps):
            telemetry.phase = 'warmup' if step < args.warmup else 'measured'
            if step == args.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
            begin = time.perf_counter()
            loss, norm, count = update(order[step * args.effective_batch:(step + 1) * args.effective_batch])
            if step == 0:
                torch.cuda.synchronize()
                first_update_seconds = time.perf_counter() - begin
                print(json.dumps({'event': 'first_update', 'seconds': first_update_seconds}), flush=True)
            records.append((loss, norm, count))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        measured = records[args.warmup:]
        result = dict(status='passed', options=options, metadata=metadata, micro_batch=args.micro_batch,
            effective_batch=args.effective_batch, mask_policy=args.mask_policy,
            fused_optimizer=args.fused_optimizer, legacy_data=args.legacy_data,
            compiled_optimizer=args.compiled_optimizer, optimizer_check=optimizer_check, optimizer_metadata=optimizer_metadata,
            measured_updates=args.steps, seconds=elapsed, seconds_per_update=elapsed / args.steps,
            tokens_per_second=sum(r[2] for r in measured) / elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            first_update_including_compile_seconds=first_update_seconds,
            numerical=numerical, losses=[float(r[0]) for r in measured], grad_norms=[float(r[1]) for r in measured],
            telemetry=telemetry.summary('measured'), precision='BF16 autocast; FP32 parameters/gradients/AdamW moments')
        assert all(p.dtype == torch.float32 and p.grad.dtype == torch.float32 for p in model.parameters())
        assert all(s['exp_avg'].dtype == torch.float32 and s['exp_avg_sq'].dtype == torch.float32 for s in optimizer.state.values())
        cpt.write(args.output / 'result.json', result)
        print(json.dumps({'event': 'benchmark_complete', **result}), flush=True)
    except Exception as error:
        cpt.write(args.output / 'result.json', dict(status='failed', error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        telemetry.stop.set()
        telemetry.thread.join(timeout=6)


if __name__ == '__main__':
    main()
