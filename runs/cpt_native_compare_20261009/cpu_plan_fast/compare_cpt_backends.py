"""Controlled short CPT branches from one immutable production checkpoint.

Fast uses the frozen production compiler backend and compiled AdamW. Native
uses the unpatched Hugging Face model, ordinary stacked batches/all-ones masks,
and uncompiled torch.optim.AdamW. Both branches keep weights, optimizer moments,
step counters, FP32 state, BF16 autocast, data order, and the global LR schedule.
These are explicitly diagnostic branches, never production resume checkpoints.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def emit(value):
    print(json.dumps(clean(value), allow_nan=False), flush=True)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast('B')).hexdigest()


def tensor_sha(value):
    if value.device.type != 'cpu':
        raise ValueError('Tensor checksum expects a CPU tensor')
    return array_sha(value.detach().contiguous().numpy())


def state_audit(saved_optimizer, torch, expected_updates):
    """Audit CPU Adam state before implementation-only adaptation or transfer."""
    digest = hashlib.sha256()
    steps = []
    rows = []
    parameter_ids = [index for group in saved_optimizer['param_groups'] for index in group['params']]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != set(saved_optimizer['state']):
        raise RuntimeError('Expected exactly one complete Adam state for each parameter')
    for index in parameter_ids:
        state = saved_optimizer['state'][index]
        row = {'parameter_id': index}
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
                raise RuntimeError('Expected CPU Adam checkpoint tensors')
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise RuntimeError('Adam checkpoint contains nonfinite/non-FP32 state')
            item = {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': tensor_sha(value)}
            row[key] = item
            digest.update(json.dumps([index, key, item], sort_keys=True).encode())
        step = float(state['step'])
        if step != expected_updates:
            raise RuntimeError(f'Adam step {step} differs from saved optimizer update counter {expected_updates}')
        steps.append(step)
        rows.append(row)
    return {'tensor_count': len(rows), 'all_finite': True, 'moments_dtype': 'torch.float32',
            'step_values': sorted(set(steps)), 'combined_tensor_sha256': digest.hexdigest(), 'rows': rows}


def adapt_native_optimizer(saved_optimizer, torch):
    """Change only execution flags/LR representation; moments and steps stay CPU."""
    changes = []
    for group in saved_optimizer['param_groups']:
        original = {key: (float(value) if isinstance(value, torch.Tensor) else value)
                    for key, value in group.items() if key != 'params'}
        if tuple(group['betas']) != (0.9, 0.95) or group['weight_decay'] != 0.1 or group['eps'] != 1e-8:
            raise RuntimeError('Unexpected source AdamW math configuration')
        group['lr'] = float(group['lr'])
        group.update(capturable=False, foreach=True, fused=False)
        changed = {key: (float(value) if isinstance(value, torch.Tensor) else value)
                   for key, value in group.items() if key != 'params'}
        changes.append({'before': original, 'after': changed,
                        'permitted_changes': ['lr tensor -> numerically identical Python float',
                                              'capturable=True -> False; foreach=True; fused=False']})
    return changes



class DiagnosticNonfinite(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__(details['reason'])


def checked_step(model, optimizer, loss, step_callable, torch, step):
    """Independent standard clipping/update boundary, shared only by diagnostics."""
    if not bool(torch.isfinite(loss)):
        raise DiagnosticNonfinite({'reason': 'nonfinite_loss', 'step': step,
                                   'loss': float(loss), 'optimizer_applied': False})
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    except RuntimeError as exc:
        rows = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                rows.append({'name': name, 'nonfinite_elements': int((~torch.isfinite(parameter.grad)).sum())})
        raise DiagnosticNonfinite({'reason': 'nonfinite_gradient_norm', 'step': step, 'loss': float(loss),
                                  'nonfinite_gradients': rows, 'optimizer_applied': False,
                                  'original_error': str(exc)}) from exc
    step_callable()
    return norm


def load_inputs(args):
    run = args.run.resolve()
    checkpoint = args.checkpoint.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    config_path = run / 'backend_config.json'
    config = json.loads(config_path.read_text())
    metadata = json.loads((checkpoint / 'checkpoint_manifest.json').read_text())
    if metadata['signature'] != manifest['signature']:
        raise RuntimeError('Checkpoint signature differs from the selected source run')
    if sha(config_path) != manifest['signature']['backend_config_sha256']:
        raise RuntimeError('Source backend configuration checksum changed')
    source = (args.source or run / 'source').resolve()
    source_hashes = {name: sha(source / name) for name in manifest['source_sha256']}
    if source_hashes != manifest['source_sha256']:
        raise RuntimeError('Frozen source differs from the source production run')
    sys.path.insert(0, str(source))
    import torch
    import train_cpt as trainer
    from cpt_orders import EpochOrders, order_indices, prefix_counts
    from packed_data import PackedTokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    if config['micro_batch'] != 16 or config['effective_batch'] != 16 or config['sequence_length'] != 2048:
        raise RuntimeError('This diagnostic expects the recorded 16 x 2048 production batch')
    if config['options']['kernels'] != 'native' or config['options']['loss'] != 'native':
        raise RuntimeError('Source fast configuration unexpectedly uses modified kernels/loss')
    data = Path(manifest['arguments']['data']).resolve()
    data_manifest = json.loads((data / 'manifest.json').read_text())
    if trainer.cpt.sha(data / 'manifest.json') != manifest['signature']['data_manifest_sha256']:
        raise RuntimeError('Prepared-data manifest changed')
    train = PackedTokens(data, sequence_length=config['sequence_length'], include_tail=True,
                         pad_token_id=tokenizer.eos_token_id)
    validation = PackedTokens(data, 'validation', sequence_length=config['sequence_length'])
    validation.explicit_mask = True
    orders = EpochOrders(checkpoint / 'orders', metadata['order_metadata'])
    start = metadata['completed_steps']
    stop = start + args.steps
    if not 0 <= start < stop <= metadata['schedule']['total_steps']:
        raise RuntimeError('Diagnostic steps exceed original global schedule')
    start_counts = prefix_counts(orders, start, config['effective_batch'], train.sequence_length, len(train.tokens))
    if start_counts != (metadata['trained_tokens'], metadata['supervised_tokens'], metadata['optimizer_updates']):
        raise RuntimeError('Saved counters do not match the shuffled data prefix')
    stop_counts = prefix_counts(orders, stop, config['effective_batch'], train.sequence_length, len(train.tokens))
    full_blocks = len(train.tokens) // train.sequence_length
    if args.probe_blocks > full_blocks:
        raise ValueError('Fixed train probe is larger than complete training blocks')
    # Dedicated RNG makes this split independent of CUDA/CPU training RNG.
    probe_indices = np.sort(np.random.default_rng(args.probe_seed).choice(
        full_blocks, size=args.probe_blocks, replace=False)).astype(np.int64)
    probe_sha = hashlib.sha256()
    for index in probe_indices:
        offset = int(index) * train.sequence_length
        probe_sha.update(memoryview(np.ascontiguousarray(train.tokens[offset:offset + train.sequence_length])).cast('B'))
    order_digest = hashlib.sha256()
    token_digest = hashlib.sha256()
    selected_blocks = 0
    tail_steps = []
    for step in range(start, stop):
        indices = np.asarray(order_indices(orders, step, config['effective_batch']), dtype=np.int64)
        order_digest.update(memoryview(indices).cast('B'))
        selected_blocks += len(indices)
        for index in indices:
            row = train[int(index)]
            for key in ('input_ids', 'labels'):
                token_digest.update(memoryview(row[key].numpy()).cast('B'))
        if len(train) - 1 in indices:
            tail_steps.append(step + 1)
    plan = {'schema_version': 1, 'kind': 'isolated controlled diagnostic branch', 'case': args.case,
            'source_run': str(run), 'checkpoint': str(checkpoint), 'source_directory': str(source),
            'source_sha256': source_hashes, 'diagnostic_script_sha256': sha(Path(__file__)),
            'checkpoint_signature': metadata['signature'], 'checkpoint_manifest_sha256': sha(checkpoint / 'checkpoint_manifest.json'),
            'checkpoint_files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
                                 for p in sorted(checkpoint.glob('*.safetensors'))},
            'checkpoint_training_state_file': {'bytes': (checkpoint / 'training_state.pt').stat().st_size,
                                               'sha256': sha(checkpoint / 'training_state.pt')},
            'started_utc': datetime.now(timezone.utc).isoformat(), 'data': str(data),
            'data_manifest_sha256': sha(data / 'manifest.json'),
            'data_artifact_sha256': {key: item['sha256'] for key, item in data_manifest['artifacts'].items()
                                     if key in ('train_tokens.npy', 'validation_tokens.npy')},
            'full_token_file_checksum_revalidation': False,
            'token_file_integrity_basis': 'Already verified production data, exact manifest SHA; checksum of selected training/probe data recorded separately',
            'start_step': start, 'end_step': stop, 'updates': args.steps,
            'schedule': metadata['schedule'], 'order_metadata': orders.metadata,
            'selected_block_indices_sha256': order_digest.hexdigest(),
            'selected_training_token_and_label_values_sha256': token_digest.hexdigest(), 'selected_blocks': selected_blocks,
            'partial_tail_global_steps': tail_steps, 'start_prefix_counts': list(start_counts),
            'end_prefix_counts': list(stop_counts), 'new_input_tokens': stop_counts[0] - start_counts[0],
            'new_supervised_tokens': stop_counts[1] - start_counts[1],
            'batch': {'effective': config['effective_batch'], 'micro': config['micro_batch'], 'sequence_length': train.sequence_length},
            'fixed_train_probe': {'seed': args.probe_seed, 'blocks': args.probe_blocks,
                                  'input_tokens': args.probe_blocks * train.sequence_length,
                                  'supervised_tokens': args.probe_blocks * (train.sequence_length - 1),
                                  'indices': probe_indices.tolist(), 'indices_sha256': array_sha(probe_indices),
                                  'tokens_sha256': probe_sha.hexdigest(), 'sampling': 'Fixed uniform complete-block sample from training corpus; not current training minibatch'},
            'validation': {'blocks': len(validation), 'input_tokens': len(validation) * validation.sequence_length,
                           'supervised_tokens': len(validation) * (validation.sequence_length - 1)},
            'evaluation_offsets': sorted(set([0, args.steps] + list(range(args.eval_every, args.steps, args.eval_every)))),
            'precision': 'BF16 autocast forward; backward outside autocast; FP32 master weights, gradients, Adam moments',
            'optimizer_math': {'betas': [.9, .95], 'eps': 1e-8, 'weight_decay': .1, 'max_grad_norm': 1.0},
            'permitted_implementation_differences': ['HF eager vs production whole-model compile',
                'standard foreach AdamW vs compiled capturable foreach AdamW',
                'standard stacked independent labels + all-ones 2D attention mask vs vectorized aliased full-block labels + resolved causal mask'],
            'unmodified': ['checkpoint weight values', 'Adam moments and initial steps', 'saved CPU/CUDA RNG',
                           'input/target token values', 'block order and global LR schedule', 'BF16/FP32 precision policy',
                           'full-parameter training', 'decay applied to all parameters including norm/bias'],
            'production_files_modified': False, 'nonfinite_policy': 'Record and fail before any optimizer update; never skip or silently fallback',
            'limitation': 'Production FlashAttention is nondeterministic; restored trajectories need not be bitwise identical.',
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy')}}
    return torch, trainer, config, metadata, tokenizer, train, validation, orders, probe_indices, plan


def finite_audit(model, optimizer, torch, expected_updates):
    rows = []
    for name, parameter in model.named_parameters():
        state = optimizer.state[parameter]
        row = {'name': name, 'parameter_finite': bool(torch.isfinite(parameter).all()),
               'parameter_dtype': str(parameter.dtype), 'gradient_dtype': str(parameter.grad.dtype) if parameter.grad is not None else None,
               'step': float(state['step'])}
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state[key]
            row[key + '_finite'] = bool(torch.isfinite(value).all())
            row[key + '_dtype'] = str(value.dtype)
        passed = (row['parameter_finite'] and row['parameter_dtype'] == 'torch.float32'
                  and row['gradient_dtype'] in (None, 'torch.float32') and row['step'] == expected_updates
                  and all(row[key + '_finite'] and row[key + '_dtype'] == 'torch.float32' for key in ('exp_avg', 'exp_avg_sq')))
        row['passed'] = passed
        rows.append(row)
    result = {'passed': all(row['passed'] for row in rows), 'expected_updates': expected_updates,
              'parameter_tensors': len(rows), 'rows': rows}
    if not result['passed']:
        raise RuntimeError('Nonfinite values, mixed precision, or wrong optimizer steps in diagnostic state')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('fast', 'native'), required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--probe-blocks', type=int, default=256)
    parser.add_argument('--probe-seed', type=int, default=20261009)
    parser.add_argument('--eval-micro-batch', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--log-every', type=int, default=100)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.probe_blocks, args.eval_micro_batch, args.log_every) < 1:
        parser.error('All lengths/intervals must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    last_step = None
    try:
        torch, trainer, config, metadata, tokenizer, train, validation, orders, probe_indices, plan = load_inputs(args)
        from cpt_orders import order_indices, prefix_counts
        from compiled_adamw import make_compiled_adamw, set_lr
        from model_backends import configure_model, eager_native
        from token_batches import PackedTokenBatches
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(8)
        write(args.output / 'plan.json', plan)
        shutil.copy2(Path(__file__), args.output / Path(__file__).name)
        emit({'event': 'plan', 'case': args.case, 'start_step': plan['start_step'], 'end_step': plan['end_step'],
              'tokens': plan['new_input_tokens'], 'probe_tokens': plan['fixed_train_probe']['input_tokens']})
        if args.plan_only:
            write(args.output / 'result.json', {'status': 'plan_only', 'case': args.case, 'plan': plan})
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 GPU required')
        saved = torch.load(args.checkpoint / 'training_state.pt', map_location='cpu', weights_only=True)
        if saved['signature'] != metadata['signature'] or saved['completed_steps'] != metadata['completed_steps']:
            raise RuntimeError('Checkpoint payload differs from its manifest')
        trainer.validate_resume(saved, metadata['signature'], train,
            argparse.Namespace(epochs=plan['order_metadata']['epochs'], seed=plan['order_metadata']['seed'],
                               steps=None, lr=plan['schedule']['peak_lr'], allow_source_migration=False), orders=orders)
        optimizer_audit = state_audit(saved['optimizer'], torch, saved['optimizer_updates'])
        rng_audit = {'cpu_sha256': tensor_sha(saved['cpu_rng']), 'cuda_sha256': tensor_sha(saved['cuda_rng'])}
        write(args.output / 'initial_optimizer_audit.json', optimizer_audit)
        write(args.output / 'initial_rng.json', rng_audit)
        # The native branch never invokes configure_model or replaces any HF module.
        if args.case == 'native':
            model = AutoModelForCausalLM.from_pretrained(str(args.checkpoint), local_files_only=True,
                torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda')
            model.config.use_cache = False
            forward = model
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(np.float32(plan['schedule']['peak_lr'])),
                betas=(.9, .95), weight_decay=.1, eps=1e-8, foreach=True, capturable=False, fused=False)
            adaptations = adapt_native_optimizer(saved['optimizer'], torch)
            adapted_audit = state_audit(saved['optimizer'], torch, saved['optimizer_updates'])
            if adapted_audit['combined_tensor_sha256'] != optimizer_audit['combined_tensor_sha256']:
                raise RuntimeError('Native implementation adaptation changed Adam moments or step values')
            write(args.output / 'optimizer_implementation_adaptation.json', {'changes': adaptations,
                  'tensor_values_unchanged': True, 'combined_tensor_sha256': adapted_audit['combined_tensor_sha256']})
            trainer.restore_optimizer(optimizer, saved['optimizer'])
            if any(group.get('capturable', False) or group.get('fused', False) for group in optimizer.param_groups):
                raise RuntimeError('Checkpoint group flags overwrote standard native optimizer settings')
            step_callable = optimizer.step
            train.explicit_mask = True
            provider = trainer.cpt.batch
            backend_metadata = {'model': 'unpatched HF AutoModelForCausalLM', 'compile': False,
                'loss': 'native HF causal language modeling loss', 'optimizer': 'standard torch.optim.AdamW foreach',
                'mask': 'standard 2D all-ones tensor', 'labels': 'independent clone, standard stacked batches'}
        else:
            model = trainer.cpt.load_model(argparse.Namespace(model=str(args.checkpoint), revision=None))
            forward, backend_metadata = configure_model(model, config['options'])
            optimizer, step_callable, optimizer_compile = make_compiled_adamw(model.parameters(), plan['schedule']['peak_lr'])
            trainer.restore_optimizer(optimizer, saved['optimizer'])
            backend_metadata['optimizer_compile'] = optimizer_compile
            train.explicit_mask = config['mask_policy'] == 'ones'
            provider = PackedTokenBatches()
        trainer.check_model_safety(model)
        trainer.check_optimizer_precision(model, optimizer)
        if args.case == 'native' and any(getattr(module, '_compiled_call_impl', None) is not None for module in model.modules()):
            raise RuntimeError('Native reference unexpectedly contains a compiled module')
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        if tensor_sha(torch.get_rng_state()) != rng_audit['cpu_sha256'] or tensor_sha(torch.cuda.get_rng_state()) != rng_audit['cuda_sha256']:
            raise RuntimeError('Training RNG restoration failed')
        initial_updates = saved['optimizer_updates']
        del saved, optimizer_audit
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        write(args.output / 'backend.json', backend_metadata)
        initial_finite = finite_audit(model, optimizer, torch, initial_updates)
        write(args.output / 'initial_finite_audit.json', initial_finite)
        eval_rows = []
        # Both branches evaluate through the original HF model with standard batches.
        validation.explicit_mask = True
        train_probe = type(train)(plan['data'], sequence_length=train.sequence_length)
        train_probe.explicit_mask = True

        def evaluate_at(offset):
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            was_training = model.training
            model.eval()
            torch.cuda.synchronize()
            eval_started = time.perf_counter()
            metrics = {}
            try:
                with torch.no_grad(), (eager_native(model) if args.case == 'fast' else nullcontext()):
                    for name, dataset, selected in (('validation', validation, range(len(validation))),
                                                    ('train_probe', train_probe, probe_indices)):
                        total = torch.zeros((), dtype=torch.float32, device='cuda')
                        count = inputs = 0
                        for start in range(0, len(selected), args.eval_micro_batch):
                            values, targets, tokens = trainer.cpt.batch(dataset, selected[start:start + args.eval_micro_batch])
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                loss = model(**values, use_cache=False).loss
                            total += loss * targets
                            count += targets
                            inputs += tokens
                        torch.cuda.synchronize()
                        value = float(total / count)
                        if not math.isfinite(value):
                            raise RuntimeError(f'Nonfinite native evaluation on {name}')
                        metrics[name] = {'loss': value, 'perplexity': math.exp(value), 'input_tokens': inputs,
                                         'supervised_tokens': count, 'blocks': len(selected), 'backend': 'unpatched HF native eager'}
                row = {'offset': offset, 'global_step': plan['start_step'] + offset,
                       'cumulative_input_tokens': prefix_counts(orders, plan['start_step'] + offset,
                             config['effective_batch'], train.sequence_length, len(train.tokens))[0],
                       'seconds': time.perf_counter() - eval_started, **metrics}
                eval_rows.append(row)
                write(args.output / 'evaluation_history.json', eval_rows)
                emit({'event': 'evaluation', 'case': args.case, **row})
            finally:
                model.train(was_training)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)

        evaluate_at(0)
        train_seconds = 0.0
        failures = []
        with (args.output / 'training.jsonl').open('x') as stream:
            for offset in range(1, args.steps + 1):
                global_step = plan['start_step'] + offset
                last_step = global_step
                indices = order_indices(orders, global_step - 1, config['effective_batch'])
                model.train()
                optimizer.zero_grad(set_to_none=True)
                rate = trainer.learning_rate(plan['schedule'], global_step - 1)
                effective_rate = float(np.float32(rate))
                if args.case == 'fast':
                    set_lr(optimizer, rate)
                else:
                    for group in optimizer.param_groups:
                        group['lr'] = effective_rate
                torch.cuda.synchronize()
                update_started = time.perf_counter()
                batches = [provider(train, indices[start:start + config['micro_batch']])
                           for start in range(0, len(indices), config['micro_batch'])]
                targets = sum(count for _, count, _ in batches)
                inputs = sum(count for _, _, count in batches)
                if targets == 0:
                    raise RuntimeError('Diagnostic unexpectedly selected an unsupervised-only update')
                total_loss = torch.zeros((), dtype=torch.float32, device='cuda')
                for values, count, _ in batches:
                    if args.case == 'fast' and config['mask_policy'] == 'causal':
                        values['attention_mask'] = {'full_attention': None}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = forward(**values, use_cache=False).loss
                    weighted = loss * (count / targets)
                    weighted.backward()
                    total_loss += weighted.detach()
                try:
                    norm = checked_step(model, optimizer, total_loss, step_callable, torch, global_step)
                except DiagnosticNonfinite as exc:
                    failures.append(exc.details)
                    write(args.output / 'nonfinite_failure.json', exc.details)
                    raise
                torch.cuda.synchronize()
                duration = time.perf_counter() - update_started
                train_seconds += duration
                if offset == 1:
                    first_update_inputs, first_update_seconds = inputs, duration
                counts = prefix_counts(orders, global_step, config['effective_batch'], train.sequence_length, len(train.tokens))
                record = {'offset': offset, 'step': global_step, 'loss': float(total_loss), 'grad_norm': float(norm),
                          'scheduled_lr': rate, 'effective_fp32_lr': effective_rate,
                          'input_tokens': inputs, 'supervised_tokens': targets, 'cumulative_tokens': counts[0],
                          'seconds': duration, 'optimizer_applied': True}
                stream.write(json.dumps(record) + '\n')
                if offset % args.log_every == 0 or offset == args.steps:
                    stream.flush()
                    write(args.output / 'progress.json', record)
                    emit({'event': 'training', 'case': args.case, **record})
                if offset in plan['evaluation_offsets']:
                    evaluate_at(offset)
        expected_updates = plan['end_prefix_counts'][2]
        final_finite = finite_audit(model, optimizer, torch, expected_updates)
        write(args.output / 'final_finite_audit.json', final_finite)
        checkpoint = args.output / 'final_checkpoint'
        temporary = args.output / 'final_checkpoint.incomplete'
        temporary.mkdir()
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        orders.copy_to(temporary / 'orders')
        torch.save({'diagnostic_schema_version': 1, 'kind': 'controlled backend comparison branch',
                    'source_checkpoint': str(args.checkpoint.resolve()), 'source_signature': metadata['signature'],
                    'case': args.case, 'completed_steps': plan['end_step'], 'optimizer_updates': expected_updates,
                    'trained_tokens': plan['end_prefix_counts'][0], 'supervised_tokens': plan['end_prefix_counts'][1],
                    'schedule': plan['schedule'], 'order_metadata': orders.metadata,
                    'optimizer': optimizer.state_dict(), 'cpu_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(), 'comparison_plan_sha256': sha(args.output / 'plan.json')},
                   temporary / 'diagnostic_state.pt')
        write(temporary / 'diagnostic_checkpoint_manifest.json', {'schema_version': 1,
              'kind': 'diagnostic only; not a production resume checkpoint', 'case': args.case,
              'completed_steps': plan['end_step'], 'optimizer_updates': expected_updates,
              'trained_tokens': plan['end_prefix_counts'][0], 'supervised_tokens': plan['end_prefix_counts'][1],
              'source_signature': metadata['signature'], 'comparison_plan_sha256': sha(args.output / 'plan.json'),
              'saved_utc': datetime.now(timezone.utc).isoformat(), 'all_parameters_and_optimizer_states_finite': True,
              'full_tensor_round_trip_verification': 'Not performed; finite values and counts checked before save'})
        temporary.rename(checkpoint)
        write(args.output / 'backend.json', backend_metadata)
        result = {'status': 'complete', 'case': args.case, 'start_step': plan['start_step'], 'end_step': plan['end_step'],
                  'optimizer_updates_applied': args.steps, 'new_input_tokens': plan['new_input_tokens'],
                  'training_token_and_label_values_sha256': plan['selected_training_token_and_label_values_sha256'],
                  'training_seconds_including_compile': train_seconds,
                  'tokens_per_second_including_compile': plan['new_input_tokens'] / train_seconds,
                  'total_seconds_including_load_eval_save': time.perf_counter() - started,
                  'evaluation_history': eval_rows, 'final_checkpoint': str(checkpoint.resolve()),
                  'initial_rng': rng_audit, 'failures': failures,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'steady_tokens_per_second': (plan['new_input_tokens'] - first_update_inputs) / (train_seconds - first_update_seconds) if args.steps > 1 else None,
                  'timing_note': 'Per-update time includes data/H2D, loss finite synchronization, native backward, clipping, AdamW, GPU synchronization; excludes evaluation, JSON logging, loading/saving. Steady rate excludes the first update in BOTH cases; no extra warmup updates.',
                  'production_files_modified': False}
        write(args.output / 'result.json', result)
        emit({'event': 'complete', 'case': args.case, 'end_step': plan['end_step'],
              'tokens_per_second': result['tokens_per_second_including_compile']})
    except Exception as exc:
        write(args.output / 'result.json', {'status': 'failed', 'case': args.case, 'last_attempted_step': last_step,
              'error': str(exc), 'traceback': traceback.format_exc(), 'production_files_modified': False,
              'elapsed_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    main()
