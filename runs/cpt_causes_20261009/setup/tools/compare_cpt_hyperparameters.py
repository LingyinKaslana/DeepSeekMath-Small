"""Native-only, single-knob diagnostic continuations of one immutable checkpoint."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import compare_cpt_backends as common

CASES = {'baseline': (1.0, .1), 'lr_0p3': (.3, .1), 'wd_zero': (1.0, 0.0)}


def effective_lr(schedule, zero_based_step, case, learning_rate):
    scheduled = learning_rate(schedule, zero_based_step)
    multiplier = CASES[case][0]
    return scheduled, scheduled * multiplier, float(np.float32(scheduled * multiplier))


def optimizer_tensor_identity(saved):
    return {(index, key): (id(value), value.data_ptr(), value._version,
                            tuple(value.shape), str(value.dtype), str(value.device))
            for index, state in saved['state'].items() for key, value in state.items()}


def apply_case(saved, case):
    """Alter only group WD; LR is set per update, never mutate Adam tensors."""
    before_tensors = optimizer_tensor_identity(saved)
    changes = []
    for group in saved['param_groups']:
        if (tuple(group['betas']) != (.9, .95) or group['eps'] != 1e-8
                or group['weight_decay'] != .1 or group.get('maximize', False)
                or group.get('capturable', False) or group.get('fused', False)
                or group.get('foreach') is not True):
            raise RuntimeError('Expected unchanged source native foreach AdamW math/flags')
        before = {key: value for key, value in group.items() if key != 'params'}
        group['weight_decay'] = CASES[case][1]
        after = {key: value for key, value in group.items() if key != 'params'}
        changed = [key for key in before if before[key] != after[key]]
        if changed != (['weight_decay'] if case == 'wd_zero' else []):
            raise RuntimeError('Intervention altered an unexpected optimizer group field')
        changes.append({'changed_fields': changed, 'before': before, 'after': after})
    if before_tensors != optimizer_tensor_identity(saved):
        raise RuntimeError('Intervention changed Adam tensor identity/storage/version')
    return {'groups': changes, 'adam_tensor_storage_identity_and_versions_unchanged': True,
            'tensor_values_unchanged_basis': 'Only weight_decay group assignment; all state tensor objects, storage pointers and versions preserved'}


def validate_probes(raw, orders, start, stop, batch, full_blocks):
    groups = raw.get('groups', raw)
    aliases = {'uniform_train': ('uniform_train', 'train_probe', 'uniform_probe'),
               'seen_train': ('seen_train',), 'unseen_train': ('unseen_train',)}
    selected = {}
    for canonical, names in aliases.items():
        found = next((groups[name] for name in names if name in groups), None)
        if found is None:
            raise ValueError('Missing probe group ' + canonical)
        values = found['indices'] if isinstance(found, dict) else found
        indices = np.asarray(values)
        if indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in 'iu':
            raise ValueError('Probe indices must be nonempty integer vectors')
        indices = indices.astype(np.int64)
        if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= full_blocks:
            raise ValueError('Probe must contain unique complete training blocks')
        selected[canonical] = indices
    if start // orders.epoch_steps or stop > orders.epoch_steps:
        raise ValueError('Seen/unseen audit currently requires a first-epoch interval')
    exclusion_stop = raw.get('unseen_exclusion_through_step', stop)
    if (isinstance(exclusion_stop, bool) or not isinstance(exclusion_stop, int)
            or not stop <= exclusion_stop <= orders.epoch_steps):
        raise ValueError('Unseen exclusion boundary must include the full intervention interval')
    prefix_start = orders.orders[0][:start * batch]
    prefix_stop = orders.orders[0][:exclusion_stop * batch]
    if not np.isin(selected['seen_train'], prefix_start).all():
        raise ValueError('Seen probe includes blocks not consumed at fork checkpoint')
    if np.isin(selected['unseen_train'], prefix_stop).any():
        raise ValueError('Unseen probe overlaps a block consumed through the declared exclusion boundary')
    if np.intersect1d(selected['seen_train'], selected['unseen_train']).size:
        raise ValueError('Seen/unseen probes overlap')
    for key, expected in [('start_step', start), ('end_step', stop)]:
        if key in raw and raw[key] != expected:
            raise ValueError('Probe JSON interval differs: ' + key)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--probe-blocks', type=int, default=256)
    parser.add_argument('--probe-seed', type=int, default=20261009)
    parser.add_argument('--probes-json', type=Path, required=True)
    parser.add_argument('--eval-micro-batch', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--expected-initial-validation', type=float, default=1.8604923486709595)
    parser.add_argument('--initial-validation-tolerance', type=float, default=1e-6)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--no-save-checkpoint', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.probe_blocks, args.eval_micro_batch, args.log_every) < 1:
        parser.error('Lengths and intervals must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    last_step = None
    try:
        torch, trainer, config, metadata, tokenizer, train, validation, orders, _, plan = common.load_inputs(args)
        from cpt_orders import order_indices, prefix_counts
        from standard_native import update_native
        from transformers import AutoModelForCausalLM
        if config.get('execution_mode') != 'standard_native':
            raise RuntimeError('Hyperparameter diagnostic requires an original native checkpoint')
        raw = json.loads(args.probes_json.read_text())
        probes = validate_probes(raw, orders, plan['start_step'], plan['end_step'],
                                 config['effective_batch'], len(train.tokens) // train.sequence_length)
        if not np.array_equal(probes['uniform_train'], np.asarray(plan['fixed_train_probe']['indices'])):
            raise ValueError('Uniform probe differs from original fixed probe seed/sample')
        probe_metadata = {}
        for name, indices in probes.items():
            digest = hashlib.sha256()
            for index in indices:
                start = int(index) * train.sequence_length
                digest.update(memoryview(np.ascontiguousarray(train.tokens[start:start + train.sequence_length])).cast('B'))
            probe_metadata[name] = {'indices': indices.tolist(), 'indices_sha256': common.array_sha(indices),
                                    'tokens_sha256': digest.hexdigest(), 'blocks': len(indices)}
        plan.update(kind='native single-knob hyperparameter diagnostic', diagnostic_only=True,
                    automatic_production_resume=False, probes=probe_metadata,
                    probes_json_sha256=common.sha(args.probes_json),
                    unseen_exclusion_through_step=raw.get('unseen_exclusion_through_step', plan['end_step']),
                    diagnostic_script_sha256=common.sha(Path(__file__)),
                    helper_sha256=common.sha(Path(common.__file__)),
                    lr_multiplier=CASES[args.case][0], weight_decay=CASES[args.case][1],
                    only_knob_changes=[] if args.case == 'baseline' else
                        ([{'knob': 'lr_multiplier', 'from': 1.0, 'to': .3}] if args.case == 'lr_0p3' else
                         [{'knob': 'weight_decay', 'from': .1, 'to': 0.0}]),
                    expected_initial_validation=args.expected_initial_validation,
                    expected_initial_validation_tolerance=args.initial_validation_tolerance,
                    save_diagnostic_checkpoint=not args.no_save_checkpoint,
                    interpretation_limitations=['LR scaling changes both Adam gradient-update scale and LR-times-WD decay; the WD-zero arm helps distinguish these contributions',
                        'A 1000-step local intervention does not establish the cause of the initial 5000-step turning point',
                        'Diagnostic token counts are separate from production progress'],
                    permitted_implementation_differences=[],
                    unmodified=['starting model weights', 'Adam moments and counters', 'CPU/CUDA RNG',
                                'global schedule progress', 'batch/order/input/targets', 'BF16/FP32', 'native backend', 'clip=1'])
        plan['optimizer_math']['weight_decay'] = CASES[args.case][1]
        common.write(args.output / 'plan.json', plan)
        shutil.copy2(Path(__file__), args.output / Path(__file__).name)
        common.write(args.output / 'probes.json', raw)
        common.emit({'event': 'plan', 'case': args.case, 'start_step': plan['start_step'], 'end_step': plan['end_step']})
        if args.plan_only:
            common.write(args.output / 'result.json', {'status': 'plan_only', 'diagnostic_only': True, 'case': args.case, 'plan': plan})
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 GPU required')
        saved = torch.load(args.checkpoint / 'training_state.pt', map_location='cpu', weights_only=True, mmap=True)
        if saved['signature'] != metadata['signature'] or saved['completed_steps'] != plan['start_step']:
            raise RuntimeError('Checkpoint payload differs from source manifest')
        trainer.validate_resume(saved, metadata['signature'], train,
            argparse.Namespace(epochs=plan['order_metadata']['epochs'], seed=plan['order_metadata']['seed'],
                               steps=None, lr=plan['schedule']['peak_lr'], allow_source_migration=False), orders=orders)
        initial_audit = common.state_audit(saved['optimizer'], torch, saved['optimizer_updates'])
        common.write(args.output / 'initial_optimizer_audit.json', initial_audit)
        common.write(args.output / 'optimizer_intervention.json', apply_case(saved['optimizer'], args.case))
        rng = {'cpu_sha256': common.tensor_sha(saved['cpu_rng']), 'cuda_sha256': common.tensor_sha(saved['cuda_rng'])}
        common.write(args.output / 'initial_rng.json', rng)
        model = AutoModelForCausalLM.from_pretrained(str(args.checkpoint), local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda')
        model.config.use_cache = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(np.float32(plan['schedule']['peak_lr'])),
            betas=(.9, .95), weight_decay=CASES[args.case][1], eps=1e-8, foreach=True, capturable=False, fused=False)
        trainer.restore_optimizer(optimizer, saved['optimizer'])
        if any(group['weight_decay'] != CASES[args.case][1] for group in optimizer.param_groups):
            raise RuntimeError('Optimizer load lost intervention WD')
        trainer.check_model_safety(model)
        trainer.check_optimizer_precision(model, optimizer)
        if any(getattr(module, '_compiled_call_impl', None) is not None for module in model.modules()):
            raise RuntimeError('Native diagnostic unexpectedly compiled')
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        if common.tensor_sha(torch.get_rng_state()) != rng['cpu_sha256'] or common.tensor_sha(torch.cuda.get_rng_state()) != rng['cuda_sha256']:
            raise RuntimeError('RNG restoration failed')
        common.write(args.output / 'initial_finite_audit.json', common.finite_audit(model, optimizer, torch, saved['optimizer_updates']))
        del saved, initial_audit
        gc.collect()
        train.explicit_mask = validation.explicit_mask = True
        torch.cuda.reset_peak_memory_stats()
        evaluations = []

        def evaluate_at(offset):
            row = {'offset': offset, 'global_step': plan['start_step'] + offset,
                   'validation': trainer.evaluate(model, validation, trainer.cpt.batch,
                       micro_batch=args.eval_micro_batch, native_standard=True)}
            for name, indices in probes.items():
                row[name] = trainer.evaluate(model, train, trainer.cpt.batch, micro_batch=args.eval_micro_batch,
                                              indices=indices, native_standard=True)
            if offset == 0 and abs(row['validation']['loss'] - args.expected_initial_validation) > args.initial_validation_tolerance:
                raise RuntimeError('Native initial validation does not match expected source checkpoint')
            evaluations.append(row)
            common.write(args.output / 'evaluation_history.json', evaluations)
            common.emit({'event': 'evaluation', 'case': args.case, **row})

        evaluate_at(0)
        training_seconds = 0.0
        with (args.output / 'training.jsonl').open('x') as stream:
            for offset in range(1, args.steps + 1):
                last_step = plan['start_step'] + offset
                scheduled, intervention, actual = effective_lr(plan['schedule'], last_step - 1, args.case, trainer.learning_rate)
                for group in optimizer.param_groups:
                    group['lr'] = actual
                indices = order_indices(orders, last_step - 1, config['effective_batch'])
                torch.cuda.synchronize()
                began = time.perf_counter()
                loss, norm, inputs, targets, applied = update_native(model, optimizer, train,
                    trainer.cpt.batch, indices, config['micro_batch'])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - began
                training_seconds += seconds
                if not applied:
                    raise RuntimeError('Diagnostic update unexpectedly not applied')
                row = {'offset': offset, 'step': last_step, 'loss': float(loss), 'grad_norm': float(norm),
                       'scheduled_lr': scheduled, 'lr_multiplier': CASES[args.case][0],
                       'intervention_lr': intervention, 'effective_fp32_lr': actual,
                       'weight_decay': CASES[args.case][1], 'input_tokens': inputs, 'supervised_tokens': targets,
                       'optimizer_applied': applied, 'seconds': seconds}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                if offset % args.log_every == 0 or offset == args.steps:
                    stream.flush()
                    common.write(args.output / 'progress.json', row)
                    common.emit({'event': 'training', 'case': args.case, **row})
                if offset in plan['evaluation_offsets']:
                    evaluate_at(offset)
        expected_updates = plan['end_prefix_counts'][2]
        common.write(args.output / 'final_finite_audit.json', common.finite_audit(model, optimizer, torch, expected_updates))
        checkpoint = None
        if not args.no_save_checkpoint:
            checkpoint = args.output / 'final_checkpoint'
            temporary = args.output / 'final_checkpoint.incomplete'
            temporary.mkdir()
            model.save_pretrained(temporary, safe_serialization=True)
            tokenizer.save_pretrained(temporary)
            orders.copy_to(temporary / 'orders')
            final_state = {'diagnostic_schema_version': 1, 'diagnostic_only': True,
                'kind': 'native single-knob hyperparameter diagnostic; not production-resumable',
                'source_checkpoint': str(args.checkpoint.resolve()), 'source_signature': metadata['signature'],
                'case': args.case, 'completed_steps': plan['end_step'], 'optimizer_updates': expected_updates,
                'trained_tokens': plan['end_prefix_counts'][0], 'supervised_tokens': plan['end_prefix_counts'][1],
                'schedule': plan['schedule'], 'lr_multiplier': CASES[args.case][0], 'weight_decay': CASES[args.case][1],
                'order_metadata': orders.metadata, 'optimizer': optimizer.state_dict(),
                'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                'comparison_plan_sha256': common.sha(args.output / 'plan.json')}
            torch.save(final_state, temporary / 'diagnostic_state.pt')
            common.write(temporary / 'diagnostic_checkpoint_manifest.json', {
                key: value for key, value in final_state.items() if key not in ('optimizer', 'cpu_rng', 'cuda_rng')})
            temporary.rename(checkpoint)
        result = {'status': 'complete', 'diagnostic_only': True, 'automatic_production_resume': False,
            'case': args.case, 'start_step': plan['start_step'], 'end_step': plan['end_step'],
            'optimizer_updates_applied': args.steps, 'new_input_tokens': plan['new_input_tokens'],
            'only_knob_changes': plan['only_knob_changes'], 'initial_rng': rng,
            'selected_block_indices_sha256': plan['selected_block_indices_sha256'],
            'training_token_and_label_values_sha256': plan['selected_training_token_and_label_values_sha256'],
            'training_seconds': training_seconds, 'tokens_per_second': plan['new_input_tokens'] / training_seconds,
            'total_seconds': time.perf_counter() - started, 'evaluation_history': evaluations,
            'final_checkpoint': str(checkpoint.resolve()) if checkpoint else None,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'production_files_modified': False}
        common.write(args.output / 'result.json', result)
        common.emit({'event': 'complete', 'case': args.case, 'end_step': plan['end_step']})
    except Exception as exc:
        common.write(args.output / 'result.json', {'status': 'failed', 'diagnostic_only': True,
            'case': args.case, 'last_attempted_step': last_step, 'error': str(exc),
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.perf_counter() - started,
            'production_files_modified': False})
        raise


if __name__ == '__main__':
    main()
