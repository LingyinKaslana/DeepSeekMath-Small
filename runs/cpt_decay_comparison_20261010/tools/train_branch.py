"""Two LR schedules from one immutable CPT checkpoint; diagnostic outputs only."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import compare_cpt_backends as common
from compare_cpt_hyperparameters import validate_probes

DIAG = Path(__file__).resolve().parents[1]

def branch_lr(case, offset, steps, initial, final):
    if not 1 <= offset <= steps or steps < 2:
        raise ValueError('LR offset outside fixed branch budget')
    if case == 'constant':
        return initial
    if case == 'linear_decay':
        return initial + (final-initial) * ((offset-1)/(steps-1))
    raise ValueError('Unknown branch')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('constant','linear_decay'), required=True)
    parser.add_argument('--plan-only', action='store_true')
    given = parser.parse_args()
    experiment = json.loads((DIAG/'plan.json').read_text())
    args = argparse.Namespace(run=Path(experiment['production_run']),
        checkpoint=Path(experiment['checkpoint']), source=Path(experiment['source']),
        output=DIAG/(('preflight_' if given.plan_only else 'branch_')+given.case), case=given.case, steps=experiment['steps'],
        eval_every=1000, probe_blocks=256, probe_seed=20261009,
        eval_micro_batch=experiment['evaluation_micro_batch'], log_every=100)
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    last_step = None
    try:
        torch, trainer, config, metadata, tokenizer, train, validation, orders, uniform, plan = common.load_inputs(args)
        from cpt_orders import order_indices, prefix_counts
        from compiled_adamw import make_compiled_adamw, set_lr
        from model_backends import configure_model
        from packed_data import PackedTokens
        from token_batches import PackedTokenBatches
        if config.get('execution_mode') == 'standard_native' or not config.get('compiled_optimizer'):
            raise RuntimeError('Expected original optimized model and compiled AdamW')
        if common.sha(args.checkpoint/'checkpoint_manifest.json') != experiment['checkpoint_manifest_sha256']:
            raise RuntimeError('Frozen checkpoint manifest changed')
        if plan['start_step'] != experiment['start_step'] or plan['end_step'] != experiment['end_step']:
            raise RuntimeError('Fork/budget differs from experiment')
        if common.sha(DIAG/'probes.json') != experiment['probes_sha256']:
            raise RuntimeError('Frozen probe file changed')
        raw = json.loads((DIAG/'probes.json').read_text())
        probes = validate_probes(raw, orders, plan['start_step'], plan['end_step'],
            config['effective_batch'], len(train.tokens)//train.sequence_length)
        if not np.array_equal(probes['uniform_train'], uniform):
            raise RuntimeError('Production uniform train probe changed')
        checked = {}
        for name, indices in probes.items():
            digest = hashlib.sha256()
            for index in indices:
                begin = int(index)*train.sequence_length
                digest.update(memoryview(np.ascontiguousarray(train.tokens[begin:begin+train.sequence_length])).cast('B'))
            checked[name] = dict(indices_sha256=common.array_sha(indices), tokens_sha256=digest.hexdigest())
            for key,value in checked[name].items():
                if value != raw['groups'][name][key]:
                    raise RuntimeError('Probe content changed: '+name+'/'+key)
        options = copy.deepcopy(config['options'])
        options['compile_shape'] = [experiment['diagnostic_micro_batch'],train.sequence_length]
        plan.update(kind='controlled LR schedule comparison', diagnostic_only=True,
            case=args.case, evaluation_offsets=experiment['evaluation_offsets'],
            probes=raw['groups'], probes_json_sha256=common.sha(DIAG/'probes.json'),
            checked_probe_hashes=checked, initial_lr=experiment['initial_lr'], final_decay_lr=experiment['final_decay_lr'],
            actual_training_micro_batch=experiment['diagnostic_micro_batch'], actual_backend_options=options,
            expected_initial_validation=experiment['expected_initial_validation'],
            shared_implementation_difference=experiment['shared_implementation_difference'],
            permitted_implementation_differences=['Both arms micro8 accumulation2 and compile_shape8x2048; production micro16'],
            unmodified=experiment['unmodified'],
            only_knob_changes=[] if args.case=='constant' else ['linear LR decay from3e-6 to3e-7 over6104updates'],
            automatic_production_resume=False, production_files_modified=False,
            nonfinite_policy=experiment['finite_policy'],
            diagnostic_script_sha256=common.sha(__file__),
            common_helper_sha256=common.sha(common.__file__),
            interpretation_limitations=experiment['limits'])
        plan['batch']['micro'] = experiment['diagnostic_micro_batch']
        common.write(args.output/'plan.json',plan)
        common.write(args.output/'probes.json',raw)
        common.emit(dict(event='plan',case=args.case,start_step=plan['start_step'],end_step=plan['end_step'],tokens=plan['new_input_tokens']))
        if given.plan_only:
            common.write(args.output/'result.json',dict(status='plan_only',plan=plan))
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 required')
        torch.cuda.set_per_process_memory_fraction(experiment['allocator_memory_fraction'])
        free,total = torch.cuda.mem_get_info()
        if free < total*.42:
            raise RuntimeError('Insufficient shared GPU headroom for bounded micro8 diagnostic')
        saved = torch.load(args.checkpoint/'training_state.pt',map_location='cpu',weights_only=True,mmap=True)
        if saved['signature'] != metadata['signature'] or saved['completed_steps'] != plan['start_step']:
            raise RuntimeError('Checkpoint state differs from immutable manifest')
        trainer.validate_resume(saved,metadata['signature'],train,
            argparse.Namespace(epochs=orders.epochs,seed=orders.metadata['seed'],steps=None,
                lr=plan['schedule']['peak_lr'],allow_source_migration=False),orders=orders)
        initial_audit = common.state_audit(saved['optimizer'],torch,saved['optimizer_updates'])
        common.write(args.output/'initial_optimizer_audit.json',initial_audit)
        initial_rng = dict(cpu_sha256=common.tensor_sha(saved['cpu_rng']),cuda_sha256=common.tensor_sha(saved['cuda_rng']))
        common.write(args.output/'initial_rng.json',initial_rng)
        model = trainer.cpt.load_model(argparse.Namespace(model=str(args.checkpoint),revision=None))
        forward,backend = configure_model(model,options)
        optimizer,step_callable,optimizer_compile = make_compiled_adamw(model.parameters(),experiment['initial_lr'])
        trainer.restore_optimizer(optimizer,saved['optimizer'])
        for group in optimizer.param_groups:
            if (tuple(group['betas']) != (.9,.95) or group['weight_decay'] != .1 or group['eps'] != 1e-8
                or not group.get('capturable') or group.get('foreach') is not True or group.get('fused')
                or not isinstance(group['lr'],torch.Tensor) or group['lr'].device.type != 'cuda'):
                raise RuntimeError('Original compiled AdamW math or flags changed')
        trainer.check_model_safety(model)
        trainer.check_optimizer_precision(model,optimizer)
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        if common.tensor_sha(torch.get_rng_state()) != initial_rng['cpu_sha256'] or common.tensor_sha(torch.cuda.get_rng_state()) != initial_rng['cuda_sha256']:
            raise RuntimeError('RNG restoration failed')
        common.write(args.output/'initial_finite_audit.json',common.finite_audit(model,optimizer,torch,saved['optimizer_updates']))
        del saved
        gc.collect()
        backend['optimizer_compile'] = optimizer_compile
        common.write(args.output/'backend.json',backend)
        provider = PackedTokenBatches()
        train.explicit_mask = config['mask_policy']=='ones'
        validation.explicit_mask = True
        train_eval = PackedTokens(Path(plan['data']),sequence_length=train.sequence_length,
            include_tail=True,pad_token_id=tokenizer.eos_token_id)
        train_eval.explicit_mask = True
        evaluations = []

        def evaluate_at(offset):
            before = dict(cpu_sha256=common.tensor_sha(torch.get_rng_state()),cuda_sha256=common.tensor_sha(torch.cuda.get_rng_state()))
            row = dict(offset=offset,global_step=plan['start_step']+offset)
            row['validation'] = trainer.evaluate(model,validation,trainer.cpt.batch,micro_batch=args.eval_micro_batch)
            for name,indices in probes.items():
                row[name] = trainer.evaluate(model,train_eval,trainer.cpt.batch,micro_batch=args.eval_micro_batch,indices=indices)
            after = dict(cpu_sha256=common.tensor_sha(torch.get_rng_state()),cuda_sha256=common.tensor_sha(torch.cuda.get_rng_state()))
            if before != after:
                raise RuntimeError('Evaluation consumed training RNG')
            if offset==0 and abs(row['validation']['loss']-experiment['expected_initial_validation'])>experiment['initial_validation_tolerance']:
                raise RuntimeError('Initial native validation differs from source checkpoint')
            if offset==0 and before != initial_rng:
                raise RuntimeError('Initial evaluation began with altered checkpoint RNG')
            evaluations.append(row)
            common.write(args.output/'evaluation_history.json',evaluations)
            common.emit(dict(event='evaluation',case=args.case,**row))

        torch.cuda.reset_peak_memory_stats()
        evaluate_at(0)
        training_seconds = 0.0
        with (args.output/'training.jsonl').open('x') as stream:
            for offset in range(1,args.steps+1):
                last_step = plan['start_step']+offset
                rate = branch_lr(args.case,offset,args.steps,experiment['initial_lr'],experiment['final_decay_lr'])
                set_lr(optimizer,rate)
                actual = float(optimizer.param_groups[0]['lr'])
                if actual != float(np.float32(rate)):
                    raise RuntimeError('Effective CUDA LR differs from planned FP32 value')
                indices = order_indices(orders,last_step-1,config['effective_batch'])
                torch.cuda.synchronize()
                began = time.perf_counter()
                loss,norm,inputs,targets,applied = trainer.update(model,forward,optimizer,train,provider,indices,
                    experiment['diagnostic_micro_batch'],config['mask_policy'],step_callable=step_callable,
                    retry_native_on_nonfinite=False,step=last_step)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-began
                if not applied or inputs != 32768 or targets != 32752 or not math.isfinite(float(loss)) or not math.isfinite(float(norm)):
                    raise RuntimeError('Invalid/nonfinite/skipped diagnostic update')
                training_seconds += seconds
                row = dict(offset=offset,step=last_step,loss=float(loss),grad_norm=float(norm),lr=rate,
                    effective_fp32_lr=actual,input_tokens=inputs,supervised_tokens=targets,
                    cumulative_tokens=prefix_counts(orders,last_step,16,train.sequence_length,len(train.tokens))[0],
                    seconds=seconds,optimizer_applied=applied)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if offset==1 or offset%args.log_every==0 or offset==args.steps:
                    stream.flush()
                    common.write(args.output/'progress.json',row)
                    common.emit(dict(event='training',case=args.case,**row))
                if offset in experiment['evaluation_offsets']:
                    evaluate_at(offset)
        common.write(args.output/'final_finite_audit.json',common.finite_audit(model,optimizer,torch,plan['end_step']))
        temporary = args.output/'final_checkpoint.incomplete'
        temporary.mkdir()
        model.save_pretrained(temporary,safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        orders.copy_to(temporary/'orders')
        state = dict(diagnostic_schema_version=1,diagnostic_only=True,kind='LR comparison, not production-resumable',
            source_checkpoint=str(args.checkpoint),source_signature=metadata['signature'],case=args.case,
            completed_steps=plan['end_step'],optimizer_updates=plan['end_step'],
            trained_tokens=plan['end_prefix_counts'][0],supervised_tokens=plan['end_prefix_counts'][1],
            optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            lr_policy=dict(case=args.case,initial=experiment['initial_lr'],final=experiment['final_decay_lr'],steps=args.steps),
            order_metadata=orders.metadata,comparison_plan_sha256=common.sha(args.output/'plan.json'))
        torch.save(state,temporary/'diagnostic_state.pt')
        common.write(temporary/'diagnostic_checkpoint_manifest.json',{key:value for key,value in state.items() if key not in ('optimizer','cpu_rng','cuda_rng')})
        checkpoint = args.output/'final_checkpoint'
        temporary.rename(checkpoint)
        result = dict(status='complete',case=args.case,diagnostic_only=True,start_step=plan['start_step'],end_step=plan['end_step'],
            optimizer_updates_applied=args.steps,new_input_tokens=plan['new_input_tokens'],
            new_supervised_tokens=plan['new_supervised_tokens'],initial_rng=initial_rng,
            initial_optimizer_tensor_sha256=initial_audit['combined_tensor_sha256'],
            checkpoint_files=plan['checkpoint_files'],checkpoint_training_state_file=plan['checkpoint_training_state_file'],
            checkpoint_manifest_sha256=plan['checkpoint_manifest_sha256'],
            selected_block_indices_sha256=plan['selected_block_indices_sha256'],
            training_token_and_label_values_sha256=plan['selected_training_token_and_label_values_sha256'],
            probes_sha256=common.sha(args.output/'probes.json'),checked_probe_hashes=checked,
            evaluation_history=evaluations,training_seconds=training_seconds,tokens_per_second=plan['new_input_tokens']/training_seconds,
            total_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),final_checkpoint=str(checkpoint),
            completed_utc=datetime.now(timezone.utc).isoformat(),production_files_modified=False)
        common.write(args.output/'result.json',result)
        common.emit(dict(event='complete',case=args.case,end_step=plan['end_step']))
    except BaseException as error:
        common.write(args.output/'result.json',dict(status='failed',case=args.case,last_attempted_step=last_step,
            error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started,production_files_modified=False))
        raise

if __name__ == '__main__':
    main()
