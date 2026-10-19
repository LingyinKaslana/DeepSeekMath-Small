"""Unpatched HF/PyTorch CPT execution and observational fixed train probes."""
from contextlib import contextmanager
import hashlib
import json
import math
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def check_native_config(config):
    if (config.get('execution_mode') != 'standard_native' or config['mask_policy'] != 'ones'
            or config.get('compiled_optimizer', False) or config['fused_optimizer']
            or config['options'].get('kernels', 'native') != 'native'
            or config['options'].get('loss', 'native') != 'native'
            or config['options'].get('compile_scope', 'none') != 'none'
            or config['options'].get('preserve_silu', False)):
        raise ValueError('standard_native requires native HF kernels/loss, no compile/fused optimizer, and all-ones masks')


def load_native_model(args):
    begin = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda')
    model.config.use_cache = False
    print(json.dumps({'event': 'model_loaded', 'execution_mode': 'standard_native',
                      'model': args.model, 'revision': args.revision,
                      'seconds': time.perf_counter() - begin,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
                      'model_type': model.config.model_type,
                      'attention': model.config._attn_implementation}), flush=True)
    return model


def make_native_optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=float(np.float32(lr)), betas=(.9, .95),
                             weight_decay=.1, eps=1e-8, foreach=True, capturable=False, fused=False)


def set_native_lr(optimizer, rate):
    value = float(np.float32(rate))
    if not math.isfinite(value) or value < 0:
        raise ValueError('Native learning rate must be finite and nonnegative')
    for group in optimizer.param_groups:
        group['lr'] = value


def _moment_digest(saved):
    digest = hashlib.sha256()
    for index, state in sorted(saved['state'].items()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or value.device.type != 'cpu' or not bool(torch.isfinite(value).all())):
                raise ValueError('Native restoration requires finite CPU FP32 Adam state')
            digest.update(str((index, key, tuple(value.shape), str(value.dtype))).encode())
            digest.update(memoryview(value.contiguous().numpy()).cast('B'))
    return digest.hexdigest()


def adapt_native_optimizer(saved):
    """Adapt execution flags only; CPU SHA proves all Adam tensor values unchanged."""
    before = _moment_digest(saved)
    changes = []
    for group in saved['param_groups']:
        if (tuple(group.get('betas', ())) != (.9, .95) or group.get('weight_decay') != .1
                or group.get('eps') != 1e-8 or group.get('amsgrad', False)
                or group.get('maximize', False) or group.get('differentiable', False)):
            raise ValueError('Cannot migrate a different AdamW math protocol')
        prior = {'lr': float(group['lr']), 'lr_type': type(group['lr']).__name__,
                 'capturable': group.get('capturable', False), 'foreach': group.get('foreach'),
                 'fused': group.get('fused')}
        group['lr'] = float(group['lr'])
        group.update(capturable=False, foreach=True, fused=False)
        changes.append({'before': prior, 'after': {'lr': group['lr'], 'lr_type': 'float',
                        'capturable': False, 'foreach': True, 'fused': False}})
    after = _moment_digest(saved)
    if before != after:
        raise RuntimeError('Implementation adaptation altered saved Adam tensors')
    return {'tensor_values_unchanged': True, 'before_tensor_sha256': before,
            'after_tensor_sha256': after, 'groups': changes,
            'step_device': 'CPU scalar; checkpoint CPU mapping preserved',
            'scope': 'Execution flags and numerically identical LR representation only; moment values/step counts/parameter ordering unchanged'}


def update_native(model, optimizer, dataset, provider, indices, micro_batch):
    """Ordinary independent forward/backward/clip/step, with no retry or compile."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = [provider(dataset, indices[start:start + micro_batch])
               for start in range(0, len(indices), micro_batch)]
    targets = sum(count for _, count, _ in batches)
    inputs = sum(count for _, _, count in batches)
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    if targets == 0:
        return total, total.clone(), inputs, targets, False
    for values, count, _ in batches:
        if count == 0:
            continue
        if not isinstance(values.get('attention_mask'), torch.Tensor) or values['attention_mask'].ndim != 2:
            raise ValueError('Native standard path requires ordinary 2D attention masks')
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = model(**values, use_cache=False).loss
        weighted = loss * (count / targets)
        weighted.backward()
        total += weighted.detach()
    # No failed update may mutate parameters or Adam state.
    if not bool(torch.isfinite(total)):
        raise RuntimeError(f'Nonfinite native training loss {float(total)}; optimizer not applied')
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    except RuntimeError as exc:
        raise RuntimeError(f'Native gradient clipping failed with loss={float(total)}; optimizer not applied: {exc}') from exc
    optimizer.step()
    return total.detach(), norm.detach(), inputs, targets, True


def prepare_train_probe(dataset, blocks, seed, existing=None):
    if blocks < 0:
        raise ValueError('Train probe block count must be nonnegative')
    if existing is not None and (existing['blocks'] != blocks or existing['seed'] != seed):
        raise ValueError('Resume must preserve the fixed train probe block count/seed')
    if blocks == 0:
        return None
    full = len(dataset.tokens) // dataset.sequence_length
    if blocks > full:
        raise ValueError('Fixed train probe exceeds complete training blocks')
    indices = np.sort(np.random.default_rng(seed).choice(full, size=blocks, replace=False)).astype(np.int64)
    digest = hashlib.sha256()
    for index in indices:
        start = int(index) * dataset.sequence_length
        digest.update(memoryview(np.ascontiguousarray(dataset.tokens[start:start + dataset.sequence_length])).cast('B'))
    probe = {'schema_version': 1, 'seed': seed, 'blocks': blocks, 'indices': indices.tolist(),
             'indices_sha256': hashlib.sha256(memoryview(indices).cast('B')).hexdigest(),
             'tokens_sha256': digest.hexdigest(), 'input_tokens': blocks * dataset.sequence_length,
             'supervised_tokens': blocks * (dataset.sequence_length - 1),
             'sampling': 'Fixed uniform complete training-corpus blocks; private NumPy RNG; not current minibatch'}
    if existing is not None and existing != probe:
        raise ValueError('Saved fixed training probe differs from current dataset/sample')
    return probe


@contextmanager
def preserve_rng(device):
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)
