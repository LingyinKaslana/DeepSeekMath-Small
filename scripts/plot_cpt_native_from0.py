"""CPU-only report for fresh native CPT's five-hour early-curve observation.

Validation compares real native-eager measurements on the same held-out tokens.
Rolling training losses are deliberately excluded. Historical compiler/source
implementation differences remain explicit; they do not invalidate this recipe
comparison or prove a cause of a learning-curve change.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
BASELINE_LOSS = 1.86186945438385
PROBE_INDEX_SHA = '0275fe980c1fd3bf853fa63d40c27291d4368b0eb41b4c183127d9f964169a81'
PROBE_TOKEN_SHA = 'f2d447c272813ee27c047b0a895234932d1d05cb1e4275b11b982dfedee42619'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def tokens_at(step, manifest):
    length, batch = manifest['signature']['sequence_length'], manifest['signature']['effective_batch']
    budget, epochs = manifest['dataset_input_tokens'], manifest['signature']['epochs']
    blocks = math.ceil(budget / length)
    steps = math.ceil(blocks / batch)
    if not isinstance(step, int) or not 0 <= step <= epochs * steps:
        raise ValueError('Step exceeds recorded epoch/token budget')
    complete, partial = divmod(step, steps)
    total = complete * budget
    if partial:
        selected = min(partial * batch, blocks)
        total += selected * length
        if budget % length and manifest['order_metadata']['files'][complete]['tail_position'] < selected:
            total -= length - budget % length
    return total


def verify_sources(run, manifest):
    checked = {}
    for name, digest in manifest['source_sha256'].items():
        path = run / 'source' / name
        if Path(name).name != name or sha(path) != digest:
            raise ValueError(f'Immutable source snapshot mismatch: {path}')
        checked[name] = digest
    backend = run / 'backend_config.json'
    if sha(backend) != manifest['signature']['backend_config_sha256']:
        raise ValueError('Backend configuration no longer matches its signature')
    data = run / 'data_manifest.json'
    if sha(data) != manifest['signature']['data_manifest_sha256']:
        raise ValueError('Data manifest no longer matches its signature')
    orders = {}
    for item in manifest['order_metadata']['files']:
        path = run / 'orders' / item['name']
        if sha(path) != item['sha256']:
            raise ValueError('Order file no longer matches its checksum')
        orders[item['name']] = item['sha256']
    return {'source_sha256': checked, 'backend_config_sha256': sha(backend),
            'data_manifest_sha256': sha(data), 'order_sha256': orders}


def adam_math(source_path):
    calls = [node for node in ast.walk(ast.parse(source_path.read_text()))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'AdamW']
    protocols = []
    for call in calls:
        keywords = {key.arg: key.value for key in call.keywords}
        if not {'betas', 'weight_decay'} <= keywords.keys():
            continue
        try:
            record = {'betas': list(ast.literal_eval(keywords['betas'])),
                      'weight_decay': ast.literal_eval(keywords['weight_decay']),
                      'eps': ast.literal_eval(keywords['eps']) if 'eps' in keywords else 1e-8,
                      'eps_basis': 'Explicit source value' if 'eps' in keywords else 'torch.optim.AdamW default, same recorded torch version'}
        except (ValueError, TypeError):
            continue
        if record['betas'] == [.9, .95] and record['weight_decay'] == .1 and record['eps'] == 1e-8:
            protocols.append(record)
    if not protocols:
        raise ValueError(f'Expected CPT AdamW math protocol missing from {source_path}')
    return protocols[0]


def clip_math(source_path):
    for call in ast.walk(ast.parse(source_path.read_text())):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'clip_grad_norm_' and len(call.args) > 1):
            try:
                limit = ast.literal_eval(call.args[1])
                keywords = {key.arg: key.value for key in call.keywords}
                strict = ast.literal_eval(keywords['error_if_nonfinite'])
            except (KeyError, ValueError, TypeError):
                continue
            if limit == 1.0 and strict is True:
                return {'max_grad_norm': 1.0, 'error_if_nonfinite': True}
    raise ValueError('Expected strict CPT gradient clipping is absent')



def verify_fresh_lineage(run, manifest, historical=False):
    keys = ('original_signature', 'source_migrations')
    if all(key in manifest for key in keys):
        if manifest['original_signature'] != manifest['signature'] or manifest['source_migrations'] != []:
            raise ValueError('Fresh comparison run contains a migration lineage')
        return {'run': str(run), 'basis': 'Explicit original_signature equals own signature; explicit empty migration list'}
    if (not historical or any(key in manifest for key in keys)
            or 'source_migration.py' in manifest['source_sha256']
            or any(key in manifest['arguments'] for key in ('allow_source_migration', 'allow_native_backend_migration'))
            or manifest['source_sha256'] != manifest['signature']['source_sha256']):
        raise ValueError('Missing fresh-lineage metadata is unsupported for this source format')
    # The original run predates lineage support. Its checked immutable trainer
    # has no migration module/API; explicit no-resume CLI and zero initial
    # counters (checked below) establish this historical fresh-start format.
    tree = ast.parse((run / 'source/train_cpt.py').read_text())
    if any(isinstance(node, ast.ImportFrom) and node.module == 'source_migration' for node in ast.walk(tree)):
        raise ValueError('Historical manifest omits lineage despite a migration-capable trainer')
    if any(isinstance(node, ast.Name) and node.id in ('apply_source_migration', 'apply_native_migration') for node in ast.walk(tree)):
        raise ValueError('Historical source unexpectedly supports migration')
    return {'run': str(run), 'basis': 'Historical pre-lineage trainer: no migration module/CLI, verified source signature, no resume and explicit zero initial counters',
            'missing_historical_fields': list(keys), 'inferred_original_signature': manifest['signature'], 'inferred_migrations': []}


def verify_controls(native_run, fast_run, native, fast):
    lineage = []
    for run, manifest in ((native_run, native), (fast_run, fast)):
        lineage.append(verify_fresh_lineage(run, manifest, historical=run == fast_run))
        config, args = manifest['backend_config'], manifest['arguments']
        if (config['model'] != 'Qwen/Qwen2.5-Coder-1.5B'
                or config['revision'] != 'df3ce67c0e24480f20468b6ef2894622d69eb73b'
                or (config['effective_batch'], config['micro_batch'], config['sequence_length']) != (16, 16, 2048)
                or config['options']['kernels'] != 'native' or config['options']['loss'] != 'native'
                or manifest['precision'] != 'BF16 autocast; FP32 parameters, gradients and AdamW moments'
                or args.get('resume') is not None or args.get('steps') is not None
                or manifest['resumed_completed_steps'] != 0 or manifest['resumed_input_tokens'] != 0):
            raise ValueError(f'Expected a fresh pinned-base run with unchanged full-parameter protocol: {run}')
        if manifest['configured_input_tokens'] != 10_000_000_000 or manifest['dataset_input_tokens'] != 5_000_000_000:
            raise ValueError('The five-hour boundary must not shorten the full 10B token schedule')
    args, config = native['arguments'], native['backend_config']
    if (args.get('max_seconds') != 18000 or args.get('stop_after') is not None
            or args.get('allow_source_migration', False) or args.get('allow_native_backend_migration', False)
            or config.get('execution_mode') != 'standard_native' or config['mask_policy'] != 'ones'
            or config.get('compiled_optimizer', False) or config.get('fused_optimizer', False)
            or config['options']['compile_scope'] != 'none' or args.get('retry_native_on_nonfinite', False)):
        raise ValueError('Fresh native run must use uncompiled standard execution and an 18000-second update-boundary limit')
    mathematical_keys = ('data_manifest_sha256', 'token_file_sha256', 'peak_lr', 'seed',
                         'sequence_length', 'effective_batch', 'epochs')
    if any(native['signature'][key] != fast['signature'][key] for key in mathematical_keys):
        raise ValueError('Native/fast math or data signatures differ')
    if (native['schedule'] != fast['schedule'] or native['order_metadata'] != fast['order_metadata']
            or native['schedule']['total_steps'] != 305176 or native['signature']['seed'] != 42
            or native['signature']['peak_lr'] != 1e-5):
        raise ValueError('Native/fast order, seed, or full LR schedule differs')
    if native['versions']['torch'] != fast['versions']['torch'] or native['versions']['transformers'] != fast['versions']['transformers']:
        raise ValueError('The compared framework versions differ')
    probe = native['train_probe']
    if (probe['blocks'], probe['seed'], probe['indices_sha256'], probe['tokens_sha256']) != (
            256, 20261009, PROBE_INDEX_SHA, PROBE_TOKEN_SHA):
        raise ValueError('Fixed native train probe differs from the established corpus sample')
    native_proofs, fast_proofs = verify_sources(native_run, native), verify_sources(fast_run, fast)
    native_math = adam_math(native_run / 'source/standard_native.py')
    fast_math = adam_math(fast_run / 'source/compiled_adamw.py')
    if {key: value for key, value in native_math.items() if key != 'eps_basis'} != {key: value for key, value in fast_math.items() if key != 'eps_basis'}:
        raise ValueError('Audited AdamW math differs')
    native_clip, fast_clip = clip_math(native_run / 'source/standard_native.py'), clip_math(fast_run / 'source/train_cpt.py')
    if native_clip != fast_clip:
        raise ValueError('Gradient clipping differs')
    return {'status': 'passed', 'fresh_lineage_checks': lineage, 'same_pinned_base': True, 'fresh_weights_and_new_adam_evidence':
            'CLI has no resume; initial counters zero; native has explicit own signature/empty migrations; historical pre-lineage fast format separately audited; verified fresh-start sources',
            'initial_tensor_snapshot_checked': False, 'same_data_batch_math_order_schedule': True,
            'same_source': native['source_sha256'] == fast['source_sha256'],
            'implementation_differences': ['Historical immutable source snapshots differ',
                'Fast: whole-model compile + compiled capturable AdamW + resolved causal mask + vectorized label alias',
                'Native: untouched HF eager + standard foreach AdamW + ordinary all-ones masks + independent stacked labels'],
            'differences_do_not_invalidate_recipe_comparison': True,
            'optimizer_math': native_math, 'gradient_clip': native_clip,
            'native_snapshots': native_proofs, 'fast_snapshots': fast_proofs,
            'precision': native['precision'], 'schedule': native['schedule'], 'fixed_train_probe': probe,
            'seed': 42, 'native_eval_every': native['arguments']['eval_every'],
            'fast_eval_every': fast['arguments']['eval_every']}


def read_series(run, manifest, label, train_probe=False):
    paths = [run / 'train_probe_history.json'] if train_probe else [run / 'validation_before.json', run / 'validation_history.json']
    points, sources, duplicates = {}, [], 0
    expected = (524288, 524032, 256) if train_probe else (2097152, 2096128, 1024)
    for path in paths:
        if not path.is_file():
            continue
        sources.append({'path': str(path.resolve()), 'sha256': sha(path)})
        payload = read(path)
        for item in payload if isinstance(payload, list) else [payload]:
            if (item.get('loss_backend') != 'native eager'
                    or (item['input_tokens'], item['supervised_tokens'], item['blocks']) != expected
                    or not math.isfinite(item['loss']) or not math.isfinite(item['perplexity'])):
                raise ValueError('Evaluation is not finite native eager on the expected fixed split')
            step = item['step']
            tokens = tokens_at(step, manifest)
            if item.get('cumulative_tokens', tokens) != tokens:
                raise ValueError('Recorded evaluation token prefix differs from saved order')
            point = {'series': label, 'step': step, 'trained_input_tokens': tokens, 'trained_tokens_b': tokens / 1e9,
                     'loss': item['loss'], 'perplexity': item['perplexity'],
                     'evaluated_input_tokens': expected[0], 'evaluated_targets': expected[1],
                     'sources': [str(path.resolve())]}
            if step in points:
                old = points[step]
                if {key: value for key, value in old.items() if key != 'sources'} != {key: value for key, value in point.items() if key != 'sources'}:
                    raise ValueError('Conflicting fixed-split measurements at the same step')
                old['sources'].extend(point['sources'])
                duplicates += 1
            else:
                points[step] = point
    return [point for _, point in sorted(points.items())], sources, duplicates


def progress(run, manifest, observed):
    latest = {'step': 0, 'trained_input_tokens': 0, 'source': str(run / 'manifest.json')}
    for point in observed:
        if point['step'] >= latest['step']:
            latest = {'step': point['step'], 'trained_input_tokens': point['trained_input_tokens'], 'source': point['sources'][0]}
    path = run / 'progress.json'
    if path.is_file():
        item = read(path)
        if item['cumulative_tokens'] != tokens_at(item['step'], manifest):
            raise ValueError('Progress tokens differ from exact order prefix')
        if item['step'] >= latest['step']:
            latest = {'step': item['step'], 'trained_input_tokens': item['cumulative_tokens'], 'source': str(path)}
    result_path = run / 'result.json'
    result = read(result_path) if result_path.is_file() else None
    status = 'in_progress'
    if result is not None:
        if result['status'] == 'failed':
            status = 'failed'
        elif result['status'] == 'stopped':
            status = 'completed_observation' if result.get('stop_reason') == 'max_seconds' else 'stopped_early'
        elif result['status'] == 'complete':
            status = 'completed_observation'
        if result.get('completed_steps') is not None:
            if result['trained_tokens'] != tokens_at(result['completed_steps'], manifest):
                raise ValueError('Final result counters differ from exact order prefix')
            latest = {'step': result['completed_steps'], 'trained_input_tokens': result['trained_tokens'], 'source': str(result_path)}
    return dict(latest, trained_tokens_b=latest['trained_input_tokens'] / 1e9), status, result


def draw(summary, output):
    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 8.5), sharex=True, facecolor='white')
    fig.subplots_adjust(left=.10, right=.96, bottom=.25, top=.78, hspace=.40)
    status_text = {'in_progress': '5 小时观察进行中', 'completed_observation': '5 小时观察已结束',
                   'stopped_early': '观察提前停止', 'failed': '运行异常停止', 'waiting_for_initialization': '等待初始化完成'}
    fig.text(.10, .952, '从底座重新开始：原生训练早期曲线', fontsize=23, weight='bold', color='#18364e')
    latest = summary['latest_native_progress']
    fig.text(.10, .901, 'Qwen2.5-Coder-1.5B · OpenWebMath · 全参 BF16 / FP32 · 完整 10B LR 计划', color='#586e80')
    fig.text(.10, .852, f"{status_text[summary['status']]} · 原生进度 step {latest['step']:,} / {latest['trained_tokens_b']:.3f}B 输入 tokens", color='#34556e')
    series = summary['series']
    native_val, fast_val, probe = series['native_validation'], series['original_fast_validation'], series['native_train_probe']
    # Focus on the actually reached native interval; retain all fast points in
    # CSV/JSON. Show its next real evaluation as historical context, not a new
    # interpolated observation or a native prediction.
    beyond = [point for point in fast_val if point['trained_tokens_b'] > latest['trained_tokens_b']]
    boundary = beyond[0]['trained_tokens_b'] if beyond else latest['trained_tokens_b']
    right = max(latest['trained_tokens_b'], boundary, .025)
    visible_fast = [point for point in fast_val if point['trained_tokens_b'] <= right]
    warm, plateau_end = summary['schedule_regions']['warmup_end_tokens_b'], summary['schedule_regions']['peak_end_tokens_b']
    for ax in axes:
        ax.axvspan(0, min(warm, right), color='#f4b860', alpha=.10, zorder=0)
        if right > warm:
            ax.axvspan(warm, min(right, plateau_end), color='#70a4cc', alpha=.07, zorder=0)
            ax.axvline(warm, color='#b89056', lw=1, ls=':')
        ax.axvline(latest['trained_tokens_b'], color='#a2afba', ls=':', lw=1.1)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(axis='y', color='#e6edf3', lw=.8)
        ax.set_axisbelow(True)
        ax.set_xlim(-.025 * right, right * 1.045)
        ax.set_ylabel('平均 token loss')
    axes[0].set_title('完整固定验证集：2,097,152 输入 tokens（均用原生 eager 评估）', loc='left', fontsize=12, weight='bold', pad=13)
    axes[0].axhline(BASELINE_LOSS, color='#8998a4', ls='--', lw=1.1, label=f'共同底座 {BASELINE_LOSS:.4f}')
    for points, color, label, marker in ((visible_fast, '#a67845', '历史优化训练，每 5,000 步验证', 's'),
                                         (native_val, '#2e6eaa', '从零原生训练，每 1,000 步验证', 'o')):
        if points:
            axes[0].plot([point['trained_tokens_b'] for point in points], [point['loss'] for point in points],
                          color=color, lw=2, marker=marker, ms=4.5, label=label)
    axes[0].legend(loc='upper right', frameon=False, fontsize=9)
    axes[1].set_title('原生固定训练样本：524,288 输入 tokens（独立固定样本，原生 eager 评估）', loc='left', fontsize=12, weight='bold', pad=13)
    if probe:
        axes[1].plot([point['trained_tokens_b'] for point in probe], [point['loss'] for point in probe],
                     color='#2d8979', lw=2, marker='o', ms=4.5, label='从零原生：固定 train probe')
        axes[1].axhline(probe[0]['loss'], color='#8998a4', ls='--', lw=1, label=f"probe 底座 {probe[0]['loss']:.4f}")
        axes[1].legend(loc='upper right', frameon=False, fontsize=9)
    else:
        axes[1].text(.5, .5, '等待固定训练样本初始评估', transform=axes[1].transAxes, ha='center', color='#718695')
    axes[1].set_xlabel('累计训练输入 tokens（B，十亿；虚线标记当前原生进度）', labelpad=11)
    fig.text(.10, .131, f"浅橙区域：warmup 至 {warm:.3f}B；浅蓝区域：LR = 1e-5 平台（至约 {plateau_end:.1f}B，图中仅显示当前早期区间）。", fontsize=9, color='#627988')
    fig.text(.10, .089, '点均为真实评估；连线仅便于阅读，未生成插值评估点，也未混入滚动训练 loss。历史优化源码与原生源码不同。', fontsize=9, color='#627988')
    fig.text(.10, .047, '5 小时是观察时限；该状态不代表完成 10B。历史曲线至下一真实验证点供参考，不预测原生后续结果。', fontsize=9, color='#7b8e9b')
    for extension in ('png', 'svg'):
        fig.savefig(output / f'native_from0.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-run', type=Path, default=ROOT / 'runs/cpt_native_from0_5h_20261009')
    parser.add_argument('--fast-run', '--original-fast', dest='original_fast', type=Path, default=ROOT / 'runs/cpt_10b_20261008')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.native_run, args.original_fast = args.native_run.resolve(), args.original_fast.resolve()
    explicit_output = args.output is not None
    args.output = args.output or args.native_run / 'report'
    now = datetime.now(timezone.utc)
    summary = {'schema_version': 1, 'generated_utc': now.isoformat(),
               'generated_beijing': now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
               'native_run': str(args.native_run), 'original_fast': str(args.original_fast),
               'script_sha256': sha(Path(__file__)), 'gpu_execution': False,
               'observation_budget_seconds': 18000, 'configured_training_input_tokens': 10_000_000_000,
               'interpretation': 'Fixed native-eager split loss comparison under matched recipe; not a root-cause determination or a claim of final 10B completion.'}
    native_path, fast_path = args.native_run / 'manifest.json', args.original_fast / 'manifest.json'
    if not native_path.is_file():
        fast = read(fast_path)
        reference, reference_sources, _ = read_series(args.original_fast, fast, 'original_fast_validation')
        if not reference or reference[0]['step'] != 0 or reference[0]['loss'] != BASELINE_LOSS:
            raise ValueError('Original fast reference baseline differs from the pinned-base record')
        summary.update(status='waiting_for_initialization', missing_artifacts=[str(native_path)],
                       original_fast_read_check={'points': len(reference), 'baseline_loss': reference[0]['loss'],
                          'snapshots': verify_sources(args.original_fast, fast),
                          'adam_math': adam_math(args.original_fast / 'source/compiled_adamw.py'),
                          'clip': clip_math(args.original_fast / 'source/train_cpt.py')})
        # A pending read must never create the output directory the trainer
        # expects to publish with exist_ok=False. Explicit report destinations
        # are authorized separately by the caller.
        if explicit_output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'status': summary['status'], 'original_fast_points_checked': len(reference),
                          'report_written': explicit_output, 'output': str(args.output.resolve())}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    native, fast = read(native_path), read(fast_path)
    if any(key not in native for key in ('schedule', 'order_metadata', 'original_signature', 'train_probe')):
        summary.update(status='waiting_for_initialization', missing_artifacts=['Native manifest final initialization metadata'])
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'status': summary['status'], 'output': str(args.output.resolve())}))
        return
    controls = verify_controls(args.native_run, args.original_fast, native, fast)
    native_val, native_sources, native_dupes = read_series(args.native_run, native, 'native_validation')
    fast_val, fast_sources, fast_dupes = read_series(args.original_fast, fast, 'original_fast_validation')
    probe, probe_sources, probe_dupes = read_series(args.native_run, native, 'native_train_probe', train_probe=True)
    if not native_val:
        summary.update(status='waiting_for_initialization', controls=controls, missing_artifacts=['Native initial validation'])
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'status': summary['status'], 'output': str(args.output.resolve())}))
        return
    for points in (native_val, fast_val):
        if not points or points[0]['step'] != 0 or points[0]['loss'] != BASELINE_LOSS:
            raise ValueError('Native/fast must have exactly the same step-zero validation baseline 1.86186945438385')
    if probe and probe[0]['step'] != 0:
        raise ValueError('Native fixed train probe must begin at step zero')
    latest, status, result = progress(args.native_run, native, native_val + probe)
    schedule = native['schedule']
    summary.update(status=status, actual_run_status=result.get('status') if result else 'running',
                   stop_reason=result.get('stop_reason') if result else None, controls=controls,
                   latest_native_progress=latest, common_validation_baseline_loss=BASELINE_LOSS,
                   full_10b_schedule_completed=bool(result and result.get('status') == 'complete'),
                   series={'native_validation': native_val, 'original_fast_validation': fast_val, 'native_train_probe': probe},
                   duplicate_measurements_removed={'native_validation': native_dupes, 'original_fast_validation': fast_dupes, 'native_train_probe': probe_dupes},
                   sources=[{'path': str(native_path), 'sha256': sha(native_path)}, {'path': str(fast_path), 'sha256': sha(fast_path)}] + native_sources + fast_sources + probe_sources,
                   interpolation='Straight lines between actual evaluation markers only; no fabricated evaluation rows',
                   schedule_regions={'warmup_end_step': schedule['warmup_steps'],
                     'warmup_end_tokens_b': tokens_at(schedule['warmup_steps'], native) / 1e9,
                     'peak_end_step': schedule['decay_80_step'], 'peak_end_tokens_b': tokens_at(schedule['decay_80_step'], native) / 1e9,
                     'peak_lr': schedule['peak_lr']})
    columns = ('series', 'step', 'trained_input_tokens', 'trained_tokens_b', 'loss', 'perplexity', 'evaluated_input_tokens', 'evaluated_targets', 'sources')
    with (args.output / 'native_from0.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for point in sorted(native_val + fast_val + probe, key=lambda item: (item['step'], item['series'])):
            writer.writerow(dict(point, sources=';'.join(point['sources'])))
    draw(summary, args.output)
    summary['matplotlib_version'] = matplotlib.__version__
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': status, 'native_points': len(native_val), 'native_probe_points': len(probe),
                      'fast_points': len(fast_val), 'latest_progress': latest, 'controls': controls['status'],
                      'output': str(args.output.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
