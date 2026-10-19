"""CPU-only native CPT observation plot with exact token-prefix accounting.

The native window starts at update 181650; inherited optimized-training points
are excluded. Missing later runs and an unfinished 10000-update observation
window are explicitly recorded as in_progress. This makes no causal claim.
"""
import argparse
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
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
PROBE_INDEX_SHA = '0275fe980c1fd3bf853fa63d40c27291d4368b0eb41b4c183127d9f964169a81'
PROBE_TOKEN_SHA = 'f2d447c272813ee27c047b0a895234932d1d05cb1e4275b11b982dfedee42619'
BASELINE_LOSS = 1.9010149240493774


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def input_tokens(step, manifest):
    """Exact global shuffle prefix, including each epoch's shuffled tail block."""
    length = manifest['signature']['sequence_length']
    batch = manifest['signature']['effective_batch']
    budget = manifest['dataset_input_tokens']
    blocks = math.ceil(budget / length)
    epoch_steps = math.ceil(blocks / batch)
    if not 0 <= step <= epoch_steps * manifest['signature']['epochs']:
        raise ValueError('Step exceeds saved data-order budget')
    complete, partial = divmod(step, epoch_steps)
    total = complete * budget
    if partial:
        selected = min(partial * batch, blocks)
        total += selected * length
        remainder = budget % length
        if remainder and manifest['order_metadata']['files'][complete]['tail_position'] < selected:
            total -= length - remainder
    return total


def comparability(manifest):
    return {'signature': manifest['signature'], 'schedule': manifest['schedule'],
            'order_metadata': manifest['order_metadata'], 'dataset_input_tokens': manifest['dataset_input_tokens'],
            'train_probe': manifest['train_probe']}


def check_probe(probe):
    if (probe is None or probe.get('blocks') != 256 or probe.get('seed') != 20261009
            or probe.get('indices_sha256') != PROBE_INDEX_SHA or probe.get('tokens_sha256') != PROBE_TOKEN_SHA
            or probe.get('input_tokens') != 524288 or probe.get('supervised_tokens') != 524032):
        raise ValueError('Native invocation fixed probe differs from the recorded A/B probe')
    # Hash without NumPy/Torch: explicit little-endian int64 is the .npy source
    # platform's canonical 256-index representation used by the training code.
    import struct
    indices = probe.get('indices')
    if not isinstance(indices, list) or len(indices) != 256 or indices != sorted(set(indices)):
        raise ValueError('Invalid sorted unique fixed-probe indices')
    packed = b''.join(struct.pack('<q', index) for index in indices)
    if hashlib.sha256(packed).hexdigest() != PROBE_INDEX_SHA:
        raise ValueError('Fixed probe index values do not match their checksum')


def load_observation(args):
    sources, missing, warnings = [], [], []
    manifests, run_status, candidates = [], [], []
    points = {'validation': {}, 'train_probe': {}}
    duplicates = {'validation': 0, 'train_probe': 0}
    excluded = {'validation': 0, 'train_probe': 0}
    reference = None

    def add_source(path):
        sources.append({'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size})

    for run in args.run:
        run = run.resolve()
        manifest_path = run / 'manifest.json'
        if not manifest_path.is_file():
            missing.append(str(run))
            continue
        manifest = read(manifest_path)
        add_source(manifest_path)
        if manifest.get('execution_mode', manifest.get('backend_config', {}).get('execution_mode')) != 'standard_native':
            raise ValueError(f'Input run is not standard_native: {run}')
        config = manifest['backend_config']
        if (config.get('compiled_optimizer', False) or config.get('fused_optimizer', False)
                or config['options']['compile_scope'] != 'none' or config['mask_policy'] != 'ones'):
            raise ValueError('Native plot input contains an optimized backend')
        check_probe(manifest.get('train_probe'))
        if reference is None:
            reference = manifest
        elif comparability(manifest) != comparability(reference):
            raise ValueError('Native runs have different signature/schedule/order/data/probe')
        manifests.append({'run': str(run), 'resumed_completed_steps': manifest['resumed_completed_steps']})
        initial = manifest['resumed_completed_steps']
        candidates.append({'step': initial, 'input_tokens': input_tokens(initial, manifest), 'source': str(manifest_path)})
        for filename, kind in (('validation_before.json', 'validation'), ('validation_history.json', 'validation'),
                               ('train_probe_history.json', 'train_probe')):
            path = run / filename
            if not path.is_file():
                warnings.append(f'Not yet available: {path}')
                continue
            add_source(path)
            payload = read(path)
            rows = payload if isinstance(payload, list) else [payload]
            for item in rows:
                step = item['step']
                if step < args.start_step:
                    excluded[kind] += 1
                    continue
                expected_inputs, expected_targets, expected_blocks = ((2097152, 2096128, 1024)
                    if kind == 'validation' else (524288, 524032, 256))
                if (item.get('loss_backend') != 'native eager'
                        or (item.get('input_tokens'), item.get('supervised_tokens'), item.get('blocks'))
                        != (expected_inputs, expected_targets, expected_blocks)
                        or not math.isfinite(item['loss']) or not math.isfinite(item['perplexity'])):
                    raise ValueError(f'Invalid native evaluation in {path}, step {step}')
                tokens = input_tokens(step, manifest)
                if item.get('cumulative_tokens', tokens) != tokens:
                    raise ValueError('Recorded probe token prefix differs from exact order accounting')
                row = {'kind': kind, 'step': step, 'input_tokens': tokens, 'loss': item['loss'],
                       'perplexity': item['perplexity'], 'evaluated_input_tokens': expected_inputs,
                       'evaluated_targets': expected_targets, 'evaluated_blocks': expected_blocks,
                       'sources': [str(path.resolve())]}
                if step in points[kind]:
                    old = points[kind][step]
                    if {key: value for key, value in row.items() if key != 'sources'} != {key: value for key, value in old.items() if key != 'sources'}:
                        raise ValueError(f'Conflicting {kind} evaluations at the same step {step}; refusing to silently choose one')
                    old['sources'].extend(row['sources'])
                    duplicates[kind] += 1
                else:
                    points[kind][step] = row
                candidates.append({'step': step, 'input_tokens': tokens, 'source': str(path)})
        progress_path = run / 'progress.json'
        if progress_path.is_file():
            add_source(progress_path)
            progress = read(progress_path)
            step = progress['step']
            tokens = input_tokens(step, manifest)
            if progress['cumulative_tokens'] != tokens:
                raise ValueError('Progress token count differs from exact shuffle prefix')
            candidates.append({'step': step, 'input_tokens': tokens, 'source': str(progress_path)})
        result_path = run / 'result.json'
        if result_path.is_file():
            add_source(result_path)
            result = read(result_path)
            run_status.append({'run': str(run), 'status': result['status'], 'completed_steps': result.get('completed_steps')})
            if result.get('completed_steps') is not None:
                step = result['completed_steps']
                tokens = input_tokens(step, manifest)
                if result['trained_tokens'] != tokens:
                    raise ValueError('Result trained token count differs from exact shuffle prefix')
                candidates.append({'step': step, 'input_tokens': tokens, 'source': str(result_path)})
        else:
            run_status.append({'run': str(run), 'status': 'in_progress'})
    if reference is None:
        raise ValueError('No native run manifest is available yet')
    start_tokens = input_tokens(args.start_step, reference)
    target_step = args.start_step + args.window_updates
    target_tokens = input_tokens(target_step, reference)
    if args.start_step not in points['validation']:
        raise ValueError('Missing native-window validation baseline at start_step')
    baseline = points['validation'][args.start_step]
    if args.start_step == 181650 and baseline['loss'] != BASELINE_LOSS:
        raise ValueError('Start validation differs from the saved same-weight baseline 1.9010149240493774')
    latest = max(candidates, key=lambda row: row['step'])
    if latest['step'] < args.start_step:
        raise ValueError('Latest native progress is before window start')
    series = {}
    for kind in points:
        series[kind] = [dict(row, native_updates=row['step'] - args.start_step,
                             new_input_tokens=row['input_tokens'] - start_tokens,
                             new_input_tokens_m=(row['input_tokens'] - start_tokens) / 1e6)
                        for _, row in sorted(points[kind].items())]
    latest = dict(latest, native_updates=latest['step'] - args.start_step,
                  new_input_tokens=latest['input_tokens'] - start_tokens,
                  new_input_tokens_m=(latest['input_tokens'] - start_tokens) / 1e6)
    target_evaluated = all(rows and rows[-1]['step'] >= target_step for rows in series.values())
    target_reached = latest['step'] >= target_step
    window_status = ('observed' if target_reached and target_evaluated else
                     'awaiting_evaluation' if target_reached else 'in_progress')
    now = datetime.now(timezone.utc)
    return {'schema_version': 1, 'generated_utc': now.isoformat(),
            'generated_beijing': now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
            'status': window_status, 'run_status': run_status, 'missing_runs': missing, 'warnings': warnings,
            'window': {'start_step': args.start_step, 'start_input_tokens': start_tokens,
                       'target_updates': args.window_updates, 'target_step': target_step,
                       'target_new_input_tokens': target_tokens - start_tokens,
                       'target_new_input_tokens_m': (target_tokens - start_tokens) / 1e6,
                       'training_target_reached': target_reached, 'evaluation_target_observed': target_evaluated},
            'latest_progress': latest, 'validation_baseline': baseline, 'series': series,
            'duplicate_measurements_removed': duplicates, 'inherited_pre_native_points_excluded': excluded,
            'native_invocations': manifests,
            'fixed_train_probe': {key: reference['train_probe'][key] for key in ('blocks', 'seed', 'indices_sha256', 'tokens_sha256', 'input_tokens', 'supervised_tokens')},
            'comparison_probe_matches_ab': True, 'data_order_and_signature_match_across_native_invocations': True,
            'sources': sources, 'script_sha256': sha(Path(__file__)), 'gpu_execution': False,
            'interpretation': 'Descriptive fixed-sample native loss observations only; no claim that native is better or that a root cause has been established.'}


def draw(summary, output):
    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 8.1), sharex=True, facecolor='white')
    fig.subplots_adjust(left=.105, right=.96, bottom=.20, top=.73, hspace=.34)
    window, latest = summary['window'], summary['latest_progress']
    ratio = min(latest['native_updates'] / window['target_updates'], 1.0)
    statuses = {'in_progress': '观察进行中', 'awaiting_evaluation': '已达到训练量，等待固定样本评估', 'observed': '首窗训练及评估已达到目标'}
    fig.text(.105, .946, '原生继续预训练：固定样本 loss 观察', fontsize=23, weight='bold', color='#173149')
    fig.text(.105, .897, f"HF 原生 + PyTorch AdamW · 全参 BF16 / FP32 · 起点 step {window['start_step']:,}", color='#586d7f')
    fig.text(.105, .851, f"首窗目标 {window['target_updates']:,} 更新：已训练 {latest['native_updates']:,} 更新（{100 * ratio:.2f}%） · {statuses[summary['status']]}", color='#314e67')
    fig.add_artist(Rectangle((.105, .813), .855, .014, transform=fig.transFigure, facecolor='#e5ecf3', edgecolor='none'))
    fig.add_artist(Rectangle((.105, .813), .855 * ratio, .014, transform=fig.transFigure, facecolor='#427db0', edgecolor='none'))
    fig.text(.105, .781, f"最新进度 step {latest['step']:,} · 已新增 {latest['new_input_tokens_m']:.3f}M / 首窗目标 {window['target_new_input_tokens_m']:.3f}M 输入 tokens", fontsize=10, color='#586d7f')
    right = max(latest['new_input_tokens_m'], 1.0)
    for ax, kind, title, color in zip(axes, ('validation', 'train_probe'),
            ('完整验证集：2,097,152 输入 tokens', '固定训练样本：524,288 输入 tokens'), ('#2e6caa', '#2b8776')):
        rows = summary['series'][kind]
        ax.set_title(title, loc='left', fontsize=12, weight='bold', pad=12)
        if rows:
            x, y = [row['new_input_tokens_m'] for row in rows], [row['loss'] for row in rows]
            ax.plot(x, y, lw=2, color=color, marker='o', ms=5, label='native eager loss', zorder=3)
            first = rows[0]
            ax.axhline(first['loss'], color='#8899a7', lw=1, ls='--', label=f"窗口起点 {first['loss']:.6f}")
            final = rows[-1]
            ax.annotate(f"最近评估 {final['loss']:.6f}\nstep {final['step']:,} / {final['new_input_tokens_m']:.3f}M",
                        xy=(final['new_input_tokens_m'], final['loss']), xytext=(.97, .88), textcoords='axes fraction',
                        ha='right', va='top', fontsize=10, color=color,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .87, 'pad': 3})
            span = max(max(y) - min(y), .0004)
            ax.set_ylim(min(y) - .27 * span, max(y) + .45 * span)
            ax.legend(loc='lower left', frameon=False, fontsize=9, ncols=2)
        else:
            ax.text(.5, .5, '尚无固定样本评估，观察进行中', ha='center', va='center', transform=ax.transAxes, color='#738496')
        ax.axvline(latest['new_input_tokens_m'], color='#a6b5c1', lw=1, ls=':', alpha=.8)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.set_ylabel('平均 token loss')
        ax.set_xlim(-.02 * right, 1.045 * right)
        ax.grid(axis='y', color='#e7edf2', lw=.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel(f"自 step {window['start_step']:,} 起新增训练输入 tokens（M，百万；显示当前观察区间）", labelpad=10)
    fig.text(.105, .059, '仅显示原生阶段；连接线未经平滑、不外推。首窗目标达到前状态始终为观察进行中。', fontsize=9, color='#647b8b')
    fig.text(.105, .027, f"数据读取时间（北京时间）：{summary['generated_beijing'][:19].replace('T', ' ')} · 图中 loss 不用于宣称优化改善或确定原因。", fontsize=9, color='#788b99')
    for extension in ('png', 'svg'):
        fig.savefig(output / f'native_observation.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', help='Repeat for native invocations; defaults to migration smoke + long native run')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/cpt_native_long_20261009/observation')
    parser.add_argument('--start-step', type=int, default=181650)
    parser.add_argument('--window-updates', type=int, default=10000)
    args = parser.parse_args()
    args.run = args.run or [ROOT / 'runs/cpt_native_long_20261009/migration_smoke', ROOT / 'runs/cpt_10b_20261009_native']
    if args.window_updates < 1:
        parser.error('--window-updates must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = load_observation(args)
    columns = ('kind', 'step', 'native_updates', 'input_tokens', 'new_input_tokens', 'new_input_tokens_m',
               'loss', 'perplexity', 'evaluated_input_tokens', 'evaluated_targets', 'evaluated_blocks', 'sources')
    with (args.output / 'native_observation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in sorted(summary['series']['validation'] + summary['series']['train_probe'], key=lambda item: (item['step'], item['kind'])):
            writer.writerow(dict(row, sources=';'.join(row['sources'])))
    draw(summary, args.output)
    summary['matplotlib_version'] = matplotlib.__version__
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': summary['status'], 'latest_progress': summary['latest_progress'],
                      'validation_points': len(summary['series']['validation']), 'train_probe_points': len(summary['series']['train_probe']),
                      'missing_runs': summary['missing_runs'], 'output': str(args.output.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
