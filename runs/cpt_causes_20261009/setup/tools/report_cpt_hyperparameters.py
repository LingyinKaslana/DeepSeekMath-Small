"""CPU-only, strict report of three isolated native CPT interventions."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


CASES = {'baseline': (1.0, .1), 'lr_0p3': (.3, .1), 'wd_zero': (1.0, 0.0)}
GROUPS = {'validation': '固定验证集', 'uniform_train': '均匀训练语料 probe',
          'seen_train': '已训练块 probe', 'unseen_train': '未训练块 probe'}
LABELS = {'baseline': '原配置', 'lr_0p3': '学习率 × 0.3', 'wd_zero': 'weight decay = 0'}
COMMON = ('checkpoint', 'checkpoint_manifest_sha256', 'checkpoint_files',
          'checkpoint_training_state_file', 'checkpoint_signature', 'source_sha256',
          'diagnostic_script_sha256', 'helper_sha256', 'data_manifest_sha256',
          'data_artifact_sha256', 'start_step', 'end_step', 'updates', 'schedule',
          'order_metadata', 'selected_block_indices_sha256',
          'selected_training_token_and_label_values_sha256', 'start_prefix_counts',
          'end_prefix_counts', 'batch', 'precision', 'probes', 'probes_json_sha256',
          'unseen_exclusion_through_step', 'evaluation_offsets', 'versions')
LIMITS = ['这是 1000 步、0.032768B tokens 的局部干预，仅筛查立即变化方向。',
          '学习率缩放同时改变 Adam 更新和 LR×WD 衰减；需要结合 WD=0 臂分析。',
          '局部响应不能证明长期效果或最初第 5000 步拐点的原因。',
          'seen/unseen 指本次 CPT 的块级训练历史，不保证文档或基座预训练没有重叠。',
          '应比较各固定组跨模型的变化；组间绝对难度差异不能作为过拟合证据。']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_plan(plan, case, probe_sha):
    lr, wd = CASES[case]
    expected = ([] if case == 'baseline' else
                [{'knob': 'lr_multiplier', 'from': 1.0, 'to': .3}] if case == 'lr_0p3' else
                [{'knob': 'weight_decay', 'from': .1, 'to': 0.0}])
    require(plan['case'] == case and plan['only_knob_changes'] == expected,
            'Illegal intervention fields: ' + case)
    require(plan['diagnostic_only'] is True and plan['automatic_production_resume'] is False
            and plan['production_files_modified'] is False, 'Diagnostic isolation missing')
    require(plan['lr_multiplier'] == lr and plan['weight_decay'] == wd,
            'Intervention value differs')
    require(plan['optimizer_math'] == {'betas': [.9, .95], 'eps': 1e-8,
            'weight_decay': wd, 'max_grad_norm': 1.0}, 'Optimizer math differs')
    require((plan['start_step'], plan['end_step'], plan['updates']) == (20000, 21000, 1000),
            'Unexpected diagnostic interval')
    require(plan['start_prefix_counts'] == [655360000, 655040000, 20000]
            and plan['end_prefix_counts'] == [688128000, 687792000, 21000]
            and plan['new_input_tokens'] == 32768000
            and plan['new_supervised_tokens'] == 32752000, 'Token/update counts differ')
    require(plan['schedule'] == {'total_steps': 305176, 'peak_lr': 1e-5,
            'warmup_steps': 15259, 'decay_80_step': 244141, 'decay_90_step': 274659,
            'decay_80_factor': .316, 'decay_90_factor': .1}, 'Global schedule differs')
    require(plan['batch'] == {'effective': 16, 'micro': 16, 'sequence_length': 2048}
            and plan['permitted_implementation_differences'] == [], 'Backend/batch differs')
    require(plan['precision'] == 'BF16 autocast forward; backward outside autocast; FP32 master weights, gradients, Adam moments',
            'Precision policy differs')
    require(plan['probes_json_sha256'] == probe_sha
            and plan['unseen_exclusion_through_step'] == 30860, 'Probe provenance differs')
    require(plan['evaluation_offsets'] == [0, 500, 1000], 'Evaluation schedule differs')
    require(plan['expected_initial_validation'] == 1.8604923486709595
            and 0 < plan['expected_initial_validation_tolerance'] <= 1e-6,
            'Initial validation guard weakened')


def validate_evaluations(rows):
    offsets = [row['offset'] for row in rows]
    require(offsets == sorted(set(offsets)) and set(offsets) <= {0, 500, 1000},
            'Duplicate or invalid evaluation offsets')
    for row in rows:
        require(row['global_step'] == 20000 + row['offset'], 'Evaluation global step differs')
        for group in GROUPS:
            metric = row[group]
            require(math.isfinite(metric['loss']) and metric['loss_backend'] == 'native eager',
                    'Nonfinite or non-native evaluation: ' + group)
            require(metric['blocks'] == (1024 if group == 'validation' else 256)
                    and metric['supervised_tokens'] == (2096128 if group == 'validation' else 524032),
                    'Evaluation token counts differ')
    if rows:
        require(offsets[0] == 0 and abs(rows[0]['validation']['loss'] - 1.8604923486709595) <= 1e-6,
                'Initial native validation differs from source checkpoint')


def load_branch(directory, case, probe_sha):
    branch = directory / ('branch_' + case)
    record = {'case': case, 'status': 'pending', 'directory': str(branch), 'evaluations': []}
    plan_path = branch / 'plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        validate_plan(plan, case, probe_sha)
        record['plan'] = plan
    for filename, key in [('initial_optimizer_audit.json', 'adam'), ('initial_rng.json', 'rng')]:
        if (branch / filename).exists():
            record[key] = read(branch / filename)
    if 'adam' in record:
        audit = record['adam']
        require(audit['all_finite'] is True and audit['moments_dtype'] == 'torch.float32'
                and audit['step_values'] == [20000.0] and audit['tensor_count'] == 338,
                'Initial Adam audit failed: ' + case)
    if (branch / 'evaluation_history.json').exists():
        record['evaluations'] = read(branch / 'evaluation_history.json')
        validate_evaluations(record['evaluations'])
        record['status'] = 'in_progress'
    if (branch / 'result.json').exists():
        result = read(branch / 'result.json')
        record['result'] = result
        if result['status'] == 'failed':
            record['status'] = 'failed'
        elif result['status'] == 'complete':
            require(all(key in record for key in ['plan', 'adam', 'rng']), 'Completed arm lacks initial audits')
            require(result['case'] == case and result['diagnostic_only'] is True
                    and result['automatic_production_resume'] is False
                    and result['production_files_modified'] is False,
                    'Completed result isolation differs')
            require((result['start_step'], result['end_step'], result['optimizer_updates_applied'])
                    == (20000, 21000, 1000), 'Completed update counters differ')
            require(result['new_input_tokens'] == 32768000 and result['only_knob_changes']
                    == record['plan']['only_knob_changes'], 'Completed intervention/counters differ')
            require(result['initial_rng'] == record['rng'], 'Result RNG provenance differs')
            for result_key, plan_key in [('selected_block_indices_sha256', 'selected_block_indices_sha256'),
                                        ('training_token_and_label_values_sha256', 'selected_training_token_and_label_values_sha256')]:
                require(result[result_key] == record['plan'][plan_key], 'Result batch provenance differs')
            require(result['evaluation_history'] == record['evaluations']
                    and [row['offset'] for row in record['evaluations']] == [0, 500, 1000],
                    'Completed evaluation history incomplete')
            final = read(branch / 'final_finite_audit.json')
            require(final['passed'] is True and final['expected_updates'] == 21000
                    and final['parameter_tensors'] == len(final['rows']) == 338
                    and all(row['passed'] and row['step'] == 21000
                            and row['parameter_finite'] and row['parameter_dtype'] == 'torch.float32'
                            and row['exp_avg_finite'] and row['exp_avg_sq_finite']
                            and row['exp_avg_dtype'] == row['exp_avg_sq_dtype'] == 'torch.float32'
                            for row in final['rows']),
                    'Final finite/state audit failed')
            training = [json.loads(line) for line in (branch / 'training.jsonl').read_text().splitlines()]
            require(len(training) == 1000, 'Completed arm lacks 1000 real updates')
            lr, wd = CASES[case]
            expected_lr = struct.unpack('f', struct.pack('f', 1e-5 * lr))[0]
            for offset, row in enumerate(training, 1):
                require(row['offset'] == offset and row['step'] == 20000 + offset
                        and row['scheduled_lr'] == 1e-5 and row['lr_multiplier'] == lr
                        and row['intervention_lr'] == 1e-5 * lr and row['effective_fp32_lr'] == expected_lr
                        and row['weight_decay'] == wd and row['optimizer_applied'] is True
                        and row['input_tokens'] == 32768 and row['supervised_tokens'] == 32752
                        and math.isfinite(row['loss']) and math.isfinite(row['grad_norm']),
                        'Executed update differs from intervention plan')
            checkpoint = result.get('final_checkpoint')
            if record['plan']['save_diagnostic_checkpoint']:
                require(checkpoint is not None, 'Completed arm lacks diagnostic checkpoint')
                checkpoint = Path(checkpoint)
                diagnostic = read(checkpoint / 'diagnostic_checkpoint_manifest.json')
                require(diagnostic['diagnostic_only'] is True and diagnostic['completed_steps'] == 21000
                        and diagnostic['optimizer_updates'] == 21000 and diagnostic['case'] == case
                        and diagnostic['lr_multiplier'] == lr and diagnostic['weight_decay'] == wd
                        and not (checkpoint / 'training_state.pt').exists()
                        and not (checkpoint / 'checkpoint_manifest.json').exists(),
                        'Diagnostic checkpoint isolation/counters differ')
            record['final_finite_audit'] = {key: final[key] for key in ['passed', 'expected_updates', 'parameter_tensors']}
            record['status'] = 'complete'
    return record


def snapshot_table(directory):
    path = directory / 'probe_evaluation_summary.json'
    if not path.exists():
        return {'status': 'pending', 'rows': []}
    summary = read(path)
    cases = summary.get('cases', {})
    if summary.get('status') == 'complete':
        require(set(cases) == {'base', 'step20000', 'step30860'}, 'Completed snapshot summary is incomplete')
    expected_sha = sha(directory / 'probes.json')
    require(summary['training_updates'] == 0 and summary['production_files_modified'] is False,
            'Read-only snapshot evaluation unexpectedly trained or changed production')
    table = []
    base = cases.get('base', {}).get('metrics', {})
    for label in ['base', 'step20000', 'step30860']:
        if label not in cases:
            continue
        require(cases[label]['probes_sha256'] == expected_sha, 'Snapshot probe provenance differs')
        metrics = cases[label]['metrics']
        for group in GROUPS:
            value = metrics[group]['loss']
            require(math.isfinite(value), 'Nonfinite snapshot loss')
            table.append({'snapshot': label, 'group': group, 'loss': value,
                          'delta_from_base': value - base[group]['loss'] if group in base else None})
    return {'status': summary.get('status', 'pending'), 'rows': table,
            'source_sha256': sha(path), 'interpretation': LIMITS[3:]}


def make_report(directory):
    probes = directory / 'probes.json'
    probe_sha = sha(probes)
    branches = {case: load_branch(directory, case, probe_sha) for case in CASES}
    reference = None
    for case, record in branches.items():
        if 'plan' not in record:
            continue
        if reference is None:
            reference = record
        else:
            for key in COMMON:
                require(record['plan'][key] == reference['plan'][key], 'Cross-arm control differs: ' + key)
    adam = [record['adam']['combined_tensor_sha256'] for record in branches.values() if 'adam' in record]
    require(len(set(adam)) <= 1, 'Initial Adam tensors differ across arms')
    rng = [record['rng'] for record in branches.values() if 'rng' in record]
    require(not rng or all(value == rng[0] for value in rng), 'Initial CPU/CUDA RNG differs across arms')
    initials = [record['evaluations'][0] for record in branches.values() if record['evaluations']]
    for group in GROUPS:
        values = [row[group]['loss'] for row in initials]
        require(not values or max(values) - min(values) <= 1e-6, 'Initial probe losses differ across arms')
    states = [record['status'] for record in branches.values()]
    status = ('failed' if 'failed' in states else 'complete' if all(s == 'complete' for s in states)
              else 'in_progress' if any(s == 'in_progress' for s in states) else 'pending')
    rows = []
    for case, record in branches.items():
        for evaluation in record['evaluations']:
            for group in GROUPS:
                rows.append({'case': case, 'offset': evaluation['offset'],
                             'global_step': evaluation['global_step'], 'group': group,
                             'loss': evaluation[group]['loss']})
    endpoint = {}
    base_end = next((row for row in branches['baseline']['evaluations'] if row['offset'] == 1000), None)
    for case, record in branches.items():
        end = next((row for row in record['evaluations'] if row['offset'] == 1000), None)
        if end:
            endpoint[case] = {group: {'loss': end[group]['loss'],
                'delta_from_fork': end[group]['loss'] - record['evaluations'][0][group]['loss'],
                'delta_from_baseline_endpoint': end[group]['loss'] - base_end[group]['loss'] if base_end else None}
                for group in GROUPS}
    summary = {'schema_version': 1, 'status': status, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'directory': str(directory), 'diagnostic_only': True, 'automatic_production_resume': False,
               'production_files_modified': False, 'gpu_used_for_report': False,
               'report_script_sha256': sha(Path(__file__)), 'probes_sha256': probe_sha,
               'cross_arm_controls_passed_for_available_artifacts': True,
               'cross_arm_controls_complete': status == 'complete',
               'source_controls': {key: reference['plan'][key] for key in COMMON} if reference else None,
               'initial_adam_combined_sha256': adam[0] if adam else None, 'initial_rng': rng[0] if rng else None,
               'branches': {case: {key: value for key, value in record.items() if key not in ('plan', 'adam')}
                            for case, record in branches.items()},
               'endpoint_comparison': endpoint, 'snapshot_probe_table': snapshot_table(directory),
               'interpretation_limitations': LIMITS}
    return summary, rows, branches


def render(output, summary, rows, branches):
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 'offset', 'global_step', 'group', 'loss'])
        writer.writeheader()
        writer.writerows(rows)
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), constrained_layout=True)
    for axis, (group, title) in zip(axes.flat, GROUPS.items()):
        for case, record in branches.items():
            points = record['evaluations']
            if points:
                axis.plot([row['offset'] for row in points], [row[group]['loss'] for row in points],
                          marker='o', markersize=5, label=LABELS[case])
        axis.set(title=title, xlabel='从共同检查点继续更新的步数', ylabel='原生 token-weighted loss')
        axis.set_xticks([0, 500, 1000])
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=9)
        else:
            axis.text(.5, .5, '等待真实评估结果', ha='center', transform=axis.transAxes)
    fig.suptitle('原生 CPT 局部对照：共同 step 20000 → 21000\n状态：' + summary['status'])
    fig.savefig(output / 'comparison.png', dpi=180)
    fig.savefig(output / 'comparison.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    output = directory / 'report'
    output.mkdir(parents=True, exist_ok=True)
    try:
        summary, rows, branches = make_report(directory)
        render(output, summary, rows, branches)
    except Exception as exc:
        summary = {'status': 'failed_validation', 'error': str(exc),
                   'diagnostic_only': True, 'created_utc': datetime.now(timezone.utc).isoformat()}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        raise
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(output / 'summary.json')
    if summary['status'] == 'complete':
        result = {'status': 'complete', 'diagnostic_only': True, 'automatic_production_resume': False,
                  'production_files_modified': False, 'completed_utc': summary['created_utc'],
                  'report': str(output / 'summary.json'), 'endpoint_comparison': summary['endpoint_comparison'],
                  'interpretation_limitations': LIMITS}
        result_path = directory / 'result.json'
        temporary = result_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(result_path)
        control_path = directory / 'control.json'
        if control_path.exists():
            control = read(control_path)
            control.update(status='complete', automatic_production_resume=False,
                           completed_utc=summary['created_utc'], report=str(output / 'summary.json'),
                           result=str(result_path), observation={'status': 'complete',
                               'kind': 'native matched 1000-update local interventions',
                               'production_resumed': False})
            temporary = control_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(control, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            temporary.replace(control_path)
    print(json.dumps({'status': summary['status'], 'output': str(output), 'rows': len(rows)}))


if __name__ == '__main__':
    main()
