"""Render the validated hardware-counter capture, without changing training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def derive_metrics(tensor, bandwidth, combined, sm, weights, matplotlib_version=None):
    """Summarize samples; unavailable conditional/constant-series metrics are null.

    Weighted means and fractions retain the capture's sample-interval weights.
    Percentiles and Pearson correlation remain unweighted, as in the original
    report. Empty samples or zero sampling duration do not imply zero activity.
    """
    tensor, bandwidth, combined, sm, weights = [np.asarray(value, dtype=np.float64)
                                               for value in (tensor, bandwidth, combined, sm, weights)]
    arrays = (tensor, bandwidth, combined, sm, weights)
    if any(value.ndim != 1 or len(value) != len(tensor) for value in arrays):
        raise ValueError('Hardware columns must be equally sized one-dimensional arrays')
    if any(not np.isfinite(value).all() for value in arrays) or np.any(weights < 0):
        raise ValueError('Hardware columns must be finite and sample intervals nonnegative')
    total_weight = float(weights.sum())
    high_bw = bandwidth >= 70
    low_tensor = tensor <= 10
    high_bw_count = int(high_bw.sum())
    positive_high_bw_count = int((high_bw & (weights > 0)).sum())

    def weighted_mean(values, intervals):
        return float(np.average(values, weights=intervals)) if len(values) and intervals.sum() > 0 else None

    def fraction(mask):
        return float(weights[mask].sum() / total_weight) if total_weight > 0 else None

    mean_traffic = weighted_mean(combined, weights)
    correlation = None
    if len(tensor) > 1 and np.ptp(tensor) > 0 and np.ptp(bandwidth) > 0:
        correlation = float(np.corrcoef(tensor, bandwidth)[0, 1])
        if not np.isfinite(correlation):
            correlation = None
    if not len(tensor):
        observation = 'No steady-state hardware samples remain after boundary trimming; activity metrics are unavailable'
    elif total_weight <= 0:
        observation = 'Samples have no positive total sampling duration; weighted activity metrics are unavailable'
    elif not high_bw_count:
        observation = 'No sampled interval reaches HBM throughput >=70%; conditional Tensor-pipe activity is unavailable'
    elif not positive_high_bw_count:
        observation = 'Samples with HBM throughput >=70% have zero sampling duration; conditional Tensor-pipe activity is unavailable'
    else:
        observation = 'Some sampled intervals reach HBM throughput >=70%; conditional Tensor-pipe activity is reported separately. These counters alone do not establish that the entire workload is memory-bound'
    return dict(
        sample_count=len(tensor), positive_interval_sample_count=int((weights > 0).sum()),
        hbm_over_70pct_sample_count=high_bw_count,
        hbm_over_70pct_positive_interval_sample_count=positive_high_bw_count,
        hbm_read_write_tb_s=dict(mean=mean_traffic / 1000 if mean_traffic is not None else None,
                                 p90=float(np.percentile(combined, 90) / 1000) if len(combined) else None,
                                 max=float(combined.max() / 1000) if len(combined) else None),
        tensor_hbm_correlation=correlation,
        tensor_under_10pct_time_fraction=fraction(low_tensor),
        hbm_over_70pct_time_fraction=fraction(high_bw),
        tensor_mean_when_hbm_over_70pct=weighted_mean(tensor[high_bw], weights[high_bw]),
        sm_under_50pct_time_fraction=fraction(sm < 50),
        sm_under_10pct_time_fraction=fraction(sm < 10),
        observation=observation, plot_matplotlib_version=matplotlib_version)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    result = json.loads((args.run / 'result.json').read_text())
    with (args.run / 'training.csv').open() as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        headers = reader.fieldnames or []
    rows = [r for r in rows if result['csv_start_boundary'] + 0.55 <= float(r['elapsed_sec']) <= result['csv_end_boundary'] - 0.55]
    def column(prefix):
        key = next((k for k in headers if k.startswith(prefix)), None)
        if key is None:
            raise ValueError(f'Missing hardware column with prefix {prefix!r}')
        return np.array([float(r[key]) for r in rows])
    t = column('elapsed_sec')
    if len(t):
        t -= t[0]
    tensor = column('Tensor pipe')
    bandwidth = column('HBM throughput')
    combined = column('HBM read') + column('HBM write')
    sm = column('SM active')
    weights = column('Sample interval')
    derived = derive_metrics(tensor, bandwidth, combined, sm, weights, matplotlib.__version__)
    (args.run / 'analysis.json').write_text(json.dumps(derived, indent=2, allow_nan=False) + '\n')
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout='constrained')
    axes[0].plot(t, sm, label='SM active', lw=1.0, color='#3674a8')
    axes[0].plot(t, tensor, label='Tensor pipe active', lw=1.0, color='#cb6f23')
    axes[0].set(ylabel='Active cycles (%)', ylim=(0, 105))
    axes[0].legend(loc='upper right', ncol=2)
    axes[1].plot(t, bandwidth, color='#328b55', lw=1.0)
    axes[1].set(ylabel='HBM throughput (%)', ylim=(0, 105))
    axes[2].plot(t, column('HBM read') / 1000, label='HBM read', lw=1, color='#366ca8')
    axes[2].plot(t, column('HBM write') / 1000, label='HBM write', lw=1, color='#cb6f23')
    axes[2].set(ylabel='HBM traffic (TB/s)', xlabel='Steady-state elapsed time (s)', xlim=(0, 4))
    axes[2].legend(loc='upper right', ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"Qwen2.5-Coder-1.5B CPT | GB300 | batch {result['batch']} x 2048 | 20 ms hardware samples")
    fig.savefig(args.run / 'hardware_timeline.png', dpi=180)
    fig.savefig(args.run / 'hardware_timeline.svg')
    print(json.dumps(derived, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
