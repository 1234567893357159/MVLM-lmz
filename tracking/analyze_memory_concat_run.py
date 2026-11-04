"""Paired holdout analysis; never trains a model or selects inference settings."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.analysis.extract_results import calc_seq_err_robust

DATASETS = ('lasot', 'vasttrack', 'tnl2k')
METRICS = ('AUC', 'PRE', 'NPR')
LABELS = {'baseline': 'Baseline', 'concat': 'Concat, step 4000', 'zero': 'Same weights, zero memory'}
COLORS = {'baseline': '#59636e', 'concat': '#2878b5', 'zero': '#de8f05'}


def load(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def aggregate(result):
    return {key: float(np.mean([result['metrics'][ds][key] for ds in DATASETS])) for key in METRICS}


def compare(a, b, manifest, repetitions, seed):
    """Resample paired source groups within each dataset, preserving macro weights."""
    rng = np.random.default_rng(seed)
    estimates, datasets, paired = [], {}, []
    for ds in DATASETS:
        before = {r['name']: r for r in b['per_sequence'][ds]}
        after = {r['name']: r for r in a['per_sequence'][ds]}
        assert before.keys() == after.keys()
        clusters = defaultdict(list)
        for name in sorted(before):
            record = manifest[ds + '/' + name]
            delta = {key: after[name][key] - before[name][key] for key in METRICS}
            paired.append(dict(dataset=ds, name=name, frames=record['frames'], **delta))
            clusters[record['source_group']].append(delta['AUC'])
        sums = np.array([sum(v) for v in clusters.values()])
        counts = np.array([len(v) for v in clusters.values()])
        indices = rng.integers(0, len(sums), size=(repetitions, len(sums)))
        distribution = sums[indices].sum(1) / counts[indices].sum(1)
        estimates.append(distribution)
        values = np.array([r['AUC'] for r in paired if r['dataset'] == ds])
        datasets[ds] = dict(
            delta={key: a['metrics'][ds][key] - b['metrics'][ds][key] for key in METRICS},
            median_delta_AUC=float(np.median(values)), source_groups=len(clusters), videos=len(values),
            improved_over_0_1_pp=int((values > .1).sum()), degraded_over_0_1_pp=int((values < -.1).sum()),
            within_0_1_pp=int((np.abs(values) <= .1).sum()),
            AUC_cluster_bootstrap_95ci=np.quantile(distribution, [.025, .975]).tolist())
    distribution = np.mean(estimates, axis=0)
    av, bv = aggregate(a), aggregate(b)
    return dict(delta={key: av[key] - bv[key] for key in METRICS}, datasets=datasets,
                AUC_cluster_bootstrap_95ci=np.quantile(distribution, [.025, .975]).tolist(),
                bootstrap=dict(repetitions=repetitions, seed=seed, unit='paired source_group',
                               stratification='dataset; datasets retain equal 1/3 weights',
                               limitation='Conditional on this run/holdout; not a repeated-training confidence interval'),
                largest_gains=sorted(paired, key=lambda r: r['AUC'], reverse=True)[:8],
                largest_losses=sorted(paired, key=lambda r: r['AUC'])[:8])


def save_figure(fig, directory, name):
    fig.savefig(directory / (name + '.png'), dpi=170)
    svg = directory / (name + '.svg')
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main(args):
    torch.set_num_threads(1)
    run, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    population = load(run / 'split/validation_manifest.json')['records']
    manifest = {r['key']: r for r in population}
    locations = {'baseline': run / 'validation/baseline_full150',
                 'concat': run / 'validation/step004000_full150'}
    if args.zero_metrics:
        locations['zero'] = args.zero_metrics.resolve().parent
    results = {name: load(path / 'metrics.json') for name, path in locations.items()}
    for result in results.values():
        keys = {ds + '/' + r['name'] for ds, rows in result['per_sequence'].items() for r in rows}
        assert keys == manifest.keys() and result['sequences'] == 150
    if 'zero' in results:
        assert results['zero']['memory_input'] == 'zeroed'
        assert Path(results['zero']['weights']).resolve() == Path(results['concat']['weights']).resolve()
    summary = dict(aggregation='video mean within dataset, then equal mean of three datasets',
                   holdout_videos=150, conditions={name: dict(metrics=r['metrics'], macro=aggregate(r),
                   seconds=r['seconds']) for name, r in results.items()}, comparisons={})
    pairs = [('concat_vs_baseline', 'concat', 'baseline')]
    if 'zero' in results:
        pairs += [('zero_vs_baseline', 'zero', 'baseline'), ('memory_on_vs_zero', 'concat', 'zero')]
    for label, a, b in pairs:
        summary['comparisons'][label] = compare(results[a], results[b], manifest, args.bootstrap, args.seed)

    steps = [load(run / 'validation' / ('step%06d_full150' % step) / 'metrics.json')
             for step in (1000, 2000, 4000)]
    summary['full_validation_checkpoints'] = {str(step): aggregate(result)
        for step, result in zip((1000, 2000, 4000), steps)}
    training = [json.loads(line) for line in (run / 'training_metrics.jsonl').read_text().splitlines()]
    keys = ['train/Loss/Total', 'train/IoU', 'train/Memory/Gate', 'train/Memory/ReadoutToVisualNorm',
            'train/Memory/EmptyFraction', 'train/Memory/ShortEntries', 'train/Memory/LongEntries',
            'train/Memory/WriteAcceptance']
    summary['training_windows'] = {}
    for start, stop in ((0, 200), (200, 1000), (1000, 2000), (2000, 3000), (3000, 4000)):
        rows = [r for r in training if start < r['global_step'] <= stop]
        values = {key: float(np.mean([r[key] for r in rows])) for key in keys}
        values['approx_gate_conditioned_on_nonempty'] = values['train/Memory/Gate'] / (1 - values['train/Memory/EmptyFraction'])
        summary['training_windows'][str(start) + '_' + str(stop)] = values

    indexed = {name: {ds + '/' + r['name']: r for ds, rows in result['per_sequence'].items() for r in rows}
               for name, result in results.items()}
    per_video, frame_rows, cases = [], [], {}
    case_keys = {'tnl2k/CM_baby_done---', 'tnl2k/Terminator_video_02-Done',
                 'tnl2k/GuMuLiYing_video_U01-Done', 'lasot/volleyball-11'}
    occupancy = defaultdict(list)
    for record in population:
        key, ds, name = record['key'], record['dataset'], record['name']
        row = dict(dataset=ds, name=name, frames=record['frames'], source_group=record['source_group'])
        with np.load(record['annotations']) as data:
            boxes, visible = data['boxes'].copy(), data['visible'].copy()
        for condition, location in locations.items():
            for metric in METRICS:
                row[condition + '_' + metric] = indexed[condition][key][metric]
            predictions = np.loadtxt(location / ds / (name + '.txt'), ndmin=2)
            assert predictions.shape == boxes.shape and np.isfinite(predictions).all()
            overlap, center, norm, valid = calc_seq_err_robust(
                torch.tensor(predictions, dtype=torch.float64), torch.tensor(boxes, dtype=torch.float64),
                ds, torch.tensor(visible) if ds == 'lasot' else None)
            valid_iou = overlap[valid]
            assert valid_iou.numel()
            corrected = predictions.copy()
            corrected[0] = boxes[0]  # Match calc_seq_err_robust's first-frame rule.
            pv, gv = corrected[valid.numpy()], boxes[valid.numpy()]
            size_intersection = np.minimum(pv[:, 2:], gv[:, 2:]).prod(1)
            size_iou = size_intersection / (pv[:, 2:].prod(1) + gv[:, 2:].prod(1) - size_intersection)
            center_error = np.abs((pv[:, :2] + .5 * (pv[:, 2:] - 1)) -
                                  (gv[:, :2] + .5 * (gv[:, 2:] - 1)))
            center_intersection = np.maximum(gv[:, 2:] - center_error, 0).prod(1)
            center_iou = center_intersection / (2 * gv[:, 2:].prod(1) - center_intersection)
            thresholds = torch.arange(0, 1.01, .05, dtype=torch.float64)
            frame_rows.append(dict(dataset=ds, name=name, condition=condition, valid_frames=int(valid.sum()),
                valid_only_AUC=float((valid_iou[:, None] > thresholds).double().mean() * 100),
                valid_only_PRE=float((center[valid] <= 20).double().mean() * 100),
                valid_only_NPR=float((norm[valid] <= .2).double().mean() * 100),
                valid_only_mean_IoU=float(valid_iou.mean() * 100),
                normalized_center_error=float(norm[valid].mean()),
                mean_abs_log_size_error=float(np.abs(np.log(pv[:, 2:] / gv[:, 2:])).mean()),
                size_only_IoU=float(size_iou.mean() * 100),
                center_only_IoU=float(center_iou.mean() * 100),
                valid_only_IoU_below_0_1_percent=float((valid_iou < .1).double().mean() * 100)))
            if key in case_keys:
                cases.setdefault(key, {})[condition] = (overlap.numpy(), valid.numpy())
        for label, a, b in pairs:
            row[label + '_AUC'] = row[a + '_AUC'] - row[b + '_AUC']
        telemetry = load(locations['concat'] / ds / (name + '.memory.json'))
        row.update(telemetry)
        occupancy[ds].append(dict(frames=record['frames'], **telemetry))
        per_video.append(row)
    for filename, rows in [('per_video_comparison.csv', per_video), ('valid_frame_metrics.csv', frame_rows)]:
        with (out / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    summary['online_memory'] = {ds: dict(
        video_mean_gate=float(np.mean([r['mean_gate'] for r in rows])),
        frame_weighted_mean_gate=float(np.average([r['mean_gate'] for r in rows], weights=[r['frames'] - 1 for r in rows])),
        videos_reaching_short_capacity_8=sum(r['max_short'] == 8 for r in rows),
        videos_reaching_long_capacity_16=sum(r['max_long'] == 16 for r in rows),
        max_short=max(r['max_short'] for r in rows), max_long=max(r['max_long'] for r in rows))
        for ds, rows in occupancy.items()}
    summary['valid_only_diagnostic'] = {condition: {ds: {metric: float(np.mean([
        row['valid_only_' + metric] for row in frame_rows if row['condition'] == condition and row['dataset'] == ds]))
        for metric in METRICS} for ds in DATASETS} for condition in results}
    summary['valid_only_diagnostic_note'] = ('Additional diagnostic with invalid/invisible frames excluded. '
        'Primary scores retain the original repository protocol; do not mix these scores.')
    geometry_metrics = ('normalized_center_error', 'mean_abs_log_size_error', 'size_only_IoU', 'center_only_IoU')
    summary['geometry_diagnostic'] = {condition: {ds: {metric: float(np.mean([
        row[metric] for row in frame_rows if row['condition'] == condition and row['dataset'] == ds]))
        for metric in geometry_metrics} for ds in DATASETS} for condition in results}
    write(out / 'analysis_summary.json', summary)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    x = [r['global_step'] for r in training]
    for ax, metric, label in zip(axes.flat,
        ('train/Loss/Total', 'train/IoU', 'train/Memory/Gate', 'train/Memory/ReadoutToVisualNorm'),
        ('Training loss', 'Training IoU', 'Memory gate (includes empty frames)', 'Memory / visual feature norm')):
        y = np.array([r[metric] for r in training])
        ax.plot(x, y, alpha=.25, color='#2878b5')
        window = 10
        ax.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, mode='valid'), color='#2878b5')
        ax.axvline(3000, color='gray', linestyle=':', label='LR reduced after 3000')
        ax.set(xlabel='Optimizer step', ylabel=label); ax.grid(alpha=.2)
    save_figure(fig, out, 'training_dynamics')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    for ds in DATASETS:
        ax.plot([0, 1000, 2000, 4000], [0] + [r['metrics'][ds]['AUC'] - results['baseline']['metrics'][ds]['AUC'] for r in steps], marker='o', label=ds)
    ax.axhline(0, color='gray', linestyle='--'); ax.set(xlabel='Step (0 = baseline)', ylabel='Full-holdout AUC change (pp)'); ax.legend(); ax.grid(alpha=.2)
    ax = axes[0, 1]
    xs, width = np.arange(4), .23
    for i, (name, result) in enumerate(results.items()):
        values = [result['metrics'][ds]['AUC'] for ds in DATASETS] + [aggregate(result)['AUC']]
        ax.bar(xs + (i - .5 * (len(results) - 1)) * width, values, width, label=LABELS[name], color=COLORS[name])
    ax.set(xticks=xs, xticklabels=list(DATASETS) + ['macro'], ylabel='Full-holdout AUC', ylim=(70, 84)); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    ax = axes[1, 0]
    for ds in DATASETS:
        rows = [r for r in per_video if r['dataset'] == ds]
        ax.scatter([r['baseline_AUC'] for r in rows], [r['concat_AUC'] for r in rows], s=16, alpha=.65, label=ds)
    ax.plot([0, 100], [0, 100], '--', color='gray'); ax.set(xlabel='Baseline per-video AUC', ylabel='Concat per-video AUC'); ax.legend(); ax.grid(alpha=.2)
    ax = axes[1, 1]
    if 'zero' in results:
        for i, (label, title) in enumerate([('zero_vs_baseline', 'Zero memory - baseline'), ('memory_on_vs_zero', 'Memory on - zero memory')]):
            d = summary['comparisons'][label]
            values = [d['datasets'][ds]['delta']['AUC'] for ds in DATASETS] + [d['delta']['AUC']]
            ax.bar(xs + (i - .5) * .34, values, .34, label=title)
        ax.set(xticks=xs, xticklabels=list(DATASETS) + ['macro'], ylabel='Diagnostic AUC change (pp)'); ax.legend(fontsize=8)
    else:
        ax.hist([r['concat_vs_baseline_AUC'] for r in per_video], bins=25)
        ax.set(xlabel='Per-video AUC change (pp)', ylabel='Videos')
    ax.axhline(0, color='gray', linewidth=.8); ax.grid(axis='y', alpha=.2)
    save_figure(fig, out, 'evaluation_summary')

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (key, conditions) in zip(axes.flat, cases.items()):
        for condition, (overlap, valid) in conditions.items():
            y = overlap.copy(); y[~valid] = np.nan
            ax.plot(np.arange(len(y)), y, label=LABELS[condition], color=COLORS[condition], alpha=.8, linewidth=.9)
        ax.set(title=key, xlabel='Frame', ylabel='IoU', ylim=(-.02, 1.02)); ax.legend(fontsize=7); ax.grid(alpha=.2)
    save_figure(fig, out, 'case_studies')
    print(json.dumps({name: data['macro'] for name, data in summary['conditions'].items()}))
    print(json.dumps({name: {k: v for k, v in data.items() if k in ('delta', 'AUC_cluster_bootstrap_95ci')}
                      for name, data in summary['comparisons'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--zero-metrics', type=Path)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261104)
    main(parser.parse_args())
