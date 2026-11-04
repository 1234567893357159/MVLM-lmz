"""Plot measured checkpoint accuracy separately from sampled training IoU."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    root, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    validation = json.loads((root / 'validation_summary.json').read_text())
    data = {'scope': 'same fixed holdout and inference settings for all checkpoints',
            'warning': 'Full150 was not measured at step 3000; joining observed points is not a forecast',
            'lr_decay_after_step': 3000, 'global_clips_per_update': 80, 'frames_per_clip': 4,
            'clip_draws_total': 320000, 'search_frame_draws_total': 1280000,
            'baselines': {}, 'validation': {}, 'training': []}
    for scope in ('full', 'quick'):
        baseline = json.loads((root / 'validation' / ('baseline_' + scope + ('150' if scope == 'full' else '30')) / 'metrics.json').read_text())
        data['baselines'][scope] = {'mean_AUC': baseline['mean_AUC'], 'metrics': baseline['metrics']}
        data['validation'][scope] = [dict(step=int(step), mean_AUC=value['mean_AUC'], metrics=value['metrics'])
            for step, value in sorted(validation[scope].items(), key=lambda pair: int(pair[0]))]
    for line in (root / 'training_metrics.jsonl').read_text().splitlines():
        record = json.loads(line)
        data['training'].append({key: record[key] for key in
            ('global_step', 'train/Loss/Total', 'train/IoU', 'lr/memory', 'lr/center')})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    for ax, scope, title, color in [(axes[0, 0], 'full', 'Full validation: 150 videos', '#2878b5'),
                                   (axes[0, 1], 'quick', 'Quick validation: 30 videos', '#de8f05')]:
        rows = data['validation'][scope]
        x, y = [r['step'] for r in rows], [r['mean_AUC'] for r in rows]
        baseline = data['baselines'][scope]['mean_AUC']
        ax.plot(x, y, marker='o', color=color, label='Measured checkpoint AUC')
        ax.axhline(baseline, color='#59636e', linestyle='--', label='Official-weight control: %.2f' % baseline)
        ax.axvline(3000, color='gray', linestyle=':', alpha=.7)
        for step, value in zip(x, y):
            ax.annotate('%.2f' % value, (step, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=10)
        lo, hi = min(y + [baseline]), max(y + [baseline])
        ax.set(xlim=(700, 4300), ylim=(lo - .15, hi + .2), xticks=(1000, 2000, 3000, 4000),
               xlabel='Optimizer update', ylabel='Mean tracking AUC', title=title)
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc='lower right')
    training = data['training']
    x = [r['global_step'] for r in training]
    for ax, metric, scale, title in [(axes[1, 0], 'train/Loss/Total', 1, 'Training loss on sampled 4-frame clips'),
                                   (axes[1, 1], 'train/IoU', 100, 'Training IoU on sampled clips (not validation AUC)')]:
        y = np.array([r[metric] * scale for r in training])
        ax.plot(x, y, color='#8da5b8', alpha=.35, label='20-update mean')
        ax.plot(x[9:], np.convolve(y, np.ones(10) / 10, mode='valid'), color='#2878b5', label='200-update moving mean')
        ax.axvline(3000, color='gray', linestyle=':', label='Learning rates divided by 10')
        ax.set(xlabel='Optimizer update', ylabel='Loss' if scale == 1 else 'Training IoU (%)', title=title)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Training updates and tracking quality: observed points only', fontsize=14)
    fig.savefig(out / 'steps_vs_accuracy.png', dpi=170)
    svg = out / 'steps_vs_accuracy.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    (out / 'steps_vs_accuracy.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({scope: [(r['step'], r['mean_AUC']) for r in rows]
                      for scope, rows in data['validation'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
