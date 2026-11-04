"""Append completed, locally archived diagnostics to the original W&B run."""
import argparse
import json
from pathlib import Path

import wandb


def main(args):
    experiment = args.run.resolve()
    analysis = args.analysis.resolve()
    data = json.loads((analysis / 'analysis_summary.json').read_text())
    spec = json.loads((experiment / 'run_spec.json').read_text())
    meta = spec['wandb']
    directory = analysis.parent / 'wandb_analysis'
    directory.mkdir(exist_ok=True)
    run = wandb.init(project=meta['project'], entity=meta['entity'], id=meta['id'],
                     name=experiment.name, resume='allow', mode='online', dir=str(directory),
                     settings=wandb.Settings(init_timeout=45))
    run.define_metric('global_step')
    run.define_metric('analysis/*', step_metric='global_step')
    payload = {'global_step': 4000}
    for condition, result in data['conditions'].items():
        for metric, value in result['macro'].items():
            payload['analysis/' + condition + '/' + metric] = value
    for comparison, result in data['comparisons'].items():
        for metric, value in result['delta'].items():
            payload['analysis/' + comparison + '/' + metric + '_delta_pp'] = value
        for suffix, value in zip(('low', 'high'), result['AUC_cluster_bootstrap_95ci']):
            payload['analysis/' + comparison + '/AUC_95ci_' + suffix] = value
    for name in ('evaluation_summary', 'training_dynamics', 'case_studies'):
        payload['analysis/figures/' + name] = wandb.Image(str(analysis / (name + '.png')))
    run.log(payload)
    for name in ('analysis_summary.json', 'per_video_comparison.csv', 'valid_frame_metrics.csv'):
        run.save(str(analysis / name), base_path=str(experiment), policy='now')
    run.summary['analysis/status'] = 'completed'
    run.summary['analysis/report'] = 'reports/reproduction/MEMORY_CONCAT_ANALYSIS_20261104.md'
    # Keep the completed training summary separate from diagnostic namespaces.
    run.summary.update(json.loads((experiment / 'wandb_summary.json').read_text()))
    run.finish(exit_code=0)
    (analysis.parent / 'wandb_analysis_upload_status.json').write_text(json.dumps(dict(
        status='upload_finished', sdk_exit_code=0, run_id=meta['id'], url=meta['url'],
        server_confirmation='Check analysis_wandb_verified.json; SDK return alone is not proof'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    main(parser.parse_args())
