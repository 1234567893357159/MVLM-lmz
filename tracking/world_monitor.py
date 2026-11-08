"""Optional W&B and local status reporting for the single-device world trainer."""
import datetime
import json
import os
import time


class WorldMonitor:
    def __init__(self):
        self.output = None
        self.run = None
        self.step = 0

    def status(self, state, **values):
        if self.output is None:
            return
        payload = dict(status=state, completed_optimizer_steps=self.step,
                       pid=os.getpid(), updated_at=datetime.datetime.now(
                           datetime.timezone.utc).isoformat(), **values)
        target = self.output / 'training_status.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(target)

    def start(self, args, config, step):
        from pathlib import Path
        self.output = Path(args.output).resolve()
        self.step = step
        self.started = self.last_tick = time.monotonic()
        self.device = args.device
        self.status('initializing_monitor', planned_optimizer_steps=config.TRAIN.WORLD.MAX_STEPS)
        if args.wandb_project:
            import wandb
            metadata = self.output / 'wandb_run.json'
            previous = json.loads(metadata.read_text()) if args.resume and metadata.exists() else None
            if previous and (previous['project'] != args.wandb_project or previous['entity'] != args.wandb_entity):
                raise ValueError('W&B project/entity changed during resume')
            spec = json.loads((self.output / 'run.json').read_text())
            self.run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.wandb_name or self.output.name,
                id=previous['id'] if previous else None,
                resume='must' if previous else None, mode='online',
                dir=str(self.output), config=dict(training=config, provenance=spec),
                settings=wandb.Settings(init_timeout=90))
            metadata.write_text(json.dumps(dict(id=self.run.id, url=self.run.url,
                project=args.wandb_project, entity=args.wandb_entity), indent=2) + '\n')
            self.run.define_metric('train/step')
            self.run.define_metric('train/*', step_metric='train/step')
            self.run.define_metric('validation/step')
            self.run.define_metric('validation/*', step_metric='validation/step')
            print('W&B training URL: ' + self.run.url, flush=True)
        self.status('running', planned_optimizer_steps=config.TRAIN.WORLD.MAX_STEPS)

    def train(self, row, optimizer, grad_norm):
        import torch
        now = time.monotonic()
        seconds = now - self.last_tick
        self.last_tick = now
        self.step = row['step']
        metrics = {'train/' + k: v for k, v in row.items() if isinstance(v, (int, float))}
        metrics.update({'train/seconds_per_step': seconds,
                        'train/gradient_norm': float(grad_norm),
                        'train/learning_rate': optimizer.param_groups[0]['lr']})
        if torch.device(self.device).type == 'cuda':
            metrics['train/peak_memory_mib'] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        if self.run:
            self.run.log(metrics)
        self.status('running', loss=row.get('loss'), elapsed_seconds=now - self.started)

    def validation(self, row):
        if self.run:
            self.run.log({'validation/' + k: v for k, v in row.items()
                          if isinstance(v, (int, float))})
        # Keep validation time out of the next training-step timing.
        self.last_tick = time.monotonic()

    def finish(self, error=None):
        self.status('failed' if error else 'completed', error=str(error) if error else None)
        if self.run:
            self.run.summary['status'] = 'failed' if error else 'completed'
            self.run.summary['completed_optimizer_steps'] = self.step
            self.run.finish(exit_code=1 if error else 0)
