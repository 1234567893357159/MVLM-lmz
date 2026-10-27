"""Optional rank-zero W&B logging; never reads or stores API keys itself."""
import json
import hashlib
import os
from pathlib import Path
import subprocess

import torch
import torch.distributed as dist


def is_primary():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get('RANK', '0')) == 0


class WandbLogger:
    def __init__(self, settings):
        self.settings = settings
        self.options = getattr(settings, 'wandb_config', {})
        self.enabled = bool(self.options.get('ENABLED', False))
        self.interval = int(self.options.get('LOG_INTERVAL', 20))
        if self.enabled and self.interval < 1:
            raise ValueError('WANDB.LOG_INTERVAL must be positive')
        self.run = None

    def start(self, net):
        # All ranks enter this method. Propagate initialization errors so no rank
        # enters backward while rank zero has already failed authentication.
        if not self.enabled:
            return
        error = None
        if is_primary() and self.run is None:
            try:
                self._start(net)
            except Exception as exc:
                error = 'W&B initialization failed: {}: {}'.format(type(exc).__name__, exc)
        if dist.is_available() and dist.is_initialized():
            message = [error]
            dist.broadcast_object_list(message, src=0)
            error = message[0]
        if error:
            raise RuntimeError(error + '. Check wandb login or set WANDB.MODE=offline.')

    def _start(self, net):
        import wandb
        folder = Path(self.settings.save_dir) / 'wandb_tracking' / self.settings.project_path
        folder.mkdir(parents=True, exist_ok=True)
        metadata = folder / 'run.json'
        existing = json.loads(metadata.read_text()) if metadata.exists() else {}
        mode = os.environ.get('WANDB_MODE', self.options.get('MODE', 'online'))
        project = os.environ.get('WANDB_PROJECT', self.options.get('PROJECT', 'mvlm-reproduction'))
        entity = os.environ.get('WANDB_ENTITY', self.options.get('ENTITY')) or None
        if existing and (existing['project'] != project or existing['mode'] != mode
                         or existing.get('entity') != entity):
            raise ValueError('W&B destination changed; use a new save_dir for a new run')
        root = Path(__file__).resolve().parents[3]
        config = json.loads(json.dumps(getattr(self.settings, 'training_config', {})))
        config['monitoring'] = {'script': self.settings.script_name,
                                'config_name': self.settings.config_name,
                                'save_dir': str(self.settings.save_dir),
                                'world_size': dist.get_world_size() if dist.is_initialized() else 1}
        for name, args in [('git_commit', ['rev-parse', 'HEAD']),
                           ('git_branch', ['branch', '--show-current'])]:
            try:
                config['monitoring'][name] = subprocess.check_output(
                    ['git'] + args, cwd=str(root), stderr=subprocess.DEVNULL,
                    text=True, timeout=5).strip()
            except (OSError, subprocess.SubprocessError):
                config['monitoring'][name] = 'unknown'
        config['monitoring']['seed'] = getattr(self.settings, 'base_seed', None)
        batch_size = config.get('TRAIN', {}).get('BATCH_SIZE')
        if batch_size is not None:
            config['monitoring']['batch_size_per_rank'] = batch_size
            config['monitoring']['global_batch_size'] = batch_size * config['monitoring']['world_size']
        split = Path(config.get('DATA', {}).get('TRAIN', {}).get('VASTTRACK_SPLIT') or
                     root / 'lib/train/data_specs/vasttrack_train_split.txt')
        if not split.is_absolute():
            split = root / split
        if 'VASTTRACK' in config.get('DATA', {}).get('TRAIN', {}).get('DATASETS_NAME', []) and split.exists():
            config['monitoring']['vasttrack_split_sha256'] = hashlib.sha256(split.read_bytes()).hexdigest()
        manifest = getattr(self.settings, 'custom_manifest', '')
        if manifest:
            config['monitoring']['split_sha256'] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        try:
            config['monitoring']['git_dirty'] = bool(subprocess.check_output(
                ['git', 'status', '--porcelain'], cwd=str(root), text=True, timeout=5).strip())
        except (OSError, subprocess.SubprocessError):
            pass
        model = getattr(net, 'module', net)
        config['monitoring']['parameters_total'] = sum(p.numel() for p in model.parameters())
        config['monitoring']['parameters_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.run = wandb.init(project=project, entity=entity,
                              name=os.environ.get('WANDB_NAME') or self.options.get('NAME') or self.settings.config_name,
                              id=existing.get('id') or wandb.util.generate_id(),
                              resume='allow' if mode == 'online' else None,
                              mode=mode, dir=str(folder), config=config)
        self.run.config.update(config, allow_val_change=True)
        self.run.summary['status'] = 'training'
        self.run.define_metric('global_step')
        self.run.define_metric('*', step_metric='global_step')
        url = self.run.url if mode == 'online' else None
        metadata.write_text(json.dumps({'id': self.run.id, 'url': url, 'project': project,
                                        'entity': entity, 'mode': mode}, indent=2) + '\n')
        print('W&B monitor: {}'.format(url or 'offline (local files: {})'.format(folder)), flush=True)

    def log_metrics(self, metrics, step):
        """Average scalar statistics across ranks; all ranks must call together."""
        if not self.enabled or not metrics:
            return
        keys = sorted(metrics)
        backend = dist.get_backend() if dist.is_initialized() else None
        device = torch.device('cuda', torch.cuda.current_device()) if backend == 'nccl' else torch.device('cpu')
        values = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(values)
            values /= dist.get_world_size()
        if self.run is not None:
            payload = dict(zip(keys, values.cpu().tolist()))
            self.run.log(dict(payload, global_step=step))

    def log_runtime(self, step, epoch, optimizer, batch_size, seconds, device, loader_name):
        if self.run is None:
            return
        metrics = {'global_step': step, 'epoch': epoch,
                   'runtime/' + loader_name + '/seconds_per_batch_rank0': seconds,
                   'runtime/' + loader_name + '/clips_per_second_rank0': batch_size / max(seconds, 1e-9)}
        for index, group in enumerate(optimizer.param_groups):
            metrics['lr/group{}'.format(index)] = group['lr']
        if torch.device(device).type == 'cuda':
            metrics['gpu/allocated_mib_rank0'] = torch.cuda.memory_allocated(device) / 2**20
            metrics['gpu/peak_allocated_mib_rank0'] = torch.cuda.max_memory_allocated(device) / 2**20
        self.run.log(metrics)

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.summary['status'] = 'failed' if exit_code else 'completed'
            self.run.finish(exit_code=exit_code)
            self.run = None
