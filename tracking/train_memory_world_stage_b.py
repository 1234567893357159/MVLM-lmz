"""Stage-B entry using batched independent clips, preserving the live stage-A source."""
import json
from pathlib import Path
import shutil
import torch
import torch.distributed as dist

import train_memory_world as trainer
from memory_world_stage_b_actor import BatchedStageBActor
from world_monitor import WorldMonitor
from world_distributed import WorldDistributed
from world_manifest_fast import read_manifest


class StageBDistributed(WorldDistributed):
    def metrics(self, values):
        result = super().metrics(values)
        keys = ('giou_loss', 'l1_loss', 'focal_loss')
        count = values.get('supervised_frames', 0)
        tensor = torch.tensor([values.get(key, 0.) * count for key in keys] + [count],
                              dtype=torch.float64, device=self.device)
        if self.size > 1:
            dist.all_reduce(tensor)
        if tensor[-1].item():
            result.update({key: tensor[index].item() / tensor[-1].item() for index, key in enumerate(keys)})
        return result


class StageBMonitor(WorldMonitor):
    def start(self, args, config, step):
        root = Path(__file__).resolve().parents[1]
        output = Path(args.output).resolve()
        spec = json.loads((output / 'run.json').read_text())
        if args.resume:
            if spec.get('batch_execution') != 'independent-stage-b-batched':
                raise ValueError('Resume requires the same batched stage-B entry')
        else:
            parent = torch.load(args.world_checkpoint, map_location='cpu')
            spec['parent_checkpoint_step'] = parent['step']
            del parent
            for relative in ('tracking/train_memory_world_stage_b.py',
                             'tracking/memory_world_stage_b_actor.py',
                             'tracking/world_manifest_fast.py'):
                destination = output / 'source' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(root / relative), str(destination))
                spec['source_hashes'][relative] = trainer.sha256(root / relative)
            spec['batch_execution'] = 'independent-stage-b-batched'
            (output / 'run.json').write_text(json.dumps(spec, indent=2))
        super().start(args, config, step)


if __name__ == '__main__':
    trainer.MemoryWorldActor = BatchedStageBActor
    trainer.monitor = StageBMonitor()
    trainer.parallel = StageBDistributed()
    trainer.read_manifest = read_manifest
    try:
        trainer.main()
    except BaseException as error:
        trainer.monitor.finish(error)
        raise
    else:
        trainer.monitor.finish()
    finally:
        trainer.parallel.close()
