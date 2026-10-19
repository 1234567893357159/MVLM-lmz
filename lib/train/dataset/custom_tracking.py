"""Explicit custom split adapter; never substitutes a benchmark's official train split."""
import json
from pathlib import Path

import numpy as np
import torch
from .base_video_dataset import BaseVideoDataset


class CustomTrackingDataset(BaseVideoDataset):
    def __init__(self, manifest_path, dataset_name, image_loader):
        super().__init__(dataset_name, str(Path(manifest_path).parent), image_loader)
        manifest = json.loads(Path(manifest_path).read_text())
        self.records = [r for r in manifest['records'] if r['dataset'] == dataset_name and r['split'] == 'train']
        if not self.records:
            raise ValueError('Empty custom training split: ' + dataset_name)
        self.sequence_list = [r['name'] for r in self.records]
        self._cache = {}

    def get_name(self):
        return self.name

    def _record(self, seq_id):
        if seq_id not in self._cache:
            record = self.records[seq_id]
            with np.load(record['annotations']) as a:
                boxes = torch.from_numpy(a['boxes'].copy())
                visible = torch.from_numpy(a['visible'].copy()).bool()
            paths = Path(record['frame_list']).read_text().splitlines()
            if boxes.shape != (len(paths), 4) or not record['language'].strip():
                raise ValueError('Invalid frozen custom sequence: ' + record['key'])
            self._cache[seq_id] = (boxes, visible, paths)
        return self._cache[seq_id]

    def get_sequence_info(self, seq_id):
        boxes, visible, _ = self._record(seq_id)
        valid = torch.isfinite(boxes).all(1) & (boxes[:,2:] > 0).all(1)
        return {'bbox': boxes, 'valid': valid, 'visible': visible & valid,
                'nlp': self.records[seq_id]['language']}

    def get_frames(self, seq_id, frame_ids, anno=None):
        _, _, paths = self._record(seq_id)
        frames=[]
        for i in frame_ids:
            image=self.image_loader(paths[i])
            if image is None: raise ValueError('Undecodable frame: ' + paths[i])
            frames.append(np.concatenate((image,image),axis=-1))
        anno=self.get_sequence_info(seq_id) if anno is None else anno
        selected={k:([v for _ in frame_ids] if k=='nlp' else [v[i].clone() for i in frame_ids]) for k,v in anno.items()}
        return frames,selected,{'object_class_name':self.records[seq_id]['category']}
