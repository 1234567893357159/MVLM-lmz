"""Read-only data/checkpoint audit for the completed custom experiment."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

RUN=Path('/root/MVLM/output/memory_three_custom_20261018_run01')
OUT=RUN/'diagnostics';OUT.mkdir(exist_ok=True)
manifest=json.loads((RUN/'split/manifest.json').read_text());rng=random.Random(20261019)
report={'counts':manifest['counts'],'excluded':manifest['excluded'],'numeric_errors':[],
        'decode_errors':[],'decoded_images':0,'invalid_box_frames':0,'all_frames':0,'group_overlap':[],
        'limitations':['Images are sampled, not all decoded; semantic correctness of text/boxes is not automatically established.']}
paths={s:set() for s in ['train','val','test']};groups=defaultdict(set)
for record in manifest['records']:
    with np.load(record['annotations']) as data:
        boxes=data['boxes'];visible=data['visible']
    images=Path(record['frame_list']).read_text().splitlines()
    if boxes.shape!=(len(images),4) or not np.isfinite(boxes).all() or not record['language'].strip():report['numeric_errors'].append(record['key'])
    report['all_frames']+=len(images);report['invalid_box_frames']+=int((boxes[:,2:]<=0).any(1).sum())
    paths[record['split']].update(images);groups[record['group']].add(record['split'])
# Fixed random image checks from every sequence; decoding does not change dataset files.
for record in manifest['records']:
    images=Path(record['frame_list']).read_text().splitlines()
    ids=sorted({0,len(images)//2,len(images)-1,rng.randrange(len(images))})
    for i in ids:
        image=cv2.imread(images[i]);report['decoded_images']+=1
        if image is None:report['decode_errors'].append(images[i])
report['path_overlap']={a+'_'+b:len(paths[a]&paths[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
report['group_overlap']=[g for g,ss in groups.items() if len(ss)>1]
checkpoint=RUN/'checkpoints/train/mvlm/mvlm_TF_memory_three_custom/MVLM_ep0020.pth.tar'
c=torch.load(checkpoint,map_location='cpu');base=torch.load('/root/MVLM/models/MVLM_TF.pth.tar',map_location='cpu')['net']
report['checkpoint']={'global_step':c['global_step'],
 'frozen_weights_changed':sum(not torch.equal(v,c['net'][k]) for k,v in base.items()),
 'nonfinite_weight_tensors':[k for k,v in c['net'].items() if v.is_floating_point() and not bool(torch.isfinite(v).all())],
 'optimizer_memory_states':{k:{'step':int(v['step']),'first_moment_norm':float(v['exp_avg'].norm())} for k,v in c['optimizer']['state'].items()}}
(OUT/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('data and checkpoint audit done',flush=True)
archive=Path('/root/MVLM/downloads/tnl2k.tar.gz')
if archive.exists():
    h=hashlib.sha256();started=time.time()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
    # Hugging Face mirror LFS metadata recorded during original download.
    expected='01335bc1a30577735921cc30388e9a33c678f9b4401e89cd33a79887df69b88c'
    result={'archive':str(archive),'sha256':h.hexdigest(),'expected_mirror_sha256':expected,
            'matches':h.hexdigest()==expected,'seconds':time.time()-started,
            'note':'Mirror archive integrity, not an official source equivalence guarantee.'}
    (OUT/'tnl2k_archive_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
