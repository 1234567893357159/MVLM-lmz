"""Freeze sequence/group-level custom splits without moving source datasets."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.evaluation.data import Sequence
from lib.train.dataset.otb99_lang import Otb99_lang


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out, seed):
    if out.exists():
        raise RuntimeError('Refusing to overwrite split: ' + str(out))
    out.mkdir(parents=True)
    records, excluded = [], []
    for dataset in ['otb99_lang', 'lasot', 'tnl2k']:
        sequences = list(get_dataset(dataset))
        if dataset == 'otb99_lang':
            train = Otb99_lang('/root/MVLM/data/otb99_lang', split='train',
                              multi_modal_language=True, use_nlp=True)
            for i, name in enumerate(train.sequence_list):
                info = train.get_sequence_info(i)
                sequences.append(Sequence(name, train.get_path(i, range(len(info['bbox']))),
                                          dataset, info['bbox'].numpy(), language_query=info['nlp']))
            official_train = set(train.sequence_list)
        else:
            official_train = set()
        for seq in sorted(sequences, key=lambda s: s.name):
            if not seq.language_query.strip():
                excluded.append({'dataset': dataset, 'name': seq.name, 'reason': 'empty language'})
                continue
            boxes = np.asarray(seq.ground_truth_rect, dtype=np.float32)
            assert boxes.shape == (len(seq.frames), 4) and np.isfinite(boxes).all(), seq.name
            visible = np.ones(len(boxes), dtype=bool) if seq.target_visible is None else np.asarray(seq.target_visible).astype(bool)
            visible &= (boxes[:, 2:] > 0).all(1)
            assert visible.shape == (len(boxes),), seq.name
            for path in seq.frames:
                assert Path(path).is_file() and Path(path).stat().st_size > 0, path
            fingerprints = []
            for i in sorted({0, len(boxes)//2, len(boxes)-1}):
                assert cv2.imread(seq.frames[i]) is not None, seq.frames[i]
                fingerprints.append(digest(seq.frames[i]))
            key = dataset + '/' + seq.name
            folder = out / 'sequences' / dataset / seq.name
            folder.mkdir(parents=True)
            np.savez_compressed(folder / 'annotations.npz', boxes=boxes, visible=visible)
            (folder / 'frames.txt').write_text('\n'.join(seq.frames) + '\n')
            clean = re.sub(r'(?i)[_-]done$', '', seq.name)
            source_group = re.sub(r'(?i)(?:[_-]video)?[_-][a-z]?\d+$', '', clean) if dataset == 'tnl2k' else seq.name
            # Different OTB target annotations from the same image folder share a group.
            if dataset == 'otb99_lang': source_group = str(Path(seq.frames[0]).parent)
            records.append(dict(key=key, dataset=dataset, name=seq.name, frames=len(boxes),
                                visible_frames=int(visible.sum()), language=seq.language_query,
                                category=seq.object_class or source_group,
                                source_group=dataset + '/' + source_group,
                                sampled_frame_hashes=fingerprints,
                                official_otb_train=seq.name in official_train,
                                annotations=str(folder/'annotations.npz'), frame_list=str(folder/'frames.txt')))
        print('Indexed', dataset, 'records:', sum(r['dataset']==dataset for r in records), flush=True)
    parent = {r['key']:r['key'] for r in records}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    def union(a,b): parent[find(a)] = find(b)
    seen = {}
    for r in records:
        for token in [r['source_group']] + ['image:'+h for h in r['sampled_frame_hashes']]:
            if token in seen: union(r['key'],seen[token])
            seen[token]=r['key']
    groups=defaultdict(list)
    for r in records: groups[find(r['key'])].append(r)
    targets={'otb99_lang':{'train':51,'val':12,'test':36},
             'lasot':{'train':196,'val':42,'test':42},
             'tnl2k':{'train':489,'val':105,'test':105}}
    counts={d:{s:0 for s in ['train','val','test']} for d in targets}
    rng=random.Random(seed)
    remaining=[]
    for members in groups.values():
        if any(r['official_otb_train'] for r in members):
            split='train'
            for r in members: r['split']=split; counts[r['dataset']][split]+=1
        else: remaining.append(members)
    rng.shuffle(remaining)
    # Largest same-source groups first; never split a group to force exact counts.
    remaining.sort(key=len, reverse=True)
    for members in remaining:
        possible=['val','test'] if any(r['dataset']=='otb99_lang' for r in members) else ['train','val','test']
        per=defaultdict(int)
        for r in members: per[r['dataset']]+=1
        def cost(split):
            score=0.
            for d,n in per.items():
                for s in ['train','val','test']:
                    after=counts[d][s]+(n if s==split else 0)
                    score+=(after-targets[d][s])**2/max(targets[d][s],1)
            return score
        split=min(possible,key=cost)
        for r in members: r['split']=split; counts[r['dataset']][split]+=1
    for split in ['train','val','test']:
        paths={p for r in records if r['split']==split for p in Path(r['frame_list']).read_text().splitlines()}
        for other in ['train','val','test']:
            if other>=split: continue
            other_paths={p for r in records if r['split']==other for p in Path(r['frame_list']).read_text().splitlines()}
            assert not paths & other_paths, (split,other)
    for r in records:
        if r['split']=='train': assert r['visible_frames'] > 2*(12+1), r['key']
        r['group']=find(r['key'])
    manifest={'seed':seed,'protocol':'custom split of already evaluated datasets; not standard benchmark reproduction',
              'grouping':'OTB source image folder; conservative TNL2K name families; identical sampled image SHA256 across datasets',
              'limitations':'Three image fingerprints do not prove absence of all duplicate or near-duplicate footage; class/length balance reported, not guaranteed.',
              'counts':counts,'excluded':excluded,'records':records}
    path=out/'manifest.json'
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    sha=digest(path)
    (out/'manifest.sha256').write_text(sha+'\n')
    for d in counts:
        for split in ['train','val','test']:
            (out/(d+'_'+split+'.txt')).write_text('\n'.join(r['name'] for r in records if r['dataset']==d and r['split']==split)+'\n')
    print(json.dumps({'manifest':str(path),'sha256':sha,'counts':counts,'excluded':excluded},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--seed',type=int,default=20261018)
    args=p.parse_args(); prepare(args.out,args.seed)
