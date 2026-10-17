import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import _save_tracker_output
from lib.test.evaluation.tracker import Tracker
from lib.train.dataset.otb99_lang import Otb99_lang


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def evaluate(run_dir):
    config_name = 'mvlm_TF_memory_otb_smoke'
    config_path = ROOT / 'experiments/mvlm' / (config_name + '.yaml')
    config = yaml.safe_load(config_path.read_text())
    selected_train = config['DATA']['TRAIN']['OTB_SEQUENCES']
    dataset = Otb99_lang(root=str(ROOT / 'data/otb99_lang'), split='train',
                         multi_modal_language=True, use_nlp=True)
    available_train = list(dataset.sequence_list)
    test_names = ['Biker', 'Bird2', 'Dancer2']
    sequences = [sequence for sequence in get_dataset('otb99_lang') if sequence.name in test_names]
    require(len(sequences) == len(test_names), 'Missing selected evaluation sequences')
    training_files = set()
    training_sequences = []
    for name in selected_train:
        sequence_id = available_train.index(name)
        info = dataset.get_sequence_info(sequence_id)
        require(torch.isfinite(info['bbox']).all() and bool(info['valid'].all()), 'Invalid training boxes: ' + name)
        paths = dataset.get_path(sequence_id, range(len(info['bbox'])))
        require(all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths),
                'Missing training frames: ' + name)
        for frame_index in (0, len(paths) // 2, len(paths) - 1):
            require(cv2.imread(paths[frame_index]) is not None, 'Undecodable training sample')
        training_files.update(str(Path(path).resolve()) for path in paths)
        training_sequences.append({'name': name, 'frames': len(paths), 'language': info['nlp']})
    testing_files = {str(Path(path).resolve()) for sequence in sequences for path in sequence.frames}
    require(not training_files.intersection(testing_files), 'Training and evaluation frames overlap')
    checkpoints = sorted((run_dir / 'checkpoints/train/mvlm' / config_name).glob('MVLM_ep*.pth.tar'))
    require(len(checkpoints) == 1, 'Expected one trained smoke checkpoint, found ' + str(len(checkpoints)))
    checkpoint_path = checkpoints[0]
    baseline_path = ROOT / 'models/MVLM_TF.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    baseline = torch.load(baseline_path, map_location='cpu')['net']
    weights = checkpoint['net']
    require(checkpoint['epoch'] == 1, 'Training did not finish the requested epoch')
    require(all(torch.equal(value, weights[name]) for name, value in baseline.items()),
            'A frozen baseline parameter changed during training')
    output_norm = float(weights['memory_fusion.output_proj.weight'].norm())
    require(output_norm > 0, 'Memory adapter remained at zero initialization')
    optimizer_steps = sorted({int(state['step']) for state in checkpoint['optimizer']['state'].values()})
    require(optimizer_steps == [20], 'Unexpected optimizer update count: ' + str(optimizer_steps))
    summary = {
        'status': 'evaluating', 'purpose': 'small real-data train/evaluate pipeline check; not a convergence experiment',
        'config': config_name, 'train_sequences': training_sequences,
        'train_available_sequences': len(available_train), 'train_eval_frame_overlap': 0,
        'test_sequences': [{'name': sequence.name, 'frames': len(sequence.frames)} for sequence in sequences],
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest(checkpoint_path),
        'baseline_sha256': digest(baseline_path), 'epoch': checkpoint['epoch'],
        'optimizer_steps': optimizer_steps, 'frozen_baseline_unchanged': True,
        'memory_output_weight_norm': output_norm, 'metrics': {},
        'protocol': 'repository language-only tracking, including ground-truth first-frame output',
    }
    shutil.copy2(config_path, run_dir / 'config.yaml')
    (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    del checkpoint, baseline, weights
    gc.collect()
    for label, weight_path, enabled in [('baseline', baseline_path, False), ('memory', checkpoint_path, True)]:
        exp_id = run_dir.name + '_' + label
        wrapper = Tracker('mvlm', config_name, exp_id, 'otb99_lang', str(weight_path), display_name=label)
        params = wrapper.get_parameters()
        params.cfg.MODEL.MEMORY.ENABLED = enabled
        params.cfg.MODEL.PRETRAINED_MODEL = None
        params.debug = False
        params.device = 'cuda:0'
        tracker = wrapper.create_tracker(params)
        diagnostics = []
        for sequence in sequences:
            output_file = Path(wrapper.results_dir) / sequence.dataset / (sequence.name + '.txt')
            require(not output_file.exists(), 'Refusing to overwrite predictions: ' + str(output_file))
            predictions, timings, memory_sizes = [], [], []
            for frame_index, frame_path in enumerate(sequence.frames):
                image = cv2.imread(frame_path)
                require(image is not None, 'Missing evaluation frame: ' + frame_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                started = time.perf_counter()
                if frame_index == 0:
                    tracker.initialize(image, sequence.init_info())
                    prediction = sequence.ground_truth_rect[0].tolist()
                else:
                    result = tracker.track(image)
                    prediction = result['target_bbox']
                    if enabled:
                        memory_sizes.append([result['memory_short_size'], result['memory_long_size']])
                timings.append(time.perf_counter() - started)
                predictions.append(prediction)
            boxes = np.asarray(predictions)
            require(boxes.shape == (len(sequence.frames), 4) and np.isfinite(boxes).all()
                    and (boxes[:, 2:] > 0).all(), 'Invalid predictions: ' + sequence.name)
            _save_tracker_output(sequence, wrapper, {'target_bbox': predictions, 'time': timings})
            diagnostics.append({'sequence': sequence.name, 'frames': len(predictions),
                                'seconds': sum(timings),
                                'max_memory_sizes': np.max(memory_sizes, axis=0).tolist() if memory_sizes else []})
            print(label, sequence.name, 'frames=', len(predictions), 'seconds=', round(sum(timings), 2), flush=True)
        evaluation = extract_results([wrapper], sequences, exp_id)
        overlap = np.asarray(evaluation['ave_success_rate_plot_overlap'])[:, 0].mean(0)
        precision = np.asarray(evaluation['ave_success_rate_plot_center'])[:, 0].mean(0)
        norm_precision = np.asarray(evaluation['ave_success_rate_plot_center_norm'])[:, 0].mean(0)
        summary['metrics'][label] = {'PRE': float(precision[20] * 100),
                                     'NPR': float(norm_precision[20] * 100),
                                     'AUC': float(overlap.mean() * 100),
                                     'exp_id': exp_id, 'sequences': diagnostics}
        (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        del tracker
        gc.collect()
        torch.cuda.empty_cache()
    summary['status'] = 'completed'
    (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    report = ['# OTB 记忆模块小规模训练与评测检查', '',
              '目的：验证真实数据加载、参数更新、checkpoint 保存／重载、在线跟踪和指标计算。',
              '这不是收敛训练或完整 OTB 基准结果，短训后的指标不能证明模块有益或无益。', '',
              '训练：Basketball、Bolt、Boy、Car4；1 epoch，20 次更新，每批 2 个四帧片段。',
              '评测：Biker、Bird2、Dancer2；与训练序列无图像交集。',
              '冻结基线权重逐项相同，记忆输出投影已从零初始化更新。', '',
              '| 模型 | PRE | NPR | AUC |', '| --- | ---: | ---: | ---: |']
    for label, metrics in summary['metrics'].items():
        report.append('| {} | {:.2f} | {:.2f} | {:.2f} |'.format(label, metrics['PRE'], metrics['NPR'], metrics['AUC']))
    report += ['', '配置、权重哈希、预测位置与逐序列运行统计见同目录 `summary.json`。',
               '训练日志见 `train.log`，评测日志见 `evaluate.log`。']
    (run_dir / '结果说明.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(summary['metrics'], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    evaluate(args.run_dir.resolve())


if __name__ == '__main__':
    main()
