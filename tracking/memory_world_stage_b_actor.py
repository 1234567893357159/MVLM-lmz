"""Batch frozen encodings and differentiable fusion/CENTER across independent B clips."""
import torch
import torch.nn.functional as F

from lib.train.actors.memory_world import MemoryWorldActor
from lib.models.mvlm.temporal_core import (TemporalState, predict_prior_batch,
                                          finish_location, observe)
from lib.models.mvlm.world_state import (search_crop, xywh_to_center, center_to_xywh,
                                        image_to_crop, extract_roi)
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.focal_loss import FocalLoss


class BatchedStageBActor(MemoryWorldActor):
    def __init__(self, model, cfg, stage):
        super().__init__(model, cfg, stage)
        if (stage != 'B' or not model.world_enabled or not model.memory_enabled
                or cfg.TRAIN.WORLD.TRAIN_HEAD or cfg.TRAIN.WORLD.HEAD_ONLY
                or not cfg.TRAIN.WORLD.FREEZE_BN_STATS):
            raise ValueError('Batched B requires joint memory/world, frozen CENTER and BN statistics')

    def frames(self, clips):
        if not clips:
            return
        if len({len(clip['images']) for clip in clips}) != 1:
            raise ValueError('Batched clips must have the same temporal length')
        cfg, model = self.cfg, self.model
        states = [TemporalState(cfg) for _ in clips]
        with torch.no_grad():
            language = model.forward_textencoder([clip['language'] for clip in clips])
        for t in range(len(clips[0]['images'])):
            frames = [clip['frame_ids'][t:t + 1].to(self.device) for clip in clips]
            patches, regions = [], []
            # All reads precede observation writes at this temporal index.
            with torch.no_grad():
                prediction = predict_prior_batch(model, states, language, frames)
                for clip, state in zip(clips, states):
                    patch, region = search_crop(clip['images'][t], cfg.TEST.SEARCH_SIZE,
                                               cfg.TEST.SEARCH_FACTOR,
                                               None if state.global_search else state.previous_box)
                    patches.append(patch)
                    regions.append(region[None].to(self.device))
                search = self.preprocessor.process_batch(patches)
                if cfg.DATA.MULTI_MODAL_VISION:
                    search = torch.cat((search, search), 1)
                raw = model.forward_encoder([], [search], language)['feat_x']
            learning = t >= cfg.TRAIN.WORLD.WARMUP_FRAMES
            with torch.set_grad_enabled(torch.is_grad_enabled() and learning):
                # CENTER's weights are frozen, but its input gradient must reach fusion.
                features, extra = model.fuse_features(raw, language, prediction,
                                                     torch.cat(regions), None, torch.cat(frames))
                head = model.forward_head(features)
            for index, (clip, state, frame) in enumerate(zip(clips, states, frames)):
                prior = None if prediction is None else {key: value[index:index + 1]
                                                         for key, value in prediction.items()}
                result = finish_location(model, state, clip['images'][t], language[index:index + 1],
                                         cfg, frame, patches[index], regions[index], raw[index:index + 1], prior,
                                         {key: value[index:index + 1] for key, value in head.items()},
                                         {key: value[index:index + 1] for key, value in extra.items()})
                yield clip, t, frame, state, result

    def __call__(self, clips, validation=False):
        losses, components = [], {key: [] for key in ('giou_loss', 'l1_loss', 'focal_loss')}
        predictions, active, writes = 0, 0, 0
        cfg, model = self.cfg, self.model
        for clip, t, frame, state, result in self.frames(clips):
            output, prior = result['output'], result['prior']
            gt = clip['boxes'][t:t + 1].to(self.device) / result['scale']
            target = center_to_xywh(image_to_crop(xywh_to_center(gt), result['region']))
            _, roi_valid = extract_roi(output['feat_x'], target)
            center = target[:, :2] + target[:, 2:] / 2
            supervised = (clip['valid'][t:t + 1].to(self.device).bool()
                          & clip['visible'][t:t + 1].to(self.device).bool()
                          & roi_valid & ((center > 0) & (center < 1)).all(-1))
            if supervised.any() and t >= cfg.TRAIN.WORLD.WARMUP_FRAMES:
                predicted = box_cxcywh_to_xyxy(output['pred_boxes'].reshape(-1, 4))
                truth = box_xywh_to_xyxy(target).clamp(0, 1)
                giou, _ = giou_loss(predicted, truth)
                l1 = F.l1_loss(predicted, truth)
                heatmap = generate_heatmap(target[None], cfg.DATA.SEARCH.SIZE,
                                           cfg.MODEL.ENCODER.STRIDE)[0][:, None]
                focal = FocalLoss()(output['score_map'].clamp(1e-6, 1 - 1e-6), heatmap)
                value = cfg.TRAIN.GIOU_WEIGHT * giou + cfg.TRAIN.L1_WEIGHT * l1 + focal
                if validation or value.requires_grad:
                    losses.append(value)
                    for key, term in zip(components, (giou, l1, focal)):
                        components[key].append(float(term.detach()))
            if prior is not None:
                predictions += 1
                active += int(prior['valid'].sum())
            writes += sum(observe(model, state, result, frame))
        metrics = dict(supervised_frames=len(losses), predictions=predictions, active=active,
                       memory_writes=writes, predicted_history_probability=1.)
        metrics.update({key: sum(values) / len(values) for key, values in components.items() if values})
        if not losses:
            return None, metrics
        loss = torch.stack(losses).mean()
        metrics['loss'] = float(loss.detach())
        return loss, metrics
