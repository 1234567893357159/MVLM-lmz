"""Prediction-driven actor with ground truth confined to the loss path."""
import torch
from lib.models.mvlm.tracking_core import TrackingCore
from lib.utils.box_ops import giou_loss, box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap


def frame_loss(output, transform, box_image, valid, visible, known_absent, cfg):
    score = output['score_map']
    zero = score.sum()*0
    stats = dict(supervised=0, regression=0, outside=0, partial=0, background=0)
    if not valid and not known_absent:
        return zero, stats
    target = torch.zeros_like(score)
    regression = zero
    stats['supervised'] = 1
    if known_absent or not visible:
        stats['background'] = 1
    else:
        box = transform.to_crop(box_image.to(score.device).float())
        corners = box_xywh_to_xyxy(box)
        intersection = bool((corners[2:] > 0).all() and (corners[:2] < 1).all())
        if not intersection:
            stats['outside'] = 1
            stats['background'] = 1
        else:
            fully_inside = bool((corners[:2] >= 0).all() and (corners[2:] <= 1).all())
            center = box[:2] + box[2:]/2
            h, w = score.shape[-2:]
            if bool((center >= 0).all() and (center < 1).all()):
                target = generate_heatmap(box[None, None].detach().cpu(), cfg.DATA.SEARCH.SIZE,
                                          cfg.MODEL.ENCODER.STRIDE)[0][:, None].to(score.device)
                x = int((center[0]*w).round().clamp(0, w-1))
                y = int((center[1]*h).round().clamp(0, h-1))
                target[0, 0, y, x] = 1.
                if fully_inside:
                    size = output['size_map'][0, :, y, x]
                    offset = output['offset_map'][0, :, y, x]
                    predicted_center = (offset + offset.new_tensor([x, y]))/offset.new_tensor([w, h])
                    predicted = box_cxcywh_to_xyxy(torch.cat((predicted_center, size))[None])
                    giou, _ = giou_loss(predicted, corners[None])
                    regression = cfg.TRAIN.GIOU_WEIGHT*giou + cfg.TRAIN.L1_WEIGHT*torch.nn.functional.l1_loss(predicted, corners[None])
                    stats['regression'] = 1
            if not fully_inside:
                stats['partial'] = 1
    # Clamp both ends (the original FocalLoss only clamps its lower endpoint).
    loss = regression + FocalLoss()(score.clamp(1e-6, 1-1e-6), target)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite supervised loss')
    return loss, stats


class ReliableMemoryActor:
    def __init__(self, model, cfg, device):
        self.model, self.cfg, self.device = model, cfg, device
        if not 0 <= cfg.TRAIN.WARMUP_FRAMES < cfg.TRAIN.CLIP_FRAMES:
            raise ValueError('Warmup must be shorter than the clip')

    def supervision_count(self, clips):
        count = 0
        for clip in clips:
            known = clip['valid'] | clip['known_absent']
            count += int(known[self.cfg.TRAIN.WARMUP_FRAMES:].sum())
        return count

    def __call__(self, clips, backward=False, normalization=None):
        count = self.supervision_count(clips)
        normalization = max(1, count) if normalization is None else normalization
        stats = dict(supervised=0, regression=0, outside=0, partial=0, background=0,
                     committed=0, pending=0, rejected=0, loss=0.)
        results, losses = [], []
        for clip in clips:
            core = TrackingCore(self.model, self.cfg, self.device)
            core.reset(clip['language'], clip['name'])
            if len(clip['images']) != self.cfg.TRAIN.CLIP_FRAMES:
                raise ValueError('Unexpected clip length')
            for t, image in enumerate(clip['images']):
                supervise = t >= self.cfg.TRAIN.WARMUP_FRAMES
                with torch.set_grad_enabled(supervise):
                    output, result = core.step(image, clip['frame_ids'][t])
                    if supervise:
                        loss, row = frame_loss(output, core.transform, clip['boxes'][t],
                                               bool(clip['valid'][t]), bool(clip['visible'][t]),
                                               bool(clip['known_absent'][t]), self.cfg)
                        for k, value in row.items():
                            stats[k] += value
                        stats['loss'] += float(loss.detach()) / normalization
                        if backward:
                            (loss / normalization).backward()
                        else:
                            losses.append(loss/normalization)
                action = result['diagnostics']['write']['action']
                key = {'commit': 'committed', 'pending': 'pending', 'reject': 'rejected'}.get(action)
                if key:
                    stats[key] += 1
                results.append(result)
        return (None if backward else sum(losses)), stats, results
