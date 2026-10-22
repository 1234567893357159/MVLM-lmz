"""Shared causal rollout for predictor pretraining and frozen-predictor adaptation."""
import torch
import torch.nn.functional as F

from lib.models.mvlm.world_state import (WorldHistory, search_crop, extract_roi,
    xywh_to_center, center_to_xywh, crop_to_image, image_to_crop, motion_target)
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss, clip_box
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.focal_loss import FocalLoss
from lib.test.tracker.utils import Preprocessor


@torch.no_grad()
def select_observation(model, output, language, kmem, cfg):
    """Same candidate confidence/margin/EWMA rules as the original MVLM tracker."""
    from lib.test.tracker.mvlm import batched_slice_mean_from_grid, select_per_box_distractor
    from lib.test.utils.hann import hann2d
    response = output['score_map']
    if cfg.TEST.WINDOW:
        side = response.shape[-1]
        window = hann2d(torch.tensor([side, side]), centered=True).to(response)
        response = response * window
    boxes, scores = model.head.cal_bboxv2(response, output['size_map'],
                                          output['offset_map'], return_score=True)
    boxes, scores = boxes[0], scores[0]
    visual = F.normalize(batched_slice_mean_from_grid(output['feat_x'], boxes), dim=-1)
    text = F.normalize(language.mean(1)[0], dim=-1)
    correlations = (visual * text).sum(-1)
    parameters = cfg.TEST.MVLM
    distract_visual, distract_class = select_per_box_distractor(
        boxes, correlations, scores, parameters.PSI_OUT)
    confidence = (parameters.ALPHA_CORR * (correlations - distract_visual).tanh()
                  + (1 - parameters.ALPHA_CORR) * (scores - distract_class).tanh())
    kmem = confidence if kmem is None else ((1 - parameters.LAMBDA) * confidence
                                           + parameters.LAMBDA * kmem)
    reliable = ((1 - parameters.OMEGA) * confidence + parameters.OMEGA * kmem) >= parameters.TAU
    candidates = reliable.nonzero().flatten()
    index = candidates[scores[candidates].argmax()] if len(candidates) else scores.argmax()
    return boxes[index:index + 1], scores[index:index + 1].clamp(0, 1), reliable[index:index + 1], kmem


def tracking_loss(output, target, cfg):
    # Supervision is crop-relative xywh, predictions are crop-relative cxcywh.
    predicted = box_cxcywh_to_xyxy(output['pred_boxes'].reshape(-1, 4))
    gt = box_xywh_to_xyxy(target).clamp(0, 1)
    giou, _ = giou_loss(predicted, gt)
    l1 = F.l1_loss(predicted, gt)
    heatmap = generate_heatmap(target[None], cfg.DATA.SEARCH.SIZE,
                               cfg.MODEL.ENCODER.STRIDE)[0][:, None]
    focal = FocalLoss()(output['score_map'].clamp(1e-6, 1 - 1e-6), heatmap)
    return cfg.TRAIN.GIOU_WEIGHT * giou + cfg.TRAIN.L1_WEIGHT * l1 + focal


class WorldModelActor:
    def __init__(self, model, cfg, stage):
        self.model, self.cfg, self.stage = model, cfg, stage
        self.device = next(model.parameters()).device
        self.preprocessor = Preprocessor(self.device)
        self.step = 0

    def __call__(self, clips, validation=False):
        cfg, model = self.cfg, self.model
        losses, appearance_errors, motion_errors, copy_errors, still_errors, velocity_errors = [], [], [], [], [], []
        active, predictions = 0, 0
        ramp = cfg.TRAIN.WORLD.HISTORY_RAMP_STEPS
        history_probability = cfg.TRAIN.WORLD.PREDICTED_HISTORY_PROB * (min(1., self.step / ramp) if ramp > 0 else 1.)
        for clip in clips:
            history = WorldHistory(cfg.MODEL.WORLD.HISTORY)
            with torch.no_grad():
                language = model.forward_textencoder([clip['language']])
            previous_state, kmem = None, None
            global_search = True
            for t, image in enumerate(clip['images']):
                height, width = image.shape[:2]
                scale = torch.tensor([width, height, width, height], device=self.device)
                frame = clip['frame_ids'][t:t + 1].to(self.device)
                packed = history.pack()
                prediction = None
                if packed is not None:
                    # In stage B the predictor is frozen; never build its autograd graph.
                    if self.stage == 'B':
                        with torch.no_grad():
                            prediction = model.world_model(packed, language, frame)
                    else:
                        prediction = model.world_model(packed, language, frame)
                    predictions += 1
                patch, region = search_crop(image, cfg.DATA.SEARCH.SIZE, cfg.TEST.SEARCH_FACTOR,
                                             None if global_search else previous_state)
                region = region[None].to(self.device)
                search = self.preprocessor.process(patch)
                if cfg.DATA.MULTI_MODAL_VISION:
                    search = torch.cat((search, search), 1)
                with torch.no_grad():
                    encoded = model.forward_encoder([], [search], language)
                raw = encoded['feat_x']
                fused = raw
                if prediction is not None and self.stage == 'B':
                    fused = model.imagination_fusion(raw, prediction, region)
                    active += int(prediction['valid'].sum())
                # Head stays differentiable in stage B. Its parameters are frozen in A.
                if self.stage == 'A':
                    with torch.no_grad():
                        output = model.forward_head(fused)
                else:
                    output = model.forward_head(fused)
                output['feat_x'] = raw
                selected, quality, reliable, kmem = select_observation(model, output, language, kmem, cfg)
                full_predicted = crop_to_image(selected, region)
                pixel_box = clip_box((center_to_xywh(full_predicted)[0] * scale).tolist(),
                                     height, width, margin=10)
                full_predicted = xywh_to_center(raw.new_tensor(pixel_box)[None] / scale)
                gt_box = clip['boxes'][t:t + 1].to(self.device) / scale
                gt_center = xywh_to_center(gt_box)
                gt_valid = clip['valid'][t:t + 1].to(self.device)
                visible = clip['visible'][t:t + 1].to(self.device)
                crop_gt = center_to_xywh(image_to_crop(gt_center, region))
                target_tokens, target_roi_valid = extract_roi(raw, crop_gt)

                if prediction is not None:
                    if self.stage == 'A':
                        valid_geometry = gt_valid & packed['valid'][:, -1]
                        valid_appearance = valid_geometry & visible & target_roi_valid
                        target_delta = motion_target(packed['boxes'][:, -1], gt_center)
                        # Select valid rows BEFORE loss operations: masked NaNs must not reach backward.
                        if valid_geometry.any():
                            motion = F.smooth_l1_loss(prediction['delta'][valid_geometry],
                                                      target_delta[valid_geometry])
                            motion_errors.append(motion)
                            still_errors.append(F.smooth_l1_loss(torch.zeros_like(target_delta[valid_geometry]),
                                                                 target_delta[valid_geometry]))
                            elapsed = (frame - packed['frame_ids'][:, -1]).float()[:, None]
                            dt = (packed['frame_ids'][:, -1] - packed['frame_ids'][:, -2]).float()[:, None]
                            constant_velocity = motion_target(packed['boxes'][:, -2], packed['boxes'][:, -1]) * elapsed / dt
                            velocity_errors.append(F.smooth_l1_loss(constant_velocity[valid_geometry], target_delta[valid_geometry]))
                            loss = cfg.TRAIN.WORLD.MOTION_WEIGHT * motion
                            if valid_appearance.any():
                                predicted_tokens = prediction['tokens'][valid_appearance]
                                targets = target_tokens[valid_appearance]
                                cosine = (1 - F.cosine_similarity(predicted_tokens, targets, dim=-1)).mean()
                                regression = F.smooth_l1_loss(F.layer_norm(predicted_tokens, (raw.shape[-1],)),
                                                              F.layer_norm(targets, (raw.shape[-1],)))
                                appearance = cosine + cfg.TRAIN.WORLD.FEATURE_WEIGHT * regression
                                loss = loss + cfg.TRAIN.WORLD.APPEARANCE_WEIGHT * appearance
                                appearance_errors.append(cosine)
                                copy_errors.append((1 - F.cosine_similarity(
                                    packed['tokens'][:, -1][valid_appearance], targets, dim=-1)).mean())
                            losses.append(loss)
                    else:
                        center = crop_gt[:, :2] + crop_gt[:, 2:] / 2
                        supervised = gt_valid & visible & target_roi_valid & ((center > 0) & (center < 1)).all(-1)
                        if supervised.any():
                            losses.append(tracking_loss(output, crop_gt, cfg))

                # The current observation is appended only AFTER predicting/supervising this frame.
                teacher = (self.stage == 'A' and not validation
                           and torch.rand((), device=self.device).item() >= history_probability)
                if teacher and bool((gt_valid & visible & target_roi_valid).all()):
                    observation, observed_box = target_tokens, gt_center
                    observed_quality, observed_valid = torch.ones_like(quality), gt_valid & visible & target_roi_valid
                    previous_state = (gt_box[0] * scale).tolist()
                    global_search = False
                else:
                    observed_box = full_predicted
                    observation, observed_valid = extract_roi(
                        raw, center_to_xywh(image_to_crop(observed_box, region)))
                    observed_valid &= reliable
                    observed_quality = quality
                    previous_state = pixel_box
                    global_search = not bool(reliable.all())
                history.append(observation, observed_box, observed_quality, observed_valid, frame)
        if not losses:
            return None, dict(supervised_frames=0, predictions=predictions, active=active)
        loss = torch.stack(losses).mean()
        metrics = dict(loss=float(loss.detach()), supervised_frames=len(losses), predictions=predictions, active=active, predicted_history_probability=history_probability)
        for name, values in [('appearance_cosine_error', appearance_errors), ('motion_error', motion_errors),
                             ('copy_cosine_error', copy_errors), ('stationary_motion_error', still_errors),
                             ('constant_velocity_error', velocity_errors)]:
            if values:
                metrics[name] = float(torch.stack(values).mean().detach())
        return loss, metrics
