from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F
from lib.test.tracker.utils import Preprocessor, sample_target, transform_image_to_crop, resize_sample_target
import cv2
from lib.test.utils.hann import hann2d
from lib.models.mvlm import build_mvlm
from lib.models.mvlm.mvlm import prepare_memory_checkpoint
from lib.utils.box_ops import clip_box
from lib.models.mvlm.memory import extract_target_tokens
import warnings


class MVLM(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MVLM, self).__init__(params)
        network = build_mvlm(params.cfg, training=False)
        ckpt = torch.load(self.params.checkpoint, map_location='cpu')
        state_dict = ckpt['net'] if 'net' in ckpt else ckpt
        state_dict, _ = prepare_memory_checkpoint(
            network, state_dict, allow_baseline=bool(params.cfg.TEST.MEMORY_ALLOW_UNTRAINED))
        incompatible = network.load_state_dict(state_dict, strict=False)
        if network.memory_enabled:
            missing_memory = [name for name in incompatible.missing_keys if name.startswith('memory_fusion.')]
            missing_base = [name for name in incompatible.missing_keys
                            if not name.startswith(('memory_fusion.', 'text_encoder.clip.'))]
            if missing_base or incompatible.unexpected_keys:
                raise ValueError('Incompatible memory tracker checkpoint: missing={}, unexpected={}'.format(
                    missing_base, incompatible.unexpected_keys))
            if missing_memory:
                if not params.cfg.TEST.MEMORY_ALLOW_UNTRAINED:
                    raise ValueError('Checkpoint has no complete trained memory module; use a memory checkpoint '
                                     'or explicitly enable TEST.MEMORY_ALLOW_UNTRAINED for a smoke check')
                warnings.warn('Memory weights are incomplete or untrained; this run is only a smoke check')
        self.cfg = params.cfg
        if hasattr(params, 'device') and params.device:
            self.device = torch.device(params.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = network.to(self.device)
        self.network.eval()
        self.preprocessor = Preprocessor(self.device)
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).to(self.device)

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0
        
        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT

        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT

        # mapping similar datasets
        if 'LASOT' in DATASET_NAME:
            DATASET_NAME = 'LASOT'
        if 'OTB' in DATASET_NAME:
            DATASET_NAME = 'TNL2K'

        # multi modal vision
        if hasattr(self.cfg.TEST.MULTI_MODAL_VISION, DATASET_NAME):
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION[DATASET_NAME]
        else:
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION.DEFAULT

        #multi modal language
        if hasattr(self.cfg.TEST.MULTI_MODAL_LANGUAGE, DATASET_NAME):
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE[DATASET_NAME]
        else:
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT

        #using nlp information
        if hasattr(self.cfg.TEST.USE_NLP, DATASET_NAME):
            self.use_nlp = self.cfg.TEST.USE_NLP[DATASET_NAME]
        else:
            self.use_nlp = self.cfg.TEST.USE_NLP.DEFAULT

        self.prev_conf = 1.0
        
        self.template_free = self.cfg.MODEL.ENCODER.TEMPLATE_FREE
        
        self.omega = self.cfg.TEST.MVLM.OMEGA
        self.lambda_mem = self.cfg.TEST.MVLM.LAMBDA

        self.alpha_corr = self.cfg.TEST.MVLM.ALPHA_CORR
        self.alpha_cls = 1 - self.alpha_corr
        self.iou_thresh = self.cfg.TEST.MVLM.PSI_OUT
        self.tau = self.cfg.TEST.MVLM.TAU
        self.reliable_core = None
        if getattr(self.cfg.TEST, 'RELIABLE_CORE', False):
            from lib.models.mvlm.tracking_core import TrackingCore
            self.reliable_core = TrackingCore(self.network, self.cfg, self.device)

    def initialize(self, image, info: dict):
        if self.reliable_core is not None:
            self.init_nlp = info['init_nlp']
            self.reliable_core.reset(self.init_nlp, info.get('seq_name', 'video'))
            # Unlike the legacy wrapper, return a real prediction for frame zero.
            return self.track(image)
        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and (template.size(1) == 3):
            template = torch.cat((template, template), axis=1)
        self.template_list = [template] * self.num_template

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)]
        self.frame_id = 0
        
        # language information
        if self.multi_modal_language:
            if self.use_nlp:
                self.init_nlp = info['init_nlp']
            else:
                self.init_nlp = None
            with torch.no_grad():
                self.text_emb = self.network.text_encoder(self.init_nlp)
        else:
            self.text_emb = None
            self.init_nlp = None
        
        if self.template_free:
            self.flag = True
        else:
            self.flag = False

        self.kmem = None
        self.kmvlm = None
        self.content_memory = self.network.new_memory(1)

    def set_language(self, text):
        if self.reliable_core is not None:
            if text != self.init_nlp:
                self.init_nlp = text
                self.reliable_core.reset(text, self.reliable_core.state.video_id)
            return
        if text == self.init_nlp:
            return
        self.init_nlp = text
        with torch.no_grad():
            self.text_emb = self.network.text_encoder(text) if text is not None else None
        if self.content_memory is not None:
            self.content_memory.reset()
        self.kmem = None
        self.kmvlm = None
        self.flag = True

    def track(self, image, info: dict = None):
        if self.reliable_core is not None:
            if info is not None and 'init_nlp' in info:
                self.set_language(info['init_nlp'])
            with torch.no_grad():
                _, result = self.reliable_core.step(image)
            self.state = result['target_bbox']
            return result

        H, W, _ = image.shape
        self.frame_id += 1
        if info is not None and 'init_nlp' in info:
            self.set_language(info['init_nlp'])
            
        if self.flag:
            x_patch_arr, resize_factor = resize_sample_target(image, output_sz=self.params.search_size)  # (x1, y1, w, h)
            # crop region covers the entire image
        else:
            x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor, output_sz=self.params.search_size)  # (x1, y1, w, h)
            # compute crop region in original image coordinates
            x, y, w, h = self.state

        search = self.preprocessor.process(x_patch_arr)
        if self.multi_modal_vision and (search.size(1) == 3):
            search = torch.cat((search, search), axis=1)
        search_list = [search]

        # run the encoder
        memory_args = {}
        if self.network.memory_enabled:
            memory_args = dict(memory_state=self.content_memory,
                               frame_ids=torch.tensor([self.frame_id], device=self.device))
        with torch.no_grad():
            out_dict = self.network(template_list=self.template_list,
                                    search_list=search_list,
                                    text_emb=self.text_emb,
                                    raw_text=[self.init_nlp] if self.init_nlp is not None else None,
                                    **memory_args,
                                    )

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True: # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
            
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.head.cal_bboxv2(response, out_dict['size_map'],
                                                                  out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.head.cal_bboxv2(response,
                                                                  out_dict['offset_map'],
                                                                  return_score=True)

        boxes = pred_boxes[0]       # [N, 4] (cx, cy, w, h)
        cls_scores = conf_score[0]  # [N]
        
        # ── VL correlation scores s(I, t, b) for each candidate box ──
        vis_token = torch.cat([out_dict['feat_x']], dim=1)
        feats = batched_slice_mean_from_grid(vis_token, boxes)  # [N, C]

        text_vec = self.text_emb.mean(dim=1).squeeze(0)  # [C]
        feats = F.normalize(feats, p=2, dim=-1)
        text_vec = F.normalize(text_vec, p=2, dim=-1)
        sims = (feats * text_vec.unsqueeze(0)).sum(dim=-1)  # [N]

        # ── Per-box distractor selection (Eq. 4-5 in paper) ──
        # For each box b, find b_tilde_corr(b) and b_tilde_cls(b) from B^out(b)
        distract_corr_scores, distract_cls_scores = select_per_box_distractor(
            boxes, sims, cls_scores, iou_thr=self.iou_thresh
        )  # both [N]

        # ── Per-box standardized margins with tanh (Eq. 4-5) ──
        delta_corr = torch.tanh(sims - distract_corr_scores)          # [N], Eq. (4)
        delta_cls  = torch.tanh(cls_scores - distract_cls_scores)      # [N], Eq. (5)

        # ── Per-box VLM confidence (Eq. 6) ──
        kvlm_per_box = self.alpha_corr * delta_corr + self.alpha_cls * delta_cls  # [N]

        # ── Per-token EWMA temporal memory (Eq. 7-8) ──
        # Grid Omega is fixed across frames → per-token correspondence maintained
        if self.kmem is None:
            self.kmem = kvlm_per_box.clone()   # kappa^mem_0(b) := kappa^vlm_1(b)
        else:
            self.kmem = (1 - self.lambda_mem) * kvlm_per_box + self.lambda_mem * self.kmem  # Eq. (8)

        # ── Per-box MVLM confidence (Eq. 9) ──
        kmvlm_per_box = (1 - self.omega) * kvlm_per_box + self.omega * self.kmem  # [N]

        # ── ROI selection S_t(tau) (Eq. 10) ──
        selected_boxes = kmvlm_per_box >= self.tau

        if torch.any(selected_boxes):
            # |S_t| > 0: select b_hat = argmax_{b in S_t} l_t(b), local search next frame
            cand_idx = torch.nonzero(selected_boxes, as_tuple=False).squeeze(1)
            cand_confs = cls_scores[cand_idx]
            best_rel = torch.argmax(cand_confs)
            top1_idx = int(cand_idx[best_rel].item())
            next_flag = False
        else:
            # |S_t| = 0: select b_hat = argmax_{b in B_t} l_t(b), global re-localization next frame
            top1_idx = int(torch.argmax(cls_scores).item())
            next_flag = True

        top1_box = pred_boxes[0][top1_idx]

        if self.flag:
            box = (top1_box * self.params.search_size).tolist()  # (cx, cy, w, h)
            # resize_sample_target resizes (W, H) → (search_size, search_size)
            # per-axis scale factors needed for non-square images
            if isinstance(resize_factor, (tuple, list)):
                w_scale, h_scale = resize_factor
            else:
                w_scale = self.params.search_size / W
                h_scale = self.params.search_size / H
            cx, cy, w, h = box
            cx_origin = cx / w_scale
            cy_origin = cy / h_scale
            w_origin = w / w_scale
            h_origin = h / h_scale
            pred_box = [cx_origin - 0.5 * w_origin, cy_origin - 0.5 * h_origin, w_origin, h_origin]
            self.state = clip_box(pred_box, H, W, margin=10)
        else:
            box = top1_box.view(-1, 4)
            pred_box = (box.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if self.content_memory is not None:
            with torch.no_grad():
                crop_box = top1_box.detach().unsqueeze(0)
                crop_xywh = torch.cat((crop_box[:, :2] - crop_box[:, 2:] / 2, crop_box[:, 2:]), -1)
                target_tokens, roi_valid = extract_target_tokens(
                    out_dict['feat_x'], crop_xywh, self.cfg.MODEL.MEMORY.ROI_SIZE)
                roi_valid &= selected_boxes[top1_idx].reshape(1)
                full_box = target_tokens.new_tensor([self.state])
                full_box /= full_box.new_tensor([W, H, W, H])
                self.content_memory.write(target_tokens, full_box, [self.frame_id],
                                          cls_scores[top1_idx].reshape(1), roi_valid)

        # LOCAL OR GLOBAL VIEW ON NEXT FRAME?
        if next_flag:
            self.flag = True
        else:
            self.flag = False
        
        if not self.template_free and self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (cls_scores[top1_idx] > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                if self.multi_modal_vision and (template.size(1) == 3):
                    template = torch.cat((template, template), axis=1)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                if len(self.template_anno_list) > self.num_template:
                    self.template_anno_list.pop(1)

        # for debug
        if self.debug:
            image_show = image
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            image_BGR = cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
        else:
            image_BGR = None

        if self.debug:
            # visualize prediction on cropped search image
            cropped_vis = x_patch_arr.copy()
            sz = self.params.search_size
            cx_c, cy_c, w_c, h_c = (top1_box * sz).tolist()
            x1_c = int(cx_c - 0.5 * w_c)
            y1_c = int(cy_c - 0.5 * h_c)
            x2_c = int(cx_c + 0.5 * w_c)
            y2_c = int(cy_c + 0.5 * h_c)
            cv2.rectangle(cropped_vis, (x1_c, y1_c), (x2_c, y2_c), color=(0, 0, 255), thickness=2)
        else:
            cropped_vis = None

        results_dict = {
            "target_bbox": self.state,
            "conf_score": conf_score,
            "image_BGR": image_BGR,
            "cropped_vis": cropped_vis,
        }

        if self.content_memory is not None:
            results_dict['memory_short_size'] = len(self.content_memory.short[0])
            results_dict['memory_long_size'] = len(self.content_memory.long[0])
            results_dict['memory_gate'] = float(out_dict['memory_gate'].mean())

        return results_dict

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


# ─────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────

from lib.models.mvlm.tracking_ops import (
    cxcywh_to_xyxy, pairwise_iou, select_per_box_distractor, batched_slice_mean_from_grid)


def get_tracker_class():
    return MVLM
