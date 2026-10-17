import torch
import math
from torch import nn
from .encoder import build_encoder
from .head import build_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.mvlm.language.clip import build_clip_textencoder
from .memory import TargetMemoryBank, extract_target_tokens, crop_boxes_to_image
from .memory_fusion import MemoryFusion


class MVLM(nn.Module):
    def __init__(self, text_encoder, encoder, head,
                 num_frames=1, num_template=1,
                 head_type="CENTER", memory_cfg=None):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.head_type = head_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.head = head

        self.num_frames = num_frames
        self.num_template = num_template
        self.memory_cfg = memory_cfg
        self.memory_enabled = memory_cfg is not None and memory_cfg.ENABLED
        self.freeze_memory_backbone = False
        self.freeze_memory_head = False
        if self.memory_enabled:
            if head_type != 'CENTER':
                raise ValueError('Content memory currently supports the CENTER head only')
            self.memory_fusion = MemoryFusion(
                encoder.num_channels, memory_cfg.HIDDEN_DIM, memory_cfg.KEY_DIM,
                memory_cfg.ROI_SIZE, memory_cfg.USE_SEMANTIC,
                memory_cfg.SHORT_DECAY, memory_cfg.LONG_DECAY)
            if memory_cfg.ROI_SIZE < 1 or memory_cfg.HIDDEN_DIM < 1 or memory_cfg.KEY_DIM < 1:
                raise ValueError('Memory dimensions and ROI size must be positive')
            if memory_cfg.SHORT_CAPACITY < 0 or memory_cfg.LONG_CAPACITY < 0:
                raise ValueError('Memory capacities cannot be negative')
            if memory_cfg.SHORT_CAPACITY + memory_cfg.LONG_CAPACITY == 0:
                raise ValueError('At least one memory store must be enabled')

    def new_memory(self, batch_size):
        if not self.memory_enabled:
            return None
        return TargetMemoryBank(batch_size, self.memory_cfg.SHORT_CAPACITY,
                                self.memory_cfg.LONG_CAPACITY, self.memory_cfg.MIN_QUALITY,
                                self.memory_cfg.NOVELTY_THRESHOLD)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_memory_backbone:
            self.encoder.eval()
            if self.text_encoder is not None:
                self.text_encoder.eval()
        if self.freeze_memory_head:
            self.head.eval()
        return self


    def forward(self, template_list=None, search_list=None, text_emb=None, raw_text=None,
                memory_state=None, frame_ids=None, memory_boxes=None,
                memory_valid=None, predicted_write_probability=0.0,
                memory_box_transforms=None, memory_image_sizes=None):

        auc_dict = {}

        if self.text_encoder is not None:
            if text_emb is None and raw_text is not None:
                text_emb = self.forward_textencoder(raw_text)
                auc_dict.update({"text_emb": text_emb})
        else:
            text_emb = None

        if self.memory_enabled and len(search_list) > 1:
            return self.forward_memory_sequence(template_list, search_list, text_emb, frame_ids,
                                                memory_boxes, memory_valid, predicted_write_probability,
                                                memory_box_transforms, memory_image_sizes)

        enc_dict = self.forward_encoder(template_list, search_list, text_emb)
        head_features = enc_dict['feat_x']
        if self.memory_enabled:
            batch_size = head_features.shape[0]
            if memory_state is None:
                memory_state = self.new_memory(batch_size)
            if frame_ids is None:
                frame_ids = torch.zeros(batch_size, dtype=torch.long, device=head_features.device)
            head_features, memory_gate = self.memory_fusion(head_features, text_emb, memory_state, frame_ids)
            auc_dict.update(feat_x_mem=head_features, memory_gate=memory_gate)
        head_dict = self.forward_head(head_features)
        auc_dict.update(enc_dict)
        auc_dict.update(head_dict)

        return auc_dict

    def forward_memory_sequence(self, template_list, search_list, text_emb, frame_ids,
                                memory_boxes, memory_valid, predicted_write_probability,
                                memory_box_transforms=None, memory_image_sizes=None):
        batch_size = search_list[0].shape[0]
        if frame_ids is None or memory_boxes is None:
            raise ValueError('Memory sequence training requires frame IDs and crop-relative xywh boxes')
        if frame_ids.shape != (len(search_list), batch_size) or memory_boxes.shape != (len(search_list), batch_size, 4):
            raise ValueError('Memory frame IDs and boxes must match [time, batch] dimensions')
        if not 0 <= predicted_write_probability <= 1:
            raise ValueError('Predicted memory write probability must be between zero and one')
        if (memory_box_transforms is None) != (memory_image_sizes is None):
            raise ValueError('Memory box transforms and source image sizes must be supplied together')
        if not bool((frame_ids[1:] > frame_ids[:-1]).all()):
            raise ValueError('Memory sequence frames must be strictly increasing')
        memory_state = self.new_memory(batch_size)
        outputs = []
        for frame_index, search in enumerate(search_list):
            output = self.forward(template_list=template_list, search_list=[search], text_emb=text_emb,
                                  memory_state=memory_state, frame_ids=frame_ids[frame_index])
            outputs.append(output)
            if frame_index == len(search_list) - 1:
                continue
            with torch.no_grad():
                predicted_boxes = output['pred_boxes'][:, 0]
                predicted_xywh = torch.cat((predicted_boxes[:, :2] - predicted_boxes[:, 2:] / 2,
                                            predicted_boxes[:, 2:]), -1)
                use_prediction = torch.rand(batch_size, device=search.device) < predicted_write_probability
                write_boxes = torch.where(use_prediction[:, None], predicted_xywh, memory_boxes[frame_index])
                scores = output['score_map'].flatten(1).max(-1).values
                qualities = torch.where(use_prediction, scores, torch.ones_like(scores))
                tokens, roi_valid = extract_target_tokens(output['feat_x'], write_boxes, self.memory_cfg.ROI_SIZE)
                if memory_valid is not None:
                    roi_valid &= memory_valid[frame_index].bool()
                state_boxes = write_boxes
                if memory_box_transforms is not None:
                    state_boxes = crop_boxes_to_image(write_boxes, memory_box_transforms[frame_index],
                                                       memory_image_sizes[frame_index])
                memory_state.write(tokens, state_boxes, frame_ids[frame_index], qualities, roi_valid)
        return {'memory_sequence': outputs}

    def forward_textencoder(self, raw_text):
        # Forward the encoder
        text_emb = self.text_encoder(raw_text)
        return text_emb

    def forward_encoder(self, template_list, search_list, text_emb):
        enc_dict = self.encoder(template_list, search_list, text_emb)
        return enc_dict

    def forward_head(self, feature, gt_score_map=None):

        bs, HW, C = feature.size()
        if self.head_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.head(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.head(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.head_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.head(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def load_pretrained_tracker(model, pretrained):
    # load state_dict of the pretrained model except the text encoder
    checkpoint = torch.load(pretrained, map_location="cpu")
    state_dict = dict(checkpoint['net'] if 'net' in checkpoint else checkpoint)

    keys_to_remove = [name for name in state_dict if 'text_encoder' in name and not model.memory_enabled]
    for key in keys_to_remove:
        del state_dict[key]

    incompatible = model.load_state_dict(state_dict, strict=False)
    if model.memory_enabled:
        missing_base = [name for name in incompatible.missing_keys
                        if not name.startswith(('memory_fusion.', 'text_encoder.clip.'))]
        if missing_base or incompatible.unexpected_keys:
            raise ValueError('Incompatible memory initialization checkpoint: missing={}, unexpected={}'.format(
                missing_base, incompatible.unexpected_keys))
    return model

def build_mvlm(cfg, training=True):
    encoder = build_encoder(cfg, training=training and cfg.MODEL.PRETRAINED_MODEL is None)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        if cfg.MODEL.ENCODER.LANGUAGE_MODEL == 'CLIP':
            text_encoder = build_clip_textencoder(cfg, encoder) # CLIP
        else:
            print('Non-expected Language Model!')
            exit(-1)
    else:
        text_encoder = None
    head = build_head(cfg, encoder)
    model = MVLM(
        text_encoder,
        encoder,
        head,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        head_type=cfg.MODEL.HEAD.TYPE,
        memory_cfg=cfg.MODEL.MEMORY,
    )
    
    if cfg.MODEL.PRETRAINED_MODEL is not None and (training or not model.memory_enabled):
        model = load_pretrained_tracker(model, cfg.MODEL.PRETRAINED_MODEL)

    if training and model.memory_enabled:
        if cfg.MODEL.PRETRAINED_MODEL is None:
            raise ValueError('Memory training requires MODEL.PRETRAINED_MODEL with baseline tracking weights')
        model.freeze_memory_backbone = True
        model.freeze_memory_head = cfg.TRAIN.MEMORY_FREEZE_HEAD
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        if model.text_encoder is not None:
            for parameter in model.text_encoder.parameters():
                parameter.requires_grad_(False)
        if model.freeze_memory_head:
            for parameter in model.head.parameters():
                parameter.requires_grad_(False)
        model.train()
    
    return model
