import torch
import math
from torch import nn
from .encoder import build_encoder
from .head import build_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.mvlm.language.clip import build_clip_textencoder


class MVLM(nn.Module):
    def __init__(self, text_encoder, encoder, head,
                 num_frames=1, num_template=1,
                 head_type="CENTER", cfg=None):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.head_type = head_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.head = head

        self.cfg = cfg
        self.memory_enabled = cfg is not None and cfg.MODEL.MEMORY.ENABLED
        self.world_enabled = cfg is not None and cfg.MODEL.WORLD.ENABLED
        self.temporal_enabled = self.memory_enabled or self.world_enabled or (cfg is not None and cfg.MODEL.TEMPORAL_BASELINE)
        self.world_stage = None
        if self.temporal_enabled:
            from .memory_reader import MemoryReader
            from .future_fusion import FutureFusion
            from .world_model import WorldModel
            from .quality_head import CandidateQualityHead
            if head_type != 'CENTER' or num_template != 1:
                raise ValueError('Temporal tracking requires CENTER and one template')
            def initialized(factory, offset):
                # Match shared-module initialization across M/W ablations without
                # changing the caller's sampling RNG or any CUDA RNG state.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(int(cfg.TRAIN.WORLD.SEED) + offset)
                    return factory()

            channels = encoder.num_channels
            hidden = cfg.MODEL.FUSION.CHANNELS
            self.future_fusion = initialized(lambda: FutureFusion(channels, hidden, cfg.MODEL.WORLD.HEADS,
                cfg.MODEL.FUSION.MODE, cfg.MODEL.WORLD.FUSION_STRENGTH, cfg.MODEL.WORLD.MIN_QUALITY), 1)
            self.memory_reader = initialized(lambda: MemoryReader(channels, cfg.MODEL.MEMORY), 2) if self.memory_enabled else None
            if self.memory_enabled and cfg.MODEL.MEMORY.HIDDEN_DIM != hidden:
                raise ValueError('Memory readout and fusion channel dimensions must match')
            self.world_model = initialized(lambda: WorldModel(channels, cfg.MODEL.WORLD.HISTORY,
                cfg.MODEL.WORLD.HIDDEN, cfg.MODEL.WORLD.LAYERS, cfg.MODEL.WORLD.HEADS,
                cfg.MODEL.MEMORY.HIDDEN_DIM, cfg.MODEL.WORLD.QUALITY_ENABLED), 3) if self.world_enabled else None
            self.quality_head = initialized(lambda: CandidateQualityHead(cfg.MODEL.WORLD.HIDDEN), 4) if (
                self.world_enabled and cfg.MODEL.WORLD.QUALITY_ENABLED) else None

        self.num_frames = num_frames
        self.num_template = num_template


    def forward(self, template_list=None, search_list=None, text_emb=None, raw_text=None,
                world_prediction=None, crop_regions=None, memory_state=None, frame_ids=None):

        auc_dict = {}

        if self.text_encoder is not None:
            if text_emb is None and raw_text is not None:
                text_emb = self.forward_textencoder(raw_text)
                auc_dict.update({"text_emb": text_emb})
        else:
            text_emb = None

        enc_dict = self.forward_encoder(template_list, search_list, text_emb)
        features = enc_dict['feat_x']
        if self.temporal_enabled:
            features, extra = self.fuse_features(features, text_emb, world_prediction,
                                                  crop_regions, memory_state, frame_ids)
            auc_dict.update(extra)
        head_dict = self.forward_head(features)
        auc_dict.update(enc_dict)
        auc_dict.update(head_dict)

        return auc_dict

    def fuse_features(self, features, language, prediction=None, regions=None, bank=None, frame_ids=None):
        extra = {}
        if self.world_enabled:
            if self.world_stage == 'A' or self.cfg.TRAIN.WORLD.HEAD_ONLY:
                fused, reliability = self.future_fusion.absent(features), features.new_zeros(features.shape[0])
            else:
                fused, reliability = self.future_fusion(features, prediction, regions)
            if prediction is not None and self.quality_head is not None:
                extra['quality_logits'] = self.quality_head(prediction, regions, self.fx_sz)
        elif bank is not None:
            context = self.memory_reader(features, language, bank, frame_ids)
            fused = self.future_fusion.from_memory(features, context)
            reliability = context['valid'].to(features)
        else:
            fused, reliability = self.future_fusion.absent(features), features.new_zeros(features.shape[0])
        extra.update(feat_x_temporal=fused, prediction_reliability=reliability)
        return fused, extra

    def configure_world_stage(self, stage):
        if not self.temporal_enabled or stage not in ('A', 'B', 'C'):
            raise ValueError('Temporal configuration and stage A/B/C required')
        if stage == 'A' and not self.world_enabled:
            raise ValueError('Memory-only adaptation uses stage B or C')
        if self.cfg.MODEL.FUSION.MODE == 'concat' and stage != 'A' and not self.cfg.TRAIN.WORLD.TRAIN_HEAD:
            raise ValueError('Concat adaptation must train the expanded CENTER inputs')
        self.world_stage = stage
        self.requires_grad_(False)
        if self.cfg.TRAIN.WORLD.HEAD_ONLY:
            if stage != 'B' or not self.cfg.TRAIN.WORLD.TRAIN_HEAD or self.cfg.MODEL.FUSION.MODE != 'cross_attention':
                raise ValueError('HEAD_ONLY requires stage B, TRAIN_HEAD and the original CENTER shape')
            modules = []
        elif stage == 'A':
            modules = [self.memory_reader, self.world_model, self.quality_head]
        elif stage == 'B':
            modules = [self.future_fusion]
            if not self.world_enabled:
                modules.append(self.memory_reader)
        else:
            modules = [self.memory_reader, self.world_model, self.future_fusion, self.quality_head]
        if stage != 'A' and self.cfg.TRAIN.WORLD.TRAIN_HEAD:
            modules.append(self.head)
        for module in modules:
            if module is not None:
                module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.world_stage is not None:
            self.encoder.eval()
            if self.text_encoder is not None:
                self.text_encoder.eval()
            for module in (self.head, self.memory_reader, self.world_model, self.quality_head, self.future_fusion):
                if module is not None and not any(p.requires_grad for p in module.parameters()):
                    module.eval()
            if self.cfg.TRAIN.WORLD.FREEZE_BN_STATS:
                for module in self.head.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        return self

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
    state_dict = checkpoint['net']

    keys_to_remove = [k for k in state_dict.keys() if 'text_encoder' in k]
    for key in keys_to_remove:
        del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    return model

def build_mvlm(cfg, training=True):
    if cfg.MODEL.TEMPORAL_BASELINE and (cfg.MODEL.MEMORY.ENABLED or cfg.MODEL.WORLD.ENABLED or cfg.MODEL.FUSION.MODE != 'cross_attention'):
        raise ValueError('Geometry-only baseline requires disabled memory/world and the original head shape')
    if cfg.MODEL.MEMORY.ENABLED or cfg.MODEL.WORLD.ENABLED or cfg.MODEL.TEMPORAL_BASELINE:
        if not cfg.MODEL.ENCODER.TEMPLATE_FREE or not cfg.DATA.MULTI_MODAL_LANGUAGE:
            raise ValueError('Temporal tracking requires template-free language tracking')
        if cfg.DATA.SEARCH.SIZE != cfg.TEST.SEARCH_SIZE or cfg.MODEL.MEMORY.ROI_SIZE != 2:
            raise ValueError('Temporal tracking requires identical search grids and 2x2 ROI tokens')
        if min(cfg.MODEL.MEMORY.SHORT_CAPACITY, cfg.MODEL.MEMORY.LONG_CAPACITY) < 0 or (cfg.MODEL.MEMORY.ENABLED and cfg.MODEL.MEMORY.SHORT_CAPACITY + cfg.MODEL.MEMORY.LONG_CAPACITY == 0):
            raise ValueError('Invalid content memory capacities')
        if not 0 <= cfg.MODEL.WORLD.QUALITY_WEIGHT <= 1:
            raise ValueError('QUALITY_WEIGHT must be between zero and one')
        if cfg.MODEL.WORLD.QUALITY_ENABLED and not cfg.MODEL.WORLD.ENABLED:
            raise ValueError('Quality prior requires a world model')
    encoder = build_encoder(cfg, training=training)
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
        cfg=cfg,
    )
    
    if cfg.MODEL.PRETRAINED_MODEL is not None:
        if model.temporal_enabled:
            from .joint_checkpoint import load_baseline
            load_baseline(model, torch.load(cfg.MODEL.PRETRAINED_MODEL, map_location='cpu'))
        else:
            model = load_pretrained_tracker(model, cfg.MODEL.PRETRAINED_MODEL)
    
    return model
