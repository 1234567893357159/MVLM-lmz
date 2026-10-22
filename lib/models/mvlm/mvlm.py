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
                 head_type="CENTER", world_cfg=None):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.head_type = head_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.head = head

        self.world_model = None
        self.imagination_fusion = None
        self.world_stage = None
        if world_cfg is not None and world_cfg.ENABLED:
            if head_type != 'CENTER' or num_template != 1:
                raise ValueError('World model currently supports CENTER and one template only')
            from .world_model import WorldModel
            from .imagination_fusion import ImaginationFusion
            self.world_model = WorldModel(encoder.num_channels, world_cfg.HISTORY,
                                          world_cfg.HIDDEN, world_cfg.LAYERS, world_cfg.HEADS)
            self.imagination_fusion = ImaginationFusion(
                encoder.num_channels, world_cfg.HIDDEN, world_cfg.HEADS,
                world_cfg.FUSION_STRENGTH, world_cfg.MIN_QUALITY)

        self.num_frames = num_frames
        self.num_template = num_template


    def forward(self, template_list=None, search_list=None, text_emb=None, raw_text=None, world_prediction=None, crop_regions=None):

        auc_dict = {}

        if self.text_encoder is not None:
            if text_emb is None and raw_text is not None:
                text_emb = self.forward_textencoder(raw_text)
                auc_dict.update({"text_emb": text_emb})
        else:
            text_emb = None

        enc_dict = self.forward_encoder(template_list, search_list, text_emb)
        features = enc_dict['feat_x']
        if world_prediction is not None:
            if self.imagination_fusion is None or crop_regions is None:
                raise ValueError('World prediction requires enabled fusion and crop coordinates')
            features = self.imagination_fusion(features, world_prediction, crop_regions)
            auc_dict['feat_x_world'] = features
        head_dict = self.forward_head(features)
        auc_dict.update(enc_dict)
        auc_dict.update(head_dict)

        return auc_dict

    def configure_world_stage(self, stage):
        if self.world_model is None or stage not in ('A', 'B'):
            raise ValueError('Enabled world model and stage A/B required')
        self.world_stage = stage
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        modules = [self.world_model] if stage == 'A' else [self.imagination_fusion, self.head]
        for module in modules:
            module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.world_stage is not None:
            self.encoder.eval()
            if self.text_encoder is not None:
                self.text_encoder.eval()
            if self.world_stage == 'A':
                self.head.eval()
                self.imagination_fusion.eval()
            else:
                self.world_model.eval()
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
    if cfg.MODEL.WORLD.ENABLED:
        if not cfg.MODEL.ENCODER.TEMPLATE_FREE or not cfg.DATA.MULTI_MODAL_LANGUAGE:
            raise ValueError('World model requires template-free, language-conditioned MVLM')
        if cfg.DATA.SEARCH.SIZE != cfg.TEST.SEARCH_SIZE:
            raise ValueError('World training and inference require identical search feature grids')
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
        world_cfg=cfg.MODEL.WORLD,
    )
    
    if cfg.MODEL.PRETRAINED_MODEL is not None:
        model = load_pretrained_tracker(model, cfg.MODEL.PRETRAINED_MODEL)
    
    return model
