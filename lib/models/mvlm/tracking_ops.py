"""Legacy candidate operations shared without importing the online tracker."""
import torch

def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)


def pairwise_iou(boxes):
    """Compute NxN pairwise IoU matrix for boxes in (cx,cy,w,h) format.

    Args:
        boxes: [N, 4] tensor in (cx, cy, w, h) format
    Returns:
        iou_matrix: [N, N] tensor
    """
    xyxy = cxcywh_to_xyxy(boxes)  # [N, 4]
    x1 = torch.maximum(xyxy[:, None, 0], xyxy[None, :, 0])  # [N, N]
    y1 = torch.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    x2 = torch.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
    y2 = torch.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # [N, N]
    area = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0) * (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)  # [N]
    union = area[:, None] + area[None, :] - inter + 1e-12
    return inter / union


def select_per_box_distractor(boxes, corr_scores, cls_scores, iou_thr=0.3):
    """For each box b, find its strongest distractor from B^out(b).

    Paper Sec 3.3:
        B^out_t(b) = { b' in B_t : IoU(b', b) <= psi_out }
        b_tilde_corr(b) = argmax_{b' in B^out(b)} s(I, t, b')
        b_tilde_cls(b)  = argmax_{b' in B^out(b)} l_t(b')

    Args:
        boxes:       [N, 4] candidate boxes (cx, cy, w, h)
        corr_scores: [N]    VL correlation scores s(I, t, b)
        cls_scores:  [N]    classification scores l_t(b)
        iou_thr:     float  psi_out threshold
    Returns:
        distract_corr_scores: [N]  s(I, t, b_tilde_corr(b)) for each b
        distract_cls_scores:  [N]  l_t(b_tilde_cls(b)) for each b
    """
    N = boxes.size(0)
    iou_matrix = pairwise_iou(boxes)  # [N, N]

    # B^out(b) mask: IoU(b', b) <= psi_out, excluding self
    outside_mask = iou_matrix <= iou_thr  # [N, N]
    outside_mask.fill_diagonal_(False)    # exclude b itself

    # For correlation: find max s(I,t,b') within B^out(b) for each b
    corr_expanded = corr_scores.unsqueeze(0).expand(N, -1)  # [N, N] — row i = scores of all b'
    corr_masked = corr_expanded.clone()
    corr_masked[~outside_mask] = -float('inf')
    distract_corr_scores = corr_masked.max(dim=1).values  # [N]

    # For classification: find max l_t(b') within B^out(b) for each b
    cls_expanded = cls_scores.unsqueeze(0).expand(N, -1)  # [N, N]
    cls_masked = cls_expanded.clone()
    cls_masked[~outside_mask] = -float('inf')
    distract_cls_scores = cls_masked.max(dim=1).values  # [N]

    # Fallback: if B^out(b) is empty for some b, use min score as distractor
    no_outside_corr = (corr_masked > -float('inf')).sum(dim=1) == 0
    no_outside_cls  = (cls_masked  > -float('inf')).sum(dim=1) == 0
    if no_outside_corr.any():
        distract_corr_scores[no_outside_corr] = corr_scores.min()
    if no_outside_cls.any():
        distract_cls_scores[no_outside_cls] = cls_scores.min()

    return distract_corr_scores, distract_cls_scores


def batched_slice_mean_from_grid(vis_token, boxes_cxcywh):
    if vis_token.dim() == 3:
        vis_token = vis_token[0]
    L, C = vis_token.shape
    H = W = int(L ** 0.5)
    feat = vis_token.view(H, W, C)

    Scale = (H - 1)
    cx, cy, w, h = boxes_cxcywh.unbind(-1)

    x1 = torch.round((cx - 0.5 * w) * Scale).to(torch.int64)
    y1 = torch.round((cy - 0.5 * h) * Scale).to(torch.int64)
    x2 = torch.round((cx + 0.5 * w) * Scale).to(torch.int64)
    y2 = torch.round((cy + 0.5 * h) * Scale).to(torch.int64)

    x1 = x1.clamp(min=0)
    y1 = y1.clamp(min=0)
    x2 = torch.maximum(x2, x1 + 1)
    y2 = torch.maximum(y2, y1 + 1)

    x1 = x1.clamp(max=W - 1)
    y1 = y1.clamp(max=H - 1)
    x2 = x2.clamp(max=W)
    y2 = y2.clamp(max=H)

    zero_pad = feat.new_zeros((1, W, C))
    S = torch.cat([zero_pad, feat.cumsum(dim=0)], dim=0)  # [H+1, W, C]
    zero_pad2 = S.new_zeros((H + 1, 1, C))
    S = torch.cat([zero_pad2, S.cumsum(dim=1)], dim=1)  # [H+1, W+1, C]

    x1i = x1 + 1
    y1i = y1 + 1
    x2i = x2
    y2i = y2

    rect_sum = S[y2i, x2i] - S[y1i, x2i] - S[y2i, x1i] + S[y1i, x1i]
    area = (y2 - y1) * (x2 - x1)
    area = area.clamp(min=1).unsqueeze(1)

    feats = rect_sum / area
    return feats
