"""
LATEX model
"""

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.backbone import build_backbone
from models.matcher import build_matcher
from models.transformer import build_transformer_decoder
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)
import utils.boxOps as boxOps
from einops import repeat
from torchvision.ops import roi_align

logger = logging.getLogger(__name__)

# class ModuleCat(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.dims = dims
#         self.ln1 = nn.Linear(dims, dims // 2)
#         self.ln2 = nn.Linear(dims, dims // 2)
        
#     def forward(self, x1, x2):
#         x1 = self.ln1(x1)
#         x2 = self.ln2(x2)
#         return torch.cat([x1, x2], dim=-1)

class HDLayout(nn.Module):
    def __init__(self, args):
        super(HDLayout, self).__init__()
        self.bs = args.batch_size
        self.aux_loss = args.aux_loss
        self.num_queries = args.num_queries  # [num_block_queries, num_line_queries, num_bezier_queries]
        self.backbone = build_backbone(args)

        # decoder
        self.line_decoder_1 = build_transformer_decoder(args, dec_layers=2, return_intermediate=True)
        self.line_decoder_2 = build_transformer_decoder(args, dec_layers=2, return_intermediate=True)

        hidden_dim = args.hidden_dim
        # project backbone channels -> hidden_dim
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        # block encoder (从 block bbox -> hidden)
        self.block_decoder = nn.Sequential(
            nn.Linear(4, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim)
        )

        # queries embedding
        self.query_embed_line1 = nn.Embedding(self.num_queries[1], hidden_dim)
        self.query_embed_line2 = nn.Embedding(self.num_queries[2], hidden_dim)

        # heads
        # self.block_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.line1_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.line2_bezier_embed = MLP(hidden_dim, hidden_dim, 16, 3)

        # self.block_prob_embed = nn.Linear(hidden_dim, 1)
        self.line1_prob_embed = nn.Linear(hidden_dim, 1)
        self.line2_prob_embed = nn.Linear(hidden_dim, 1)

        # keep ModuleCat (if you had gating/fusion)
        # self.cat_block = ModuleCat(hidden_dim)
        # self.cat_line1 = ModuleCat(hidden_dim)

    def forward(self, samples: dict):
        """
        说明（假设）:
         - samples['imgs'] : NestedTensor with .tensors [B,3,H_img,W_img]
         - samples['block_bboxes'] : tensor [B, num_block, 4] 格式为绝对像素坐标 xyxy (x1,y1,x2,y2)
        返回:
         - out dict 包含 pred_line1 / pred_line2 / probs，以及 aux_outputs（若 aux_loss=True）
        """
        samples_img = samples['imgs']
        samples_block_bbox = samples['block_bboxes']  # [B, num_block, 4] (xyxy, pixels)

        if isinstance(samples_img, (list, torch.Tensor)):
            samples_img = nested_tensor_from_tensor_list(samples_img)

        # backbone
        features, pos = self.backbone(samples_img)
        src, mask = features[-1].decompose()  # src: [B, C, Hf, Wf]
        assert mask is not None

        # project to hidden dim
        src_proj = self.input_proj(src)  # [B, hidden, Hf, Wf]
        B, hidden_dim, Hf, Wf = src_proj.shape

        # prepare pos for transformer usage (if needed)
        # pos_embed = pos[-1].flatten(2).permute(2, 0, 1)  # [Hf*Wf, B, C_pos]
        # mask_flat = mask.flatten(1)  # [B, Hf*Wf]

        # ---------------- block encoding ----------------

        block_bboxes = samples_block_bbox.unsqueeze(1)  # [B, num_block, 4]
        B, num_block, _ = block_bboxes.shape

        # block decoder: bbox -> hidden
        hs_block = self.block_decoder(block_bboxes)  # [B, num_block, hidden_dim]

        block_query = hs_block  # [B, num_block, hidden_dim]

        # ---------------- RoIAlign: 为每个 block 裁剪特征 ----------------

        img_tensor = samples_img.tensors
        _, _, H_img, W_img = img_tensor.shape
        spatial_scale_x = float(Wf) / float(W_img)
        spatial_scale_y = float(Hf) / float(H_img)

        spatial_scale = (spatial_scale_x + spatial_scale_y) / 2.0

        # build rois list: [total_rois, 5]
        rois = []
        for b_idx in range(B):
            # block_bboxes[b_idx]: [num_block, 4]
            bboxes = block_bboxes[b_idx]  # (num_block, 4)
            batch_idx_col = torch.full((bboxes.size(0), 1), float(b_idx), device=bboxes.device)
            rois_b = torch.cat([batch_idx_col, bboxes], dim=1)  # (num_block,5)
            rois.append(rois_b)
        rois = torch.cat(rois, dim=0)  # (B*num_block, 5)

        ROI_OUT = (8, 8)

        block_roi_feats = roi_align(src_proj, rois, output_size=ROI_OUT, spatial_scale=spatial_scale)  # [B*num_block, hidden, Hroi, Wroi]

        # reshape to [B, num_block, hidden, Hroi, Wroi]
        block_roi_feats = block_roi_feats.view(B, num_block, hidden_dim, ROI_OUT[0], ROI_OUT[1])

        S = ROI_OUT[0] * ROI_OUT[1]
        memory_line1 = block_roi_feats.view(B * num_block, hidden_dim, S).permute(2, 0, 1)  # [S, B*num_block, hidden]

        # ---------------- line1: queries 派生自 block_query ----------------
        # block_query: [B, num_block, hidden]
        num_line_q = self.num_queries[1]

        tgt_line1 = block_query.unsqueeze(2).repeat(1, 1, num_line_q, 1)  # [B, num_block, Nq, hidden]

        qpos = self.query_embed_line1.weight  # [Nq, hidden]

        qpos_exp = qpos.unsqueeze(0).unsqueeze(0).repeat(B, num_block, 1, 1) # [B, num_block, Nq, hidden]

        Bstar = B * num_block
        tgt_line1 = tgt_line1.view(Bstar, num_line_q, hidden_dim).permute(1, 0, 2)  # [Nq, Bstar, hidden]
        query_pos_line1 = qpos_exp.view(Bstar, num_line_q, hidden_dim).permute(1, 0, 2)  # [Nq, Bstar, hidden]

        mem_key_padding = None

        # pos for memory (optional) - we don't have ROI positional embedding; pass None or create small pos
        pos_for_line1 = None

        # ----------------- decode line1 -----------------
        hs_line1 = self.line_decoder_1(tgt_line1, memory_line1,
                                      memory_key_padding_mask=mem_key_padding,
                                      pos=pos_for_line1, query_pos=query_pos_line1)  # -> [L, Nq, Bstar, H]
        # make shape consistent with original code: transpose to [L, Bstar, Nq, H]
        hs_line1 = hs_line1.permute(0, 2, 1, 3)

        # decode bbox + prob
        outputs_line1 = self.line1_bbox_embed(hs_line1).sigmoid()  # [L, Bstar, Nq, 4] (normalized [0,1] relative to ROI??)
        # NOTE: currently outputs are sigmoid; 若想把它转成 absolute pixel coords w.r.t. image, 需要将 ROI->image mapping 做 inverse transform。
        line1_prob = torch.sigmoid(self.line1_prob_embed(hs_line1[-1]))  # [Bstar, Nq, 1]

        # reshape outputs back to [B, num_block, Nq, ...]
        L = outputs_line1.shape[0]
        outputs_line1 = outputs_line1.view(L, B, num_block, num_line_q, -1)
        line1_prob = line1_prob.view(B, num_block, num_line_q, -1)

        hs_line1_last = hs_line1[-1].view(Bstar, num_line_q, hidden_dim)  # [Bstar, Nq, H]
        # reshape为 [B, num_block, Nq, H]
        hs_line1_last_bb = hs_line1_last.view(B, num_block, num_line_q, hidden_dim)

        memory_line2 = memory_line1  # [S, Bstar, hidden]

        num_bezier_q = self.num_queries[2]

        tgt_line2 = hs_line1_last_bb.unsqueeze(3).repeat(1, 1, 1, num_bezier_q, 1)  # [B, nb, Nq, nbz_q, H]

        # line2 query_pos
        qpos2 = self.query_embed_line2.weight  # [num_bezier_q, H]
        # expand to [B, num_block, Nq, num_bezier_q, H]
        qpos2_exp = qpos2.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, num_block, num_line_q, 1, 1)

        # merge dims: B * num_block * num_line_q as new batch
        Bstar2 = B * num_block * num_line_q
        tgt_line2 = tgt_line2.view(Bstar2, num_bezier_q, hidden_dim).permute(1, 0, 2)  # [num_bezier_q, Bstar2, H]
        query_pos_line2 = qpos2_exp.view(Bstar2, num_bezier_q, hidden_dim).permute(1, 0, 2)

        memory_line2 = memory_line2.permute(1, 0, 2)  # [Bstar, S, H]
        memory_line2 = memory_line2.view(B, num_block, S, hidden_dim)
        memory_line2 = memory_line2.unsqueeze(2).repeat(1, 1, num_line_q, 1, 1)  # [B, num_block, num_line_q, S, H]
        memory_line2 = memory_line2.view(Bstar2, S, hidden_dim).permute(1, 0, 2)  # [S, Bstar2, H]

        # ------------------ decode line2 -----------------
        hs_line2 = self.line_decoder_2(tgt_line2, memory_line2,
                                      memory_key_padding_mask=None,
                                      pos=None, query_pos=query_pos_line2)
        hs_line2 = hs_line2.permute(0, 2, 1, 3)  # [L2, Bstar2, num_bezier_q, H]

        outputs_line2 = self.line2_bezier_embed(hs_line2).sigmoid()  # [L2, Bstar2, num_bezier_q, 16]
        line2_prob = torch.sigmoid(self.line2_prob_embed(hs_line2[-1]))  # [Bstar2, num_bezier_q, 1]

        # reshape outputs_line2 back to [L2, B, num_block, Nq, num_bezier_q, 16]
        L2 = outputs_line2.shape[0]
        outputs_line2 = outputs_line2.view(L2, B, num_block, num_line_q, num_bezier_q, -1)
        line2_prob = line2_prob.view(B, num_block, num_line_q, num_bezier_q, -1)

        out = {
            'pred_line1': outputs_line1[-1].squeeze(1),     # [B, num_block, Nq, 4]
            'pred_line1_prob': line1_prob.squeeze(1),       # [B, num_block, Nq, 1]
            'pred_line2': outputs_line2[-1].squeeze(1, 3),     # [B, num_block, Nq, num_bezier_q, 16]
            'pred_line2_prob': line2_prob.squeeze(1, 3),       # [B, num_block, Nq, num_bezier_q, 1]
        }

        # aux outputs (所有 decoder 层)
        if self.aux_loss:
            # 简单示例：返回中间层 line1、line2 的所有层预测（你可以按照原来的 _set_aux_loss 实现）
            aux_line1 = [self.line1_bbox_embed(h).sigmoid() for h in hs_line1]  # each: [Bstar, Nq, 4] before reshape
            # reshape每层到 [B, num_block, Nq, 4]
            aux_line1 = [a.view(B, num_block, num_line_q, -1) for a in aux_line1]

            aux_line2 = [self.line2_bezier_embed(h).sigmoid() for h in hs_line2]
            aux_line2 = [a.view(B, num_block, num_line_q, num_bezier_q, -1) for a in aux_line2]

            out['aux_outputs'] = {'line1': aux_line1, 'line2': aux_line2}

        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_line1, outputs_line2):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_line1': b, 'pred_line2': c}
                for b, c in zip(outputs_line1[:-1], outputs_line2[:-1])]
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets, reduction='mean'):
        if self.weight is not None:
            self.weight = torch.tensor(self.weight).to(inputs.device)
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction=reduction)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

    def loss_block_bbox(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_block' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_block'][idx]
        target_boxes = torch.cat([t['block_bbox'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = F.smooth_l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_block_bbox'] = loss_bbox.sum() / num_boxes
        # losses['loss_block_bbox'] = torch.mean(loss_bbox)
        loss_giou = 1 - torch.diag(boxOps.generalized_box_iou(
            boxOps.box_cxcywh_to_xyxy(src_boxes),
            boxOps.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_block_giou'] = torch.mean(loss_giou)
        # loss_overlap = box_ops.overlap(box_ops.box_cxcywh_to_xyxy(src_boxes))
        loss_overlap = boxOps.overlap_ll(outputs['pred_block'])
        losses['loss_overlap_block'] = loss_overlap / num_boxes

        if 'pred_block_prob' in outputs:
            gt_prob = torch.zeros((outputs['pred_block_prob'].shape[0], outputs['pred_block_prob'].shape[1]))
            gt_prob[idx] = 1
            focal_loss = FocalLoss(gamma=0)
            losses['loss_block_prob'] = focal_loss(outputs['pred_block_prob'].squeeze(-1), gt_prob.to(outputs['pred_block_prob'].device))        
        return losses

    def loss_line1_bbox(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_line1' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_line1'][idx]
        target_boxes = torch.cat([t['line1_bbox'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        
        # loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = F.smooth_l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        # losses['loss_line1_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_line1_bbox'] = torch.mean(loss_bbox)
        loss_giou = 1 - torch.diag(boxOps.generalized_box_iou(
            boxOps.box_cxcywh_to_xyxy(src_boxes),
            boxOps.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_line1_giou'] = torch.mean(loss_giou)
        # loss_overlap = box_ops.overlap(box_ops.box_cxcywh_to_xyxy(src_boxes))
        # loss_overlap = boxOps.overlap_ll(outputs['pred_line1'])
        loss_overlap = boxOps.detr_overlap_loss(outputs['pred_line1'], indices)
        losses['loss_overlap_line1'] = loss_overlap

        loss_kernel = boxOps.bbox_center_variance_loss(outputs['pred_line1'])
        losses['loss_kernel_line1'] = loss_kernel
        
        if 'pred_line1_prob' in outputs:
            gt_prob = torch.zeros((outputs['pred_line1_prob'].shape[0], outputs['pred_line1_prob'].shape[1]))
            gt_prob[idx] = 1
            focal_loss = FocalLoss(gamma=0)
            losses['loss_line1_prob'] = focal_loss(outputs['pred_line1_prob'].squeeze(-1), gt_prob.to(outputs['pred_line1_prob'].device))
        return losses
    
    def loss_line2_bezier(self, outputs, targets, indices, num_points):
        """Compute the L1 regression loss"""
        assert 'pred_line2' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_line2'][idx]
        target_points = torch.cat([t['line2_bezier'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_points = F.smooth_l1_loss(src_points, target_points, reduction='none')

        losses = {}
        # losses['loss_line2_bezier'] = loss_points.sum() / num_points
        losses['loss_line2_bezier'] = torch.mean(loss_points)
        if 'pred_line2_prob' in outputs:
            gt_prob = torch.zeros((outputs['pred_line2_prob'].shape[0], outputs['pred_line2_prob'].shape[1]))
            gt_prob[idx] = 1
            focal_loss = FocalLoss(gamma=0)
            losses['loss_line2_prob'] = focal_loss(outputs['pred_line2_prob'].squeeze(-1), gt_prob.to(outputs['pred_line2_prob'].device))
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'loss_block_bbox': self.loss_block_bbox,
            'loss_line1_bbox': self.loss_line1_bbox,
            'loss_line2_bezier': self.loss_line2_bezier
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        loss_dict = {'pred_line1':'loss_line1_bbox', 'pred_line2':'loss_line2_bezier'}
        targets_dict = {'pred_line1':'line1_bbox', 'pred_line2':'line2_bezier'}
        losses = {}
        

        
        for k, v in outputs_without_aux.items():
            if 'prob' in k:
                continue
            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher({k:v}, targets)
            num = sum(len(t[targets_dict[k]]) for t in targets)
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num = torch.as_tensor([num], dtype=torch.float, device=v.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num)
            num = torch.clamp(num / get_world_size(), min=1).item()
            # Compute all the requested losses
            losses.update(self.get_loss(loss_dict[k], outputs, targets, indices, num))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for key, value in aux_outputs.items():
                    if 'prob' in key:
                        continue
                    indices = self.matcher({key:value}, targets)
                    # Compute the average number of target boxes accross all nodes, for normalization purposes
                    num = sum(len(t[targets_dict[key]]) for t in targets)
                    
                    num = torch.as_tensor([num], dtype=torch.float, device=v.device)
                    if is_dist_avail_and_initialized():
                        torch.distributed.all_reduce(num)
                    num = torch.clamp(num / get_world_size(), min=1).item()
                    # Compute all the requested losses
                    kwargs = {}
                    l_dict = self.get_loss(loss_dict[key], aux_outputs, targets, indices, num, **kwargs)
                    l_dict = {k1 + f'_{i}': v1 for k1, v1 in l_dict.items()}
                    losses.update(l_dict)
        return losses

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        Return:
            list of dicts, one dict per image:
            [dicts = {
                key: [shape_data, prob_score, prob_label],
                ...,
            }, ...]
        """
        out_line1, out_line2 = outputs['pred_line1'], outputs['pred_line2']

        # assert len(out_block) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # probability
        # pred_block_prob, pred_line1_prob, pred_line2_prob = \
        #     F.softmax(outputs['pred_block_prob'], -1), F.softmax(outputs['pred_line1_prob'], -1), F.softmax(outputs['pred_line2_prob'], -1)
        # block_scores, block_labels = pred_block_prob[..., :].max(-1)
        # line1_scores, line1_labels = pred_line1_prob[..., :].max(-1)
        # line2_scores, line2_labels = pred_line2_prob[..., :].max(-1)
        pred_line1_prob, pred_line2_prob = outputs['pred_line1_prob'], outputs['pred_line2_prob']
        # block_bbox
        # block_bbox = boxOps.box_cxcywh_to_xyxy(out_block)
        # img_h, img_w = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # block_bbox = block_bbox * scale_fct[:, None, :]
        # line1_bbox
        line1_bbox = boxOps.box_cxcywh_to_xyxy(out_line1)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        line1_bbox = line1_bbox * scale_fct[:, None, :]
        # line2_bezier
        line2_bezier = out_line2 * target_sizes.repeat(1, 8)[:, None, :]

        results = [{'line1_bbox':[l1, l1s], 'line2_bezier': [l2, l2s]} \
                   for l1, l2, l1s, l2s in \
                   zip(line1_bbox, line2_bezier, pred_line1_prob, pred_line2_prob)]

        return results

def build(args):
    device = torch.device(args.device)
    model = HDLayout(args)
    matcher = build_matcher(args)
    weight_dict = {
        'loss_line1_bbox': 20,
        'loss_line1_giou': 1,
        'loss_kernel_line1': 1e-5,
        'loss_line1_prob': 0.01,
        'loss_overlap_line1': 1,
        'loss_line2_bezier': 20,
        'loss_line2_prob': 0.01,
        }

    if args.aux_loss:
        aux_weight_dict = {}
        
        # for i in range(args.dec_layers - 1):
        #     aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        # for i in range(args.dec_layers_block - 1):
        #     for k, v in weight_dict.items():
        #         if 'block' in k:
        #             aux_weight_dict.update({k + f'_{i}': v})
        for i in range(2 - 1):
            for k, v in weight_dict.items():
                if 'line1' in k:
                    aux_weight_dict.update({k + f'_{i}': v})
        for i in range(2 - 1):
            for k, v in weight_dict.items():
                if 'line2' in k:
                    aux_weight_dict.update({k + f'_{i}': v})
        weight_dict.update(aux_weight_dict)

    # define criterion
    losses = ['loss_line1_boxes', 'loss_line2_points']
    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                            eos_coef=0.1, losses=losses)
    criterion.to(device)
    # TODO define postprocessor
    postprocessors = {'res': PostProcess()}

    return model, criterion, postprocessors