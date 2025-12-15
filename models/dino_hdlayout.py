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
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy,
                       get_world_size, is_dist_avail_and_initialized)
import utils.boxOps as boxOps
from einops import repeat
from torchvision.ops import roi_align
import models.vision_transformer as vits
import os
from models.config import Config
from utils.boxOps import box_cxcywh_to_xyxy, generalized_box_iou

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """
    Simple wrapper around timm create_model to return feature maps or pooled features.
    If timm is unavailable, falls back to a small CNN.
    """
    def __init__(self, arch='vit_base_patch16_224', pretrained=True, pretrained_weights=None, patch_size=16, out_dim=768):
        super().__init__()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.pretrained = pretrained
        self.pretrained_weights = pretrained_weights
        self.out_dim = out_dim
        self.arch = arch
        self.patch_size = patch_size
        self.checkpoint_key = "teacher"  # can be set to load specific key from checkpoint
        # features_only returns a list of feature maps
        self.net = vits.__dict__[self.arch](patch_size=patch_size, num_classes=0)
        self.net.eval()
        self.net.to(device)
        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
            if self.checkpoint_key is not None and self.checkpoint_key in state_dict:
                print(f"Take key {self.checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[self.checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            
            state_dict = {k: v for k, v in state_dict.items() if "head" not in k }
            msg = self.net.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(self.pretrained_weights, msg))
        else:
            print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
            url = None
            if self.arch == "vit_small" and self.patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif self.arch == "vit_small" and self.patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif self.arch == "vit_base" and self.patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif self.arch == "vit_base" and self.patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                self.net.load_state_dict(state_dict, strict=True)
            else:
                print("There is no reference weights available for this model => We use random weights.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return a pooled feature vector per image: (B, out_dim)
        feats = self.net(x)
        # x = self.net.forward_features(x)
        return feats

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
class DINO_HDLayout(nn.Module):
    """Shared-backbone implementation that *explicitly* concatenates features as:
    - B-stage input = [image-level pooled features || A1(features)]
    - C-stage input = [image-level pooled features || B1(features)]

    To simulate "image + A1 features" when using a single shared extractor, we compute a
    lightweight pooled image embedding (image_proj) from the raw image and concatenate it
    with the A1 pooled features.
    """
    def __init__(self, cfg: Config, args=None):
        super().__init__()
        self.cfg = cfg
        feat_dim = 768 // 2
        self.region_bbox_num = cfg.region_bbox_num
        self.line_bbox_num = cfg.line_bbox_num
        self.char_bezier_num = cfg.char_bezier_num
        self.num_classes = cfg.num_classes
        # shared backbone that produces pooled features (B, feat_dim)
        # shared backbone that produces pooled features (B, feat_dim)
        self.block_queries = nn.Parameter(torch.randn(1, cfg.region_bbox_num, feat_dim))
        self.line_queries = nn.Parameter(torch.randn(1, cfg.line_bbox_num, feat_dim))
        
        shared = Encoder(arch=self.cfg.backbone_name, 
                                   pretrained=self.cfg.backbone_pretrained, 
                                   pretrained_weights=self.cfg.backbone_pretrained_weight, 
                                   patch_size=16, 
                                   out_dim=feat_dim)

        self.shared = shared
        # self.block_fe = shared
        # self.block_decoder = build_transformer_decoder(args, dec_layers=2, return_intermediate=True)

        self.block_reg = MLPRegressor(shared.out_dim, out_dim=cfg.region_bbox_dim * cfg.region_bbox_num)
        self.block_class = nn.Linear(shared.out_dim, (self.num_classes + 1) * cfg.region_bbox_num)

        self.line_in_dim = shared.out_dim
        # self.line_fe = shared
        # self.line_decoder = build_transformer_decoder(args, dec_layers=2, return_intermediate=True)
        
        self.line_reg = MLPRegressor(self.line_in_dim, out_dim=cfg.line_bbox_dim * cfg.line_bbox_num)
        self.line_class = nn.Linear(self.line_in_dim, (self.num_classes + 1) * cfg.line_bbox_num)

        self.cross_attn_line = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=8,
            batch_first=True
        )

        self.char_in_dim = shared.out_dim
        # self.char_fe = shared
        # self.char_decoder = build_transformer_decoder(args, dec_layers=2, return_intermediate=True)
        self.char_reg = MLPRegressor(self.char_in_dim, out_dim=cfg.char_bezier_dim * cfg.char_bezier_num)
        self.char_class = nn.Linear(self.char_in_dim, (self.num_classes + 1) * cfg.char_bezier_num)

        self.cross_attn_char = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, x):
        # x: (B,3,H,W) (B,3,512,512)

        x = self.shared.net.prepare_tokens(x) # x : (B, 1025, 384)

        x = self.shared(x)  # (B, feat_dim) # (B, 384)
        fblock = x = x.unsqueeze(1)  # (B,1,384)
        # hs_block, _ = self.block_decoder(fblock, x)  # (B, feat_dim)
        hs_block = fblock
        hs_block = hs_block.squeeze(1)
        region_bbox = self.block_reg(hs_block)
        region_bbox = region_bbox.view(-1, self.cfg.region_bbox_num, self.cfg.region_bbox_dim)
        region_class = self.block_class(hs_block).view(-1, self.cfg.region_bbox_num, self.num_classes + 1)
        # region_conf = self.A_conf(fA)

        query_line = fblock # (B,1,384)
        attn_out_line, _ = self.cross_attn_line(query_line, x, x) # cross-attention
        fline = attn_out_line # (B,1,384)
        hs_line, _ = self.line_decoder(fline, x)  # (B, feat_dim)
        hs_line = hs_line.squeeze(1)
        line_bbox = self.line_reg(hs_line)
        line_bbox = line_bbox.view(-1, self.cfg.line_bbox_num, self.cfg.line_bbox_dim)
        line_class = self.line_class(hs_line).view(-1, self.cfg.line_bbox_num, self.num_classes + 1)
        # line_conf = self.B_conf(fB)

        query_char = fline
        attn_out_char, _ = self.cross_attn_char(query_char, x, x)
        fchar = attn_out_char  # (B,1,384)
        hs_char, _ = self.char_decoder(fchar, x)  # (B, feat_dim)
        hs_char = hs_char.squeeze(1)
        char_bezier = self.char_reg(hs_char)
        char_bezier = char_bezier.view(-1, self.cfg.char_bezier_num, self.cfg.char_bezier_dim)
        char_class = self.char_class(hs_char).view(-1, self.cfg.char_bezier_num, self.num_classes + 1)
        # char_conf = self.C_conf(fC)
        

        return {
            'pred_block': region_bbox,
            'pred_block_logits': region_class,
            'pred_line': line_bbox,
            'pred_line_logits': line_class,
            'pred_char': char_bezier,
            'pred_char_logits': char_class
        }


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
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        """ Create the criterion.
        Parameters:
            num: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        # self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, loss):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        loss_dict_outputs = {
                    'loss_block_labels':'pred_block_logits',
                    'loss_line_labels':'pred_line_logits',
                    'loss_char_labels':'pred_char_logits',
                    }
        loss_dict_targets = {   
                    'loss_block_labels':'block_bbox_labels',
                    'loss_line_labels':'line_bbox_labels',
                    'loss_char_labels':'char_bezier_labels',
                    }
        outputs_key = loss_dict_outputs[loss]
        targets_key = loss_dict_targets[loss]
        
        assert outputs_key in outputs
        
        src_logits = outputs[outputs_key]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[targets_key][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {f'{loss}_ce': loss_ce}

        return losses
    
    def loss_block_bbox(self, outputs, targets, indices, num_boxes, loss):
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
        # loss_giou = 1 - torch.diag(boxOps.generalized_box_iou(
        #     boxOps.box_cxcywh_to_xyxy(src_boxes),
        #     boxOps.box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = 1 - generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        
        losses['loss_block_giou'] = torch.mean(loss_giou)
        # loss_overlap = box_ops.overlap(box_ops.box_cxcywh_to_xyxy(src_boxes))
        
        loss_overlap = boxOps.overlap_ll(outputs['pred_block'])
        losses['loss_block_overlap'] = loss_overlap / num_boxes
        
        # loss_kernel = boxOps.bbox_center_variance_loss(outputs['pred_block'])
        # losses['loss_line_kernel'] = loss_kernel
        
        if 'pred_block_conf' in outputs:
            gt_conf = torch.zeros((outputs['pred_block_conf'].shape[0], outputs['pred_block_conf'].shape[1]))
            gt_conf[idx] = 1
            focal_loss = FocalLoss(gamma=0)
            losses['loss_block_conf'] = focal_loss(outputs['pred_block_conf'].squeeze(-1), gt_conf.to(outputs['pred_block_conf'].device))        
        return losses

    def loss_line_bbox(self, outputs, targets, indices, num_boxes, loss):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_line' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_line'][idx]
        target_boxes = torch.cat([t['line_bbox'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        
        # loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = F.smooth_l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        # losses['loss_line_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_line_bbox'] = torch.mean(loss_bbox)
        # loss_giou = 1 - torch.diag(boxOps.generalized_box_iou(
        #     boxOps.box_cxcywh_to_xyxy(src_boxes),
        #     boxOps.box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = 1 - generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        losses['loss_line_giou'] = torch.mean(loss_giou)
        # loss_overlap = box_ops.overlap(box_ops.box_cxcywh_to_xyxy(src_boxes))
        loss_overlap = boxOps.overlap_ll(outputs['pred_line'])
        # loss_overlap = boxOps.detr_overlap_loss(outputs['pred_line'], indices)
        losses['loss_line_overlap'] = loss_overlap

        # loss_kernel = boxOps.bbox_center_variance_loss(outputs['pred_line'])
        # losses['loss_line_kernel'] = loss_kernel
        
        if 'pred_line_conf' in outputs:
            gt_conf = torch.zeros((outputs['pred_line_conf'].shape[0], outputs['pred_line_conf'].shape[1]))
            gt_conf[idx] = 1
            focal_loss = FocalLoss(gamma=0)
            losses['loss_line_conf'] = focal_loss(outputs['pred_line_conf'].squeeze(-1), gt_conf.to(outputs['pred_line_conf'].device))
        return losses
    
    def loss_char_bezier(self, outputs, targets, indices, num_points, loss):
        """Compute the L1 regression loss"""
        assert 'pred_char' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_char'][idx]
        target_points = torch.cat([t['char_bezier'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_points = F.smooth_l1_loss(src_points, target_points, reduction='none')

        losses = {}
        # losses['loss_char_bezier'] = loss_points.sum() / num_points
        losses['loss_char_bezier'] = torch.mean(loss_points)
        if 'pred_char_conf' in outputs:
            gt_conf = torch.zeros((outputs['pred_char_conf'].shape[0], outputs['pred_char_conf'].shape[1]))
            gt_conf[idx] = 1
            focal_loss = FocalLoss(gamma=0)
            losses['loss_char_conf'] = focal_loss(outputs['pred_char_conf'].squeeze(-1), gt_conf.to(outputs['pred_char_conf'].device))
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
            'loss_block_labels': self.loss_labels,
            'loss_line_bbox': self.loss_line_bbox,
            'loss_line_labels': self.loss_labels,
            'loss_char_bezier': self.loss_char_bezier,
            'loss_char_labels': self.loss_labels,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, loss, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        loss_dict = {'pred_block':'loss_block_bbox', 
                     'pred_block_logits':'loss_block_labels',
                     'pred_line':'loss_line_bbox', 
                     'pred_line_logits':'loss_line_labels',
                     'pred_char':'loss_char_bezier',
                     'pred_char_logits':'loss_char_labels',
                     }
        targets_dict = {'pred_block':'block_bbox', 'pred_line':'line_bbox', 'pred_char':'char_bezier'}
        losses = {}
        

        
        for k, v in outputs_without_aux.items():
            if 'logits' in k:
                continue
            # Retrieve the matching between the outputs of the last layer and the targets
            labels_k = k + '_logits'
            indices = self.matcher({k:v, labels_k:outputs_without_aux[labels_k]}, targets)
            num = sum(len(t[targets_dict[k]]) for t in targets)
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num = torch.as_tensor([num], dtype=torch.float, device=v.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num)
            num = torch.clamp(num / get_world_size(), min=1).item()
            # Compute all the requested losses
            losses.update(self.get_loss(loss_dict[k], outputs, targets, indices, num))
            losses.update(self.get_loss(loss_dict[labels_k], outputs, targets, indices, num))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for key, value in aux_outputs.items():
                    if 'conf' in key:
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
        out_block, out_line, out_char = outputs['pred_block'], outputs['pred_line'], outputs['pred_char']
        out_block_scores = F.softmax(outputs['pred_block_logits'], -1)
        out_line_scores = F.softmax(outputs['pred_line_logits'], -1)
        out_char_scores = F.softmax(outputs['pred_char_logits'], -1)
        
        prob_block, label_block = out_block_scores.max(-1)
        prob_line, label_line = out_line_scores.max(-1)
        prob_char, label_char = out_char_scores.max(-1)
        
        # assert len(out_block) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # probability
        # pred_block_conf, pred_line_conf, pred_char_conf = outputs['pred_block_conf'], outputs['pred_line_conf'], outputs['pred_char_conf']
        # block_bbox
        block_bbox = boxOps.box_cxcywh_to_xyxy(out_block)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        block_bbox = block_bbox * scale_fct[:, None, :]
        # line_bbox
        line_bbox = boxOps.box_cxcywh_to_xyxy(out_line)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        line_bbox = line_bbox * scale_fct[:, None, :]
        # char_bezier
        char_bezier = out_char * target_sizes.repeat(1, 8)[:, None, :]

        # results = [{'block_bbox':[b, bc], 'line_bbox':[l, lc], 'char_bezier': [c, cc]} \
        #            for b, l, c, bc, lc, cc in \
        #            zip(block_bbox, line_bbox, char_bezier, pred_block_conf, pred_line_conf, pred_char_conf)]
        # results = [{'block_bbox':[b], 'line_bbox':[l], 'char_bezier': [c]} \
        #            for b, l, c in \
        #            zip(block_bbox, line_bbox, char_bezier)]
        # return results
        results = []
        batch_size = len(target_sizes)
        
        for i in range(batch_size):
            # --- Block 过滤 ---
            # 找到该图片中所有 label 不为 0 的索引
            keep_block = label_block[i] != 0
            b_box_filtered = block_bbox[i][keep_block]
            # b_prob_filtered = prob_block[i][keep_block]
            # b_label_filtered = label_block[i][keep_block]

            # --- Line 过滤 ---
            keep_line = label_line[i] != 0
            l_box_filtered = line_bbox[i][keep_line]
            # l_prob_filtered = prob_line[i][keep_line]
            # l_label_filtered = label_line[i][keep_line]

            # --- Char 过滤 ---
            keep_char = label_char[i] != 0
            c_bezier_filtered = char_bezier[i][keep_char]
            # c_prob_filtered = prob_char[i][keep_char]
            # c_label_filtered = label_char[i][keep_char]

            # 按照 docstring 组装字典: key: [shape, prob, label]
            results.append({
                'block_bbox': [b_box_filtered],
                'line_bbox':  [l_box_filtered],
                'char_bezier': [c_bezier_filtered]
            })

        return results

def build(args):
    device = torch.device(args.device)
    cfg = Config()
    num_classes = cfg.num_classes
    model = DINO_HDLayout(cfg, args)
    matcher = build_matcher(args)
    weight_dict = {
        'loss_block_bbox': 5,
        'loss_block_giou': 1,
        'loss_block_overlap': 1,
        'loss_block_labels_ce': 2,
        'loss_block_conf': 0.01,
        
        'loss_line_bbox': 5,
        'loss_line_giou': 1,
        'loss_line_overlap': 1,
        'loss_line_conf': 0.01,
        'loss_line_labels_ce': 2,
        
        'loss_char_bezier': 5,
        'loss_char_conf': 0.01,
        'loss_char_labels_ce': 2,
        }

    if args.aux_loss:
        aux_weight_dict = {}
        
        # for i in range(args.dec_layers - 1):
        #     aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        for i in range(2 - 1):
            for k, v in weight_dict.items():
                if 'block' in k:
                    aux_weight_dict.update({k + f'_{i}': v})
        for i in range(2 - 1):
            for k, v in weight_dict.items():
                if 'line' in k:
                    aux_weight_dict.update({k + f'_{i}': v})
        for i in range(2 - 1):
            for k, v in weight_dict.items():
                if 'char' in k:
                    aux_weight_dict.update({k + f'_{i}': v})
        weight_dict.update(aux_weight_dict)

    # define criterion
    # losses = ['loss_block_bboxes', 'loss_line_bboxes', 'loss_char_beziers', ]
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict,
                            eos_coef=0.1)
    criterion.to(device)
    # TODO define postprocessor
    postprocessors = {'res': PostProcess()}

    return model, criterion, postprocessors