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
logger = logging.getLogger(__name__)

class ModuleCat(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.ln1 = nn.Linear(dims, dims // 2)
        self.ln2 = nn.Linear(dims, dims // 2)
        
    def forward(self, x1, x2):
        x1 = self.ln1(x1)
        x2 = self.ln2(x2)
        return torch.cat([x1, x2], dim=-1)

class HDLayout(nn.Module):
    """ Whole HDLayout model composited by Block, Line and Character module """
    def __init__(self, args):
        """ Initialize the model
        """
        super(HDLayout, self).__init__()
        self.bs = args.batch_size
        self.aux_loss = args.aux_loss
        self.num_queries = args.num_queries # block 4 / line 4*4
        self.backbone = build_backbone(args)
        # coefficient
        # self.block_coef = args.block_memory_coef
        # self.line1_coef = args.line_memory_coef
        # self.line2_coef = args.bezier_memory_coef
        self.block_coef = 1
        self.line1_coef = 3
        self.line2_coef = 3
        # self.transformer = build_transformer(args)
        # self.block_decoder = build_transformer_decoder(args, dec_layers=args.dec_layers_block, return_intermediate=True)

        self.line_decoder_1 = build_transformer_decoder(args, dec_layers=2, return_intermediate=True)
        self.line_decoder_2 = build_transformer_decoder(args, dec_layers=2, return_intermediate=True)
        # hidden_dim = self.transformer.d_model
        hidden_dim = args.hidden_dim
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.block_decoder = nn.Sequential(
            nn.Linear(4, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim)
        )
        # queries embedding
        # self.query_embed_block = nn.Embedding(self.num_queries[0], hidden_dim)
        
        self.query_embed_line1 = nn.Embedding(self.num_queries[1], hidden_dim)
        self.query_embed_line2 = nn.Embedding(self.num_queries[2], hidden_dim)
        # results embedding
        self.block_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.line1_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.line2_bezier_embed = MLP(hidden_dim, hidden_dim, 16, 3)
        # results probility
        self.block_prob_embed = nn.Linear(hidden_dim, 1)
        self.line1_prob_embed = nn.Linear(hidden_dim, 1)
        self.line2_prob_embed = nn.Linear(hidden_dim, 1)
        # connect embedding
        self.block_line1_embed = nn.Linear(self.num_queries[0], hidden_dim)
        self.line1_line2_embed = nn.Linear(self.num_queries[1], hidden_dim)
        # src embedding
        self.src_block_embed = nn.Linear(args.dim_feedforward, hidden_dim)
        self.src_line1_embed = nn.Linear(args.dim_feedforward, hidden_dim)
        self.src_line2_embed = nn.Linear(args.dim_feedforward, hidden_dim)
        # cat
        self.cat_block = ModuleCat(hidden_dim)
        self.cat_line1 = ModuleCat(hidden_dim)

    def forward(self, samples: dict):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # backbone
        samples_img = samples['imgs']
        samples_block_bbox = samples['block_bboxes']
        if isinstance(samples_img, (list, torch.Tensor)):
            samples_img = nested_tensor_from_tensor_list(samples_img)
        # if isinstance(samples_img, (list, torch.Tensor)):
        #     samples_block_bbox = nested_tensor_from_tensor_list(samples_block_bbox)
            
        features, pos = self.backbone(samples_img)

        src, mask = features[-1].decompose()
        assert mask is not None

        mask = mask.flatten(1)
        pos_embed = pos[-1].flatten(2).permute(2, 0, 1)
        # block encoder
        # TODO: 添加实例 Bbox 编码 Bbox_embed
        hs_block = self.block_decoder(samples_block_bbox)
        hs_block = repeat(hs_block, 'b d -> b d c', c=1)
        # line
        # TODO: 添加已有实例 Bbox 信息 memory = Bbox_embed + src_embed
        memory = self.line1_coef * self.block_line1_embed(hs_block).permute(2, 0, 1)
        # memory = 1.5*memory + self.src_line1_embed(src.reshape(src.shape[0], src.shape[1], -1).permute(0, 2, 1)).permute(2, 0, 1)
        # memory = self.line1_coef * memory + self.src_line1_embed(src.reshape(src.shape[0], src.shape[1], -1).permute(0, 2, 1)).permute(2, 0, 1)
        block_embed = self.src_line1_embed(src.reshape(src.shape[0], src.shape[1], -1).permute(0, 2, 1)).permute(2, 0, 1)
        memory = self.cat_block(memory, block_embed)
        query_embed = self.query_embed_line1.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, memory.shape[1], 1)
        tgt = torch.zeros_like(query_embed)
        # tgt = query_embed
        hs_line1 = self.line_decoder_1(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed).transpose(1, 2)
        outputs_line1 = self.line1_bbox_embed(hs_line1).sigmoid()
        line1_prob = torch.sigmoid(self.line1_prob_embed(hs_line1[-1]))
        
        # bezier
        memory = self.line2_coef * self.line1_line2_embed(hs_line1[-1].permute(0, 2, 1)).permute(2, 0, 1)
        # memory = 3*memory + self.src_line2_embed(src.reshape(src.shape[0], src.shape[1], -1).permute(0, 2, 1)).permute(2, 0, 1)
        # memory = self.line2_coef * memory + self.src_line2_embed(src.reshape(src.shape[0], src.shape[1], -1).permute(0, 2, 1)).permute(2, 0, 1)
        line2_embed = self.src_line2_embed(src.reshape(src.shape[0], src.shape[1], -1).permute(0, 2, 1)).permute(2, 0, 1)
        memory = self.cat_line1(memory, line2_embed)
        query_embed = self.query_embed_line2.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, memory.shape[1], 1)
        tgt = torch.zeros_like(query_embed)
        # tgt = query_embed
        hs_line2 = self.line_decoder_2(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed).transpose(1, 2)
        outputs_line2 = self.line2_bezier_embed(hs_line2).sigmoid()
        line2_prob = torch.sigmoid(self.line2_prob_embed(hs_line2[-1]))

        out = {'pred_line1': outputs_line1[-1], 'pred_line2': outputs_line2[-1],
               'pred_line1_prob': line1_prob, 'pred_line2_prob': line2_prob}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_line1, outputs_line2)
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
        loss_overlap = boxOps.overlap_ll(outputs['pred_line1'])
        losses['loss_overlap_line1'] = loss_overlap / num_boxes

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
        # 'loss_line1_bbox': args.bbox_loss_coef,
        # 'loss_line1_giou': args.giou_loss_coef,
        # 'loss_line1_prob': args.prob_loss_coef,
        # 'loss_overlap_line1': args.overlap_loss_coef,
        # 'loss_line2_bezier': args.point_loss_coef,
        # 'loss_line2_prob': args.prob_loss_coef,
        'loss_line1_bbox': 10,
        'loss_line1_giou': 0.1,
        'loss_line1_prob': 0.01,
        'loss_overlap_line1': 0.0001,
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