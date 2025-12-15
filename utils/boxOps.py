# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
    # b = torch.stack(b, dim=-1)
    # b = b.clamp(min=0.0, max=1.0)
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    # before match IOU
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    # matched IOU
    # lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,M,2]
    # rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,M,2]

    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # inter = wh[:, 0] * wh[:, 1]  # [N,M]

    # union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    # before match IOU
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    # matched
    # lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    # rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def compute_iou(box1, box2):
    """
    计算两个矩形框之间的交并比(IoU)
    :param box1: 第一个框，格式为 (x1, y1, x2, y2)
    :param box2: 第二个框，格式为 (x1, y1, x2, y2)
    :return: IoU值
    """
    x1_inter = torch.max(box1[0], box2[0])
    y1_inter = torch.max(box1[1], box2[1])
    x2_inter = torch.min(box1[2], box2[2])
    y2_inter = torch.min(box1[3], box2[3])

    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)  # 添加一个小值防止除零错误
    return iou

def overlap(boxes):
    """
    计算生成框之间的overlap loss
    :param boxes: 包含n个生成框的张量, 形状为 (bs, n, 4), 每个框格式为 (cx, cy, w, h)
    :return: overlap loss
    """
    bs = boxes.size(0)
    loss_res = torch.tensor(0.0, device=boxes.device)
    for k in range(bs):
        loss = torch.tensor(0.0, device=boxes.device)
        sub_boxes = box_cxcywh_to_xyxy(boxes[k])
        num_boxes = sub_boxes.size(0)
        for i in range(num_boxes):
            for j in range(i + 1, num_boxes):
                iou = compute_iou(sub_boxes[i], sub_boxes[j])
                loss += iou
        loss /= (num_boxes * (num_boxes - 1) / 2)
        loss_res += loss
    loss_res /= bs
    return loss_res

def compute_iou_ll(box1, box2):
    """
    计算两个矩形框之间的交并比(IoU)
    :param box1: [bs, num, bbox]第一个框，格式为 (x1, y1, x2, y2)
    :param box2: [bs, num, bbox]第二个框，格式为 (x1, y1, x2, y2)
    :return: IoU值
    """
    x1_inter = torch.max(box1[:, 0], box2[:, 0])
    y1_inter = torch.max(box1[:, 1], box2[:, 1])
    x2_inter = torch.min(box1[:, 2], box2[:, 2])
    y2_inter = torch.min(box1[:, 3], box2[:, 3])

    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)  # 添加一个小值防止除零错误
    return iou

def overlap_ll(boxes):
    """
    并行计算生成框之间的overlap loss
    :param boxes: 包含n个生成框的张量, 形状为 (bs, n, 4), 每个框格式为 (cx, cy, w, h)
    :return: overlap loss
    """
    bs = boxes.size(0)
    num = boxes.size(1)
    if num == 1:
        return torch.tensor(0.0, device=boxes.device)
    num_res = num * (num - 1) // 2
    loss_res = torch.tensor(0.0, device=boxes.device)
    
    boxes = box_cxcywh_to_xyxy(boxes).permute(1, 0, 2)
    src_boxes = torch.zeros((num_res, bs, 4)).to(boxes.device)
    target_boxes = torch.zeros((num_res, bs, 4)).to(boxes.device)
    k = 0
    for i in range(boxes.size(0)):
        for j in range(i + 1, boxes.size(0)):
            src_boxes[k] = boxes[i]
            target_boxes[k] = boxes[j]
            k += 1
    iou = compute_iou_ll(src_boxes.view(-1, 4), target_boxes.view(-1, 4))
    loss_res = iou.sum() / (num_res*2)
    return loss_res

def bbox_center_variance_loss(bboxes, epsilon=1e-6):
    """
    计算4个边界框中心的方差损失，方差越小损失越大
    
    参数:
        bboxes: 张量，形状为 (batch_size, 4, 4)，每个边界框格式为 [x1, y1, x2, y2]
                其中x1,y1是左上角坐标，x2,y2是右下角坐标
        epsilon: 防止除零错误的小值
    
    返回:
        loss: 标量张量，方差越小，损失值越大
    """
    # 计算每个边界框的中心坐标
    # 中心x坐标 = (x1 + x2) / 2
    # 中心y坐标 = (y1 + y2) / 2
    bboxes = box_cxcywh_to_xyxy(bboxes)
    centers_x = (bboxes[..., 0] + bboxes[..., 2]) / 2  # 形状: (batch_size, 4)
    centers_y = (bboxes[..., 1] + bboxes[..., 3]) / 2  # 形状: (batch_size, 4)
    
    # 计算x和y方向的中心方差
    var_x = torch.var(centers_x, dim=1, unbiased=False)  # 形状: (batch_size,)
    var_y = torch.var(centers_y, dim=1, unbiased=False)  # 形状: (batch_size,)
    
    # 总方差为x和y方向方差之和
    total_variance = var_x + var_y  # 形状: (batch_size,)
    
    # 计算损失: 方差越小，损失越大
    # 使用1/(方差 + epsilon)确保这一特性，同时避免除零
    loss = 1.0 / (total_variance + epsilon)  # 形状: (batch_size,)
    
    # 计算批次的平均损失
    return torch.mean(loss)


def compute_pairwise_iou(bboxes):
    """计算一组bbox之间的两两IoU"""
    # bboxes形状: (N, 4)，格式[x1, y1, x2, y2]
    area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])  # 每个框的面积
    
    # 计算交集左上角和右下角坐标
    x1 = torch.max(bboxes[:, None, 0], bboxes[None, :, 0])  # (N, N)
    y1 = torch.max(bboxes[:, None, 1], bboxes[None, :, 1])
    x2 = torch.min(bboxes[:, None, 2], bboxes[None, :, 2])
    y2 = torch.min(bboxes[:, None, 3], bboxes[None, :, 3])
    
    # 计算交集面积（确保不小于0）
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # 计算并集面积 = 面积1 + 面积2 - 交集面积
    union = area[:, None] + area[None, :] - intersection
    
    # 计算IoU（添加epsilon避免除零）
    return intersection / (union + 1e-6)

def detr_overlap_loss(bboxes, indices, num_queries=4, epsilon=1e-6):
    """
    适配DETR结构的overlap损失计算
    
    参数:
        bboxes: 预测的边界框，形状为 (batch_size, num_queries, 4)，格式为 [x1, y1, x2, y2]
        indices: 二分匹配的结果，由DETR的matcher返回，格式为列表，每个元素是
                 (pred_indices, target_indices)，分别表示预测框和真实框的匹配索引
        num_queries: 预测框数量，这里固定为4
        epsilon: 防止数值不稳定的小值
    """
    bboxes = box_cxcywh_to_xyxy(bboxes)
    batch_size = bboxes.shape[0]
    total_loss = 0.0
    
    for batch_idx in range(batch_size):
        # 获取当前样本的预测框
        batch_bboxes = bboxes[batch_idx]  # 形状: (4, 4)
        
        # 从匹配结果中获取前景框索引（排除背景）
        pred_indices, _ = indices[batch_idx]
        # pred_indices = []
        foreground_mask = torch.zeros(num_queries, dtype=torch.bool, device=bboxes.device)
        foreground_mask[pred_indices] = True  # 标记匹配到真实框的预测框
        foreground_bboxes = batch_bboxes[foreground_mask]  # 筛选出前景框
        
        # 若前景框数量 < 2，无需计算overlap损失（单个框无重叠问题）
        if len(foreground_bboxes) < 2:
            continue
        
        # 计算前景框之间的IoU矩阵
        ious = compute_pairwise_iou(foreground_bboxes)
        
        # 提取上三角部分（排除对角线和重复计算）
        num_foreground = len(foreground_bboxes)
        triu_mask = torch.triu(torch.ones(num_foreground, num_foreground, device=bboxes.device), diagonal=1).bool()
        overlaps = ious[triu_mask]  # 形状: (N*(N-1)/2,)，N为前景框数量
        
        # 计算overlap损失：重叠越大，损失越大
        # 采用IoU的平方作为损失，放大高重叠的惩罚
        loss = torch.mean(overlaps **2)
        total_loss += loss
    
    # 计算批次平均损失
    return total_loss / batch_size