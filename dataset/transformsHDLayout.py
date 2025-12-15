"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.boxOps import box_xyxy_to_cxcywh
from utils.misc import interpolate

def hmove(image, target, distance):
    affined_image = F.affine(image, angle=0, translate=(distance, 0), fill=255, scale=1.0, shear=0)
    
    w, h = image.size
    
    for k, v in target.items():
        if "box" in k:
            # boxes are in xyxy format
            boxes = torch.tensor(v).view(-1, 4)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] + w * distance
            # 坐标裁剪（确保在图像范围内）
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w)
            
            # 修正坐标顺序（确保x2 >= x1，y2 >= y1）
            x1 = torch.minimum(boxes[:, 0], boxes[:, 2])  # 新的左上x
            y1 = torch.minimum(boxes[:, 1], boxes[:, 3])  # 新的左上y
            x2 = torch.maximum(boxes[:, 0], boxes[:, 2])  # 新的右下x
            y2 = torch.maximum(boxes[:, 1], boxes[:, 3])  # 新的右下y
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            
            # 过滤无效框（宽高需>0）
            valid = ((x2 - x1) > 0) & ((y2 - y1) > 0)
            boxes = boxes[valid]
            
            target[k] = boxes.view(-1, 2, 2)
        if "bezier" in k:
            # beziers are in xy format
            bezier = torch.tensor(v)
            mask = torch.arange(bezier.size(1)) % 2 == 0
            bezier[:, mask] = bezier[:, mask] + w * distance
            # bezier = bezier + torch.as_tensor([w * distance, 0])
            target[k] = bezier
            
    return affined_image, target
            
def vmove(image, target, distance):
    affined_image = F.affine(image, angle=0, translate=(0, distance), fill=255, scale=1.0, shear=0)
    
    w, h = image.size
    
    for k, v in target.items():
        if "box" in k:
            # boxes are in xyxy format
            boxes = torch.tensor(v).view(-1, 4)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] + h * distance
            # 坐标裁剪（确保在图像范围内）
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h)
            
            # 修正坐标顺序（确保x2 >= x1，y2 >= y1）
            x1 = torch.minimum(boxes[:, 0], boxes[:, 2])  # 新的左上x
            y1 = torch.minimum(boxes[:, 1], boxes[:, 3])  # 新的左上y
            x2 = torch.maximum(boxes[:, 0], boxes[:, 2])  # 新的右下x
            y2 = torch.maximum(boxes[:, 1], boxes[:, 3])  # 新的右下y
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            
            # 过滤无效框（宽高需>0）
            valid = ((x2 - x1) > 0) & ((y2 - y1) > 0)
            boxes = boxes[valid]

            target[k] = boxes.view(-1, 2, 2)
        if "bezier" in k:
            # beziers are in xy format
            bezier = torch.tensor(v)
            mask = torch.arange(bezier.size(1)) % 2 == 1
            bezier[:, mask] = bezier[:, mask] + h * distance
            # bezier = bezier.clamp(min=0, max=)
            # bezier = bezier + torch.as_tensor([0, h * distance])
            target[k] = bezier
            
    return affined_image, target

class RandomMove(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, target):
        rand = random.random()
        if rand < self.p:
            return hmove(img, target, distance=random.uniform(0, 0.2))
        elif rand < self.p * 2:
            return vmove(img, target, distance=random.uniform(0, 0.2))
        return img, target

def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    # target = target.copy()
    for k, v in target.items():
        if "bbox" in k:
            # boxes are in xyxy format
            boxes = torch.tensor(v).view(-1, 4)
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
            target[k] = boxes.view(-1, 2, 2)
        if "bezier" in k:
            # beziers are in xy format
            bezier = torch.tensor(v)
            mask = torch.arange(bezier.size(1)) % 2 == 0
            bezier[:, mask] = bezier[:, mask] * -1 + w
            bezier = bezier[:, [6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9]]

            target[k] = bezier
    return flipped_image, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        for key, value in target.items():
            if "box" in key and "label" not in key:
                boxes = value.view(-1, 4)
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target[key] = boxes
            elif "size" not in key and "label" not in key:
                if value.shape[-1] == 2:
                    value = value.view(-1, 8)
                    target[key] = value / torch.tensor([w, h, w, h, w, h, w, h], dtype=torch.float32)
                elif value.shape[-1] == 16:
                    target[key] = value / torch.tensor([w, h, w, h, w, h, w, h, w, h, w, h, w, h, w, h], dtype=torch.float32)
        return image, target

class ToTensor(object):
    def __call__(self, img, target):
        for key, value in target.items():
            target[key] = torch.tensor(value)
        return F.to_tensor(img), target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

