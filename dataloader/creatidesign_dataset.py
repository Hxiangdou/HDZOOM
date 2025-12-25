import os
import json
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T
from datasets import load_dataset

# 尝试导入 pycocotools 用于解码 RLE
try:
    from pycocotools import mask as mask_utils
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("⚠️ Warning: 'pycocotools' not found. Segmentation masks will be ignored (fallback to bbox).")
    print("   Please install it via: pip install pycocotools")

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------
def find_nearest_bucket_size(input_width, input_height, mode="x64"):
    buckets = [
        (512, 2048), (512, 1984), (512, 1920), (512, 1856), (576, 1792), (576, 1728), 
        (576, 1664), (640, 1600), (640, 1536), (704, 1472), (704, 1408), (704, 1344), 
        (768, 1344), (768, 1280), (832, 1216), (832, 1152), (896, 1152), (896, 1088), 
        (960, 1088), (960, 1024), (1024, 1024), (1024, 960), (1088, 960), (1088, 896), 
        (1152, 896), (1152, 832), (1216, 832), (1280, 768), (1344, 768), (1408, 704), 
        (1472, 704), (1536, 640), (1600, 640), (1664, 576), (1728, 576), (1792, 576), 
        (1856, 512), (1920, 512), (1984, 512), (2048, 512)
    ]
    aspect_ratios = [w / h for (w, h) in buckets]
    
    asp = input_width / input_height
    diff = [abs(ar - asp) for ar in aspect_ratios]
    bucket_id = int(np.argmin(diff))
    gen_width, gen_height = buckets[bucket_id]
    
    return gen_width, gen_height

def img_transforms(image, size):
    """
    image: PIL Image (RGB)
    size: (H, W) tuple
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize & Normalize to [-1, 1] for Flux
    t = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return t(image)

def mask_transforms(mask, size):
    """
    mask: PIL Image (L) or numpy array
    size: (H, W) tuple
    """
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    if mask.mode != "L":
        mask = mask.convert("L")

    # Resize (Nearest for masks) & ToTensor [0, 1]
    t = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    return t(mask)

# -----------------------------------------------------------------------------
# 核心 Dataset 类
# -----------------------------------------------------------------------------
class DesignDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        resolution=512,
        condition_resolution=512,
        condition_resolution_scale_ratio=0.5,
        max_boxes_per_image=10,
        neg_condition_image='same',
        background_color='gray',
        use_bucket=True,
        box_confidence_th=0.0,
        split="train",
    ):
        print(f"Loading dataset: {dataset_name} (Split: {split})")
        self.dataset = load_dataset(dataset_name, split=split)
        num_samples = 10000 
        self.dataset = self.dataset.select(range(num_samples))
        print(f"Loaded {len(self.dataset)} samples")
        
        self.max_boxes = max_boxes_per_image
        self.resolution = resolution
        self.cond_res = condition_resolution
        self.cond_scale = condition_resolution_scale_ratio
        self.use_bucket = use_bucket
        self.neg_condition_image = neg_condition_image
        self.box_th = box_confidence_th

    def __len__(self):
        return len(self.dataset)

    def decode_rle(self, rle_data):
        """解码 RLE 数据为 Binary Mask (numpy)"""
        if not HAS_COCO or rle_data is None:
            return None
        
        try:
            # RLE 格式可能是 {"size": [H, W], "counts": "..."}
            # pycocotools 需要 counts 为 bytes 或正确格式的 list
            if isinstance(rle_data.get('counts'), str):
                # 编码转换: string -> bytes
                rle_data['counts'] = rle_data['counts'].encode('utf-8')
            
            mask = mask_utils.decode(rle_data)
            return mask # (H, W) uint8 0/1
        except Exception as e:
            # print(f"RLE Decode Error: {e}")
            return None

    def get_subject_crop(self, image_pil, anno_item, orig_w, orig_h):
        """
        从原图和标注中提取主体 Crop 和 Mask。
        如果存在 Segmentation，生成精确 Mask；否则生成 BBox Mask。
        """
        bbox = anno_item.get('bbox', [0, 0, orig_w, orig_h])
        seg = anno_item.get('segmentation', None)
        
        # 1. 解码全图 Mask
        full_mask = self.decode_rle(seg)
        
        # 2. 确定 Crop 区域 (xyxy)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(orig_w, x2); y2 = min(orig_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            # Fallback: Center Crop if bbox invalid
            cx, cy = orig_w // 2, orig_h // 2
            x1, y1 = cx - 128, cy - 128
            x2, y2 = cx + 128, cy + 128
            full_mask = None # BBox 坏了，mask 也不可信
            
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        # 3. Crop Image
        subject_img = image_pil.crop((x1, y1, x2, y2))
        
        # 4. Crop Mask
        if full_mask is not None:
            # 有分割：裁剪对应的 mask 区域
            crop_mask_np = full_mask[y1:y2, x1:x2]
            subject_mask = Image.fromarray((crop_mask_np * 255).astype(np.uint8), mode='L')
        else:
            # 无分割：全白 Mask (表示整个 BBox 都是物体)
            subject_mask = Image.new('L', (crop_w, crop_h), 255)
            
        return subject_img, subject_mask, full_mask

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # --- 1. Load Image ---
        # 兼容 key: 'image' (默认) 或 'original_image'
        image_source = sample.get('image') or sample.get('original_image')
        if image_source is None:
            # Fallback for debugging
            image_source = Image.new('RGB', (512, 512), (128, 128, 128))
            
        if not isinstance(image_source, Image.Image):
            image_source = Image.fromarray(image_source)
        image_source = image_source.convert("RGB")
        orig_w, orig_h = image_source.size
        img_id = str(sample.get('img_id', idx))

        # --- 2. Parse Annotations ---
        raw_anno = sample.get('annotations') or sample.get('metadata', "[]")
        if isinstance(raw_anno, str):
            try:
                annotations = json.loads(raw_anno)
            except:
                annotations = []
        else:
            annotations = raw_anno
            
        # 兼容: annotations 可能是 list (objects) 或 dict (img_info + annotations)
        if isinstance(annotations, dict):
            global_caption = annotations.get('global_caption', "")
            text_list = annotations.get('text_list', [])
            obj_list = annotations.get('annotations') or annotations.get('object_annotations', [])
        else:
            # 此时无法获取 text_list 或 global_caption，除非在其他字段
            global_caption = ""
            text_list = []
            obj_list = annotations if isinstance(annotations, list) else []

        # --- 3. Determine Resolution (Bucketing) ---
        if self.use_bucket:
            tgt_w, tgt_h = find_nearest_bucket_size(orig_w, orig_h)
            cond_w = int(tgt_w * self.cond_scale)
            cond_h = int(tgt_h * self.cond_scale)
        else:
            tgt_w, tgt_h = self.resolution, self.resolution
            cond_w, cond_h = self.cond_res, self.cond_res

        # --- 4. Process Objects & Layout ---
        valid_objs = []
        valid_masks = [] # 存储全图尺寸的 numpy mask
        
        # A. 普通对象
        for obj in obj_list:
            bbox = obj.get('bbox')
            if not bbox: continue
            
            # 过滤低置信度 (如果是预标注数据)
            score = obj.get('score', [1.0])
            if isinstance(score, list): score = score[0]
            if score < self.box_th: continue
            
            caption = obj.get('bbox_detail_description') or obj.get('class_name') or "object"
            
            # 尝试解码 Mask (用于 Layout Map)
            # 注意：这里我们暂不 resize mask，等到生成 map 时再 resize
            seg = obj.get('segmentation')
            mask_np = self.decode_rle(seg) if seg else None
            
            valid_objs.append({
                "bbox": bbox,
                "caption": caption,
                "mask": mask_np # (H_orig, W_orig) or None
            })

        # B. 文字对象 (Text)
        for txt in text_list:
            bbox = txt.get('bbox')
            if not bbox: continue
            content = txt.get('text', "")
            valid_objs.append({
                "bbox": bbox,
                "caption": f"text: {content}", # 特殊前缀
                "mask": None # 文字通常只有 bbox
            })

        # C. 随机选一个作为 Condition Subject
        if len(valid_objs) > 0:
            # 优先选有 mask 的普通对象，如果没有则随机
            candidates = [i for i, o in enumerate(valid_objs) if not o['caption'].startswith("text:")]
            if not candidates: candidates = range(len(valid_objs))
            
            subj_idx = random.choice(candidates)
            subj_anno = obj_list[subj_idx] if subj_idx < len(obj_list) else {} # 如果是 text 可能越界，text 通常不做 visual condition
            
            # 使用原始 annotation (含 RLE) 进行 Crop
            # 注意：我们需要回溯到原始 obj_list 来获取 segmentation 数据
            # 为了简化，我们直接从 valid_objs[subj_idx] 获取信息是不够的(mask已解码)，
            # 但我们已经有了解码后的 mask，可以直接用。
            
            # 重新构建用于 Crop 的信息
            sel_bbox = valid_objs[subj_idx]['bbox']
            sel_mask_np = valid_objs[subj_idx]['mask']
            
            # Crop Subject Image & Mask
            # 手动执行 Crop，因为 mask 已经是 numpy
            x1, y1, x2, y2 = [int(v) for v in sel_bbox]
            x1=max(0,x1); y1=max(0,y1); x2=min(orig_w,x2); y2=min(orig_h,y2)
            
            subj_pil = image_source.crop((x1, y1, x2, y2))
            
            if sel_mask_np is not None:
                crop_mask = sel_mask_np[y1:y2, x1:x2]
                subj_mask_pil = Image.fromarray((crop_mask * 255).astype(np.uint8), mode='L')
            else:
                subj_mask_pil = Image.new('L', subj_pil.size, 255)
                
        else:
            # 没有对象，Dummy Condition
            subj_pil = Image.new('RGB', (256, 256), (0,0,0))
            subj_mask_pil = Image.new('L', (256, 256), 0)

        # --- 5. Prepare Tensors ---
        
        # Main Image
        img_tensor = img_transforms(image_source, (tgt_h, tgt_w)) # Flux: [-1, 1]

        # Padding containers
        max_b = self.max_boxes
        boxes_padded = torch.zeros((max_b, 4))
        masks_padded = torch.zeros(max_b) # Existence mask
        captions_padded = []
        
        # 关键: Pixel-level Layout Maps
        # Shape: [Max_Boxes, H_tgt, W_tgt]
        masks_maps_padded = torch.zeros((max_b, tgt_h, tgt_w))

        # Fill Data
        num_valid = min(len(valid_objs), max_b)
        for i in range(num_valid):
            obj = valid_objs[i]
            
            # 1. BBox Norm
            x1, y1, x2, y2 = obj['bbox']
            boxes_padded[i] = torch.tensor([
                x1/orig_w, y1/orig_h, x2/orig_w, y2/orig_h
            ])
            
            # 2. Caption
            captions_padded.append(obj['caption'])
            
            # 3. Existence
            masks_padded[i] = 1.0
            
            # 4. Mask Map (Resize original mask to target size)
            if obj['mask'] is not None:
                # Numpy (H_orig, W_orig) -> PIL -> Resize -> Tensor
                m_pil = Image.fromarray((obj['mask']*255).astype(np.uint8), mode='L')
                # Resize (Nearest)
                m_resized = m_pil.resize((tgt_w, tgt_h), resample=Image.NEAREST)
                # To Tensor [0, 1]
                m_tensor = T.functional.to_tensor(m_resized).squeeze(0) # (H, W)
                masks_maps_padded[i] = m_tensor
            else:
                # Fallback to BBox Rectangle Mask
                x1_p = int((x1/orig_w) * tgt_w)
                y1_p = int((y1/orig_h) * tgt_h)
                x2_p = int((x2/orig_w) * tgt_w)
                y2_p = int((y2/orig_h) * tgt_h)
                x1_p=max(0,x1_p); y1_p=max(0,y1_p); x2_p=min(tgt_w,x2_p); y2_p=min(tgt_h,y2_p)
                masks_maps_padded[i, y1_p:y2_p, x1_p:x2_p] = 1.0

        # Subject Tensors
        subj_tensor = img_transforms(subj_pil, (cond_h, cond_w))
        subj_mask_tensor = mask_transforms(subj_mask_pil, (cond_h, cond_w))
        orig_size_subj_tensor = img_transforms(subj_pil, (tgt_h, tgt_w)) # For visualization
        
        # Negative Subject
        if self.neg_condition_image == 'black':
            neg_pil = Image.new('RGB', subj_pil.size, (0,0,0))
            neg_tensor = img_transforms(neg_pil, (cond_h, cond_w))
        elif self.neg_condition_image == 'white':
            neg_pil = Image.new('RGB', subj_pil.size, (255,255,255))
            neg_tensor = img_transforms(neg_pil, (cond_h, cond_w))
        else: # same
            neg_tensor = subj_tensor.clone()

        return {
            "id": img_id,
            "caption": global_caption,
            "img": img_tensor,
            
            # Layout
            "objects_boxes": boxes_padded,           # [10, 4]
            "objects_caption": captions_padded,      # list len<=10
            "objects_masks": masks_padded,           # [10] (is_valid)
            "objects_masks_maps": masks_maps_padded, # [10, H, W] (Pixel Mask)
            
            # Subject Condition
            "condition_img": subj_tensor,                     # [3, Hc, Wc]
            "condition_img_masks_maps": subj_mask_tensor,     # [1, Hc, Wc]
            "neg_condtion_img": neg_tensor,                   # [3, Hc, Wc]
            "original_size_condition_img": orig_size_subj_tensor,
            
            # Info
            "img_info": {"width": orig_w, "height": orig_h},
            "target_width": tgt_w,
            "target_height": tgt_h
        }

def collate_fn(batch):
    out = {}
    
    # List fields
    for key in ['id', 'caption', 'objects_caption', 'img_info', 'target_width', 'target_height']:
        out[key] = [b[key] for b in batch]
        
    # Stack Tensors
    tensor_keys = [
        'img', 'objects_boxes', 'objects_masks', 'objects_masks_maps',
        'condition_img', 'condition_img_masks_maps', 'neg_condtion_img', 
        'original_size_condition_img'
    ]
    for key in tensor_keys:
        out[key] = torch.stack([b[key] for b in batch])
        
    return out

def tensor_to_pil(t):
    # [-1, 1] -> [0, 1] -> PIL
    t = t.detach().cpu()
    t = t * 0.5 + 0.5
    t = torch.clamp(t, 0, 1)
    return T.ToPILImage()(t)