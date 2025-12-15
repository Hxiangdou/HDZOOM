import torch
import os
import json
from PIL import Image
import numpy as np
from torchvision import transforms
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from accelerate import load_checkpoint_and_dispatch

from modules.flux.attention_processor_flux_creatidesign import (
    FluxInvertedSwinPostProcessor, 
    Attention,
    DesignFluxAttnProcessor2_0 # 或者是模型当前使用的 Processor 类
)
from modules.flux.transformer_flux_creatidesign import FluxTransformer2DModel
from pipeline.pipeline_flux_creatidesign import FluxPipeline
from dataloader.creatidesign_dataset_benchmark import visualize_bbox, make_image_grid_RGB, tensor_to_pil


def img_transforms(image, height=512, width=512):
    """处理图像：Resize -> ToTensor -> Normalize"""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    transform = transforms.Compose(
        [
            transforms.Resize(
                (height, width), interpolation=transforms.InterpolationMode.BILINEAR  
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transform(image)

def mask_transforms(mask, height=512, width=512):
    """处理掩码：Resize -> ToTensor"""
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(mask)
    if mask.mode != "L":
        mask = mask.convert("L")
        
    transform = transforms.Compose(
        [
            transforms.Resize(
                (height, width), 
                interpolation=transforms.InterpolationMode.NEAREST   
            ),
            transforms.ToTensor(),
        ]
    )
    return transform(mask)

def prepare_single_input(
    global_prompt,
    condition_image_path,
    layout_data,
    target_height=1024,
    target_width=1024,
    condition_resolution_scale_ratio=0.5,
    max_boxes_per_image=10,
    mask_path=None,
    device="cuda",
    dtype=torch.bfloat16
):
    """
    将用户输入转换为模型推理所需的 Batch 格式
    
    Args:
        global_prompt (str): 全局文本提示
        condition_image_path (str): 主体/条件图片的路径
        layout_data (list): 包含布局信息的列表，每个元素为 dict: {'bbox': [x1, y1, x2, y2], 'prompt': 'description'}
                            bbox 应该是归一化坐标 [0.0 - 1.0]
        mask_path (str, optional): 主体掩码图片路径。如果不提供，默认全白（全选）。
    """
    
    # 条件图像
    condition_img_pil = Image.open(condition_image_path).convert("RGB")
    
    condition_height = int(target_height * condition_resolution_scale_ratio)
    condition_width = int(target_width * condition_resolution_scale_ratio)

    subject_tensor = img_transforms(condition_img_pil, height=condition_height, width=condition_width)
    original_size_subject_tensor = img_transforms(condition_img_pil, height=target_height, width=target_width)
    
    # 掩码
    if mask_path and os.path.exists(mask_path):
        mask_pil = Image.open(mask_path).convert("L")
    else:
        # 如果没有提供掩码，默认整个条件图像都是主体
        mask_pil = Image.new("L", condition_img_pil.size, 255)
        
    subject_mask_tensor = mask_transforms(mask_pil, height=condition_height, width=condition_width)

    # 负向条件图像
    # 也可以选择 'same' 使用原图，或者 'black'/'white'
    subject_image_gray = Image.new('RGB', condition_img_pil.size, (128, 128, 128))
    subject_image_neg_tensor = img_transforms(subject_image_gray, height=condition_height, width=condition_width)

    # Layout (Bounding Boxes & Captions)
    objects_boxes_padded = torch.zeros((max_boxes_per_image, 4))
    objects_masks_padded = torch.zeros(max_boxes_per_image)
    objects_masks_maps_padded = torch.zeros((max_boxes_per_image, target_height, target_width))
    
    processed_captions = []
    
    num_boxes = min(len(layout_data), max_boxes_per_image)
    
    for idx in range(num_boxes):
        item = layout_data[idx]
        bbox = item['bbox'] # [x1, y1, x2, y2] normalized
        caption = item['prompt']
        
        processed_captions.append(caption)
        
        # 填充 Bbox Tensor
        objects_boxes_padded[idx] = torch.tensor(bbox)
        objects_masks_padded[idx] = 1.0
        
        # 生成 Mask Map (在目标分辨率下的二进制掩码)
        x1, y1, x2, y2 = bbox
        x1_pixel = int(x1 * target_width)
        y1_pixel = int(y1 * target_height)
        x2_pixel = int(x2 * target_width)
        y2_pixel = int(y2 * target_height)
        
        # 边界裁剪
        x1_pixel = max(0, min(x1_pixel, target_width - 1))
        y1_pixel = max(0, min(y1_pixel, target_height - 1))
        x2_pixel = max(0, min(x2_pixel, target_width - 1))
        y2_pixel = max(0, min(y2_pixel, target_height - 1))
        
        objects_masks_maps_padded[idx, y1_pixel:y2_pixel+1, x1_pixel:x2_pixel+1] = 1.0

    # 补充空标题以匹配 max_boxes
    while len(processed_captions) < max_boxes_per_image:
        processed_captions.append("")

    # 组装 Batch (增加 Batch 维度)
    batch = {
        "caption": [global_prompt], # List for pipeline
        "objects_boxes": objects_boxes_padded.unsqueeze(0), # [1, 10, 4]
        "objects_caption": [processed_captions], # List of List
        "objects_masks": objects_masks_padded.unsqueeze(0), # [1, 10]
        "objects_masks_maps": objects_masks_maps_padded.unsqueeze(0), # [1, 10, H, W]
        "condition_img": subject_tensor.unsqueeze(0), # [1, 3, h, w]
        "condition_img_masks_maps": subject_mask_tensor.unsqueeze(0), # [1, 1, h, w]
        "neg_condtion_img": subject_image_neg_tensor.unsqueeze(0),
        "target_width": [target_width],
        "target_height": [target_height],
        # 用于后处理可视化的原始尺寸张量
        "original_size_condition_img": original_size_subject_tensor.unsqueeze(0) 
    }
    
    return batch

def hdzoom_load_model(model_path, ckpt_repo, weight_dtype, resolution=1024):
    print(f"Loading model checkpoint from {ckpt_repo}...")
    ckpt_path = snapshot_download(
        repo_id=ckpt_repo,
        repo_type="model",
        local_dir="./CreatiDesign_checkpoint",
        local_dir_use_symlinks=False
    )

    # Load config
    with open(os.path.join(ckpt_path, "transformer", "config.json"), 'r') as f:
        config = json.load(f)
    
    # Load Transformer
    transformer = FluxTransformer2DModel(**config)
    transformer = load_checkpoint_and_dispatch(
        transformer, 
        checkpoint=os.path.join(model_path, "transformer"), 
        device_map=None
    )

    # Load LoRA weights
    state_dict = load_file(os.path.join(ckpt_path, "transformer", "model.safetensors"))
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"Loaded parameters: {len(state_dict)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    transformer = transformer.to(dtype=weight_dtype)

    # Load Pipeline
    pipe = FluxPipeline.from_pretrained(
        model_path, 
        transformer=transformer, 
        torch_dtype=weight_dtype
    )
    
    latent_resolution = (resolution // 16, resolution // 16)  # 结果为 (64, 64)
    
    for name, module in pipe.transformer.named_modules():
        # 仅针对 SingleTransformerBlock 中的 Attention (自注意力)
        if "single_transformer_blocks" in name and isinstance(module, Attention):
        
            current_processor = module.processor
            
            dim = module.out_dim if module.out_dim is not None else module.query_dim
            # 对于 Flux.1-dev，这里通常是 3072
            dim = min(dim, 192)  # 确保不超过 4096
            window_size = 8
            
            # 检查分辨率是否匹配窗口
            if latent_resolution[0] % window_size != 0 or latent_resolution[1] % window_size != 0:
                print(f"警告: 分辨率 {latent_resolution} 不能被 window_size {window_size} 整除，可能会报错。")
            
            swin_num_heads = module.heads  
            if dim % swin_num_heads != 0:
                raise ValueError(f"维度 {dim} 无法被头数 {swin_num_heads} 整除")

            # E. 设定深度
            depths = [2, 2, 2, 2]  # 根据显存情况调整

            # print(f"正在注入层: {name}")
            # print(f"  - Dim: {dim}")
            # print(f"  - Resolution: {latent_resolution}")
            
            # --- 实例化并替换 ---
            
            # 实例化你的 Wrapper Processor
            swin_wrapper = FluxInvertedSwinPostProcessor(
                base_processor=current_processor,
                in_dim=dim,
                input_resolution=latent_resolution,
                depths=depths,
                num_heads=swin_num_heads,
                window_size=window_size
            ).to(pipe.device, dtype=pipe.dtype)
            # print(f"  - 使用的 Processor: {swin_wrapper.__class__.__name__}")
            # 替换
            module.set_processor(swin_wrapper)
    return pipe

def hdzoom_inference(
    model_path, 
    ckpt_repo, 
    resolution, 
    seed, 
    num_inference_steps, 
    guidance_scale, 
    true_cfg_scale,
    my_condition_image_path, 
    mask_path, 
    my_prompt, 
    my_layout, 
    output_dir, 
    device, 
    weight_dtype,
    pipe=None
):
    if pipe is None:
        pipe = hdzoom_load_model(model_path, ckpt_repo, weight_dtype)

    pipe = pipe.to(device)

    # 准备输入数据
    if not os.path.exists(my_condition_image_path):
        print(f"Warning: Image path '{my_condition_image_path}' not found. Please set a valid path in the config area.")
        # 创建一个假图片用于测试防止报错
        Image.new('RGB', (512, 512), color='red').save("dummy_test.jpg")
        my_condition_image_path = "dummy_test.jpg"

    print("Processing inputs...")
    batch = prepare_single_input(
        global_prompt=my_prompt,
        condition_image_path=my_condition_image_path,
        layout_data=my_layout,
        target_height=resolution,
        target_width=resolution,
        mask_path=mask_path,
        device=device,
        dtype=weight_dtype
    )

    # 提取 Batch 数据
    prompts = batch["caption"]
    objects_boxes = batch["objects_boxes"]
    objects_caption = batch['objects_caption'] 
    objects_masks = batch['objects_masks']
    condition_img = batch['condition_img']
    neg_condtion_img = batch['neg_condtion_img']
    objects_masks_maps= batch['objects_masks_maps']
    subject_masks_maps = batch['condition_img_masks_maps']
    
    # 参数计算
    condition_resolution = int(resolution * 0.5)
    if resolution == 512:
        position_delta = [0, -32]
    else:
        position_delta = [0, -64]
        
    scale_h = 1 / 0.5 # condition_resolution_scale_ratio
    scale_w = 1 / 0.5
    
    print("Running inference...")
    with torch.no_grad():
        images = pipe(
            prompt=prompts,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_inference_steps=num_inference_steps,
            objects_boxes=objects_boxes,
            objects_caption=objects_caption,
            objects_masks=objects_masks,
            objects_masks_maps=objects_masks_maps,
            condition_img=condition_img,
            subject_masks_maps=subject_masks_maps,
            neg_condtion_img=neg_condtion_img,
            height=resolution,
            width=resolution,
            true_cfg_scale=true_cfg_scale,
            position_delta=position_delta,
            guidance_scale=guidance_scale,
            scale_h=scale_h,
            scale_w=scale_w,
            use_bucket=True # 强制开启 bucket 逻辑以匹配 dataloader
        )
    
    images = images.images
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "result.jpg")
    images[0].save(save_path)
    print(f"Image saved to {save_path}")
    
    # --- 可视化 (生成带框的对比图) ---
    print("Generating visualization...")
    
    # 原始条件图 (Original Condition)
    ori_image = tensor_to_pil(batch['original_size_condition_img'][0])
    
    # 准备可视化用的 Bbox 结构
    # 将归一化坐标还原回像素坐标
    denormalized_boxes = []
    box_tensor = batch['objects_boxes'][0].cpu().numpy()
    for box in box_tensor:
        x1, y1, x2, y2 = box
        denorm_box = [
            x1 * resolution, 
            y1 * resolution, 
            x2 * resolution, 
            y2 * resolution
        ]
        denormalized_boxes.append(denorm_box)
        
    objects_result = {
        "boxes": denormalized_boxes,
        "labels": batch['objects_caption'][0],
        "masks": []
    }
    
    # 过滤掉无效的框 (mask=0)
    valid_boxes = []
    valid_labels = []
    masks_tensor = batch['objects_masks'][0]
    for i, m in enumerate(masks_tensor):
        if m > 0.5:
            valid_boxes.append(objects_result['boxes'][i])
            valid_labels.append(objects_result['labels'][i])
    
    objects_result['boxes'] = valid_boxes
    objects_result['labels'] = valid_labels

    # 绘制
    # 生成图带框
    gen_image_with_bbox = visualize_bbox(images[0].copy(), objects_result)
    
    # 拼接: [条件图] [生成图] [生成图带框]
    # 这里我们只展示 生成图 和 生成图带框
    vis_image = Image.new('RGB', (resolution * 2, resolution))
    vis_image.paste(images[0], (0, 0))
    vis_image.paste(gen_image_with_bbox, (resolution, 0))
    
    vis_save_path = os.path.join(output_dir, "result_vis.jpg")
    vis_image.save(vis_save_path)
    print(f"Visualization saved to {vis_save_path}")
    return

if __name__ == "__main__":
    
    model_path = "./black-forest-labs/FLUX.1-dev" # 基础模型
    ckpt_repo = "HuiZhang0812/CreatiDesign"     # Checkpoint 仓库

    resolution = 1024 # 生成图像分辨率
    seed = 42
    num_inference_steps = 28
    guidance_scale = 1.0
    true_cfg_scale = 3.5
    
    my_condition_image_path = "/home/fength/CreatiDesign/test_data/infer-removebg-preview_condition.jpg" 
    mask_path = "/home/fength/CreatiDesign/test_data/infer-removebg-preview_mask.jpg"
    my_prompt = "A wooden table with Text:\"What Can I SAY\", in a bright room."
    
    # 布局[x1, y1, x2, y2] 坐标范围 0.0 到 1.0
    my_layout = [
        {"bbox": [0.1, 0.1, 0.8, 0.2], "prompt": "Text:\"What Can I SAY\""},
    ]
    
    output_dir = "outputs/custom_inference"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16
    hdzoom_inference(model_path, ckpt_repo, resolution, seed, num_inference_steps, guidance_scale, true_cfg_scale,
                     my_condition_image_path, mask_path, my_prompt, my_layout, output_dir, device, weight_dtype)
