from random import uniform

import os 
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from IPython.core.debugger import set_trace 
from dataloader.creatidesign_dataset_benchmark import DesignDataset,visualize_bbox,collate_fn,tensor_to_pil,make_image_grid_RGB
import numpy as np
from PIL import Image
from safetensors.torch import save_file, load_file
from accelerate import load_checkpoint_and_dispatch
from modules.flux.transformer_flux_creatidesign import FluxTransformer2DModel
from pipeline.pipeline_flux_creatidesign import FluxPipeline
import json
from huggingface_hub import snapshot_download
from modules.flux.attention_processor_flux_creatidesign import (
    FluxInvertedSwinPostProcessor, 
    Attention,
    DesignFluxAttnProcessor2_0 # 或者是模型当前使用的 Processor 类
)
from diffusers.optimization import get_scheduler
import math
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import logging
from diffusers import FluxPipeline
from diffusers.optimization import get_scheduler
from accelerate.logging import get_logger

logger = get_logger(__name__)

from modules.flux.attention_processor_flux_creatidesign import (
    FluxInvertedSwinPostProcessor,
    Attention
)
from config import Config
import random
config = Config()


def save_adapter_weights(accelerator, model, output_dir, step):
    """
    仅保存 InvertedSwinModule 的权重，不保存整个 Flux 模型。
    """
    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)
        
        # 获取未封装的模型 (去除 DDP/Accelerate 包装)
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = {}
        count = 0
        
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, FluxInvertedSwinPostProcessor):
                prefix = f"{name}.swin_module"
                # 仅提取 swin_module 的参数
                for param_name, param in module.swin_module.state_dict().items():
                    state_dict[f"{prefix}.{param_name}"] = param.cpu()
                count += 1
        
        if count > 0:
            torch.save(state_dict, os.path.join(save_path, "swin_adapter.pt"))
            logger.info(f"💾 [Step {step}] Saved {count} adapter modules to {save_path}")
        else:
            logger.warning("⚠️ No adapter modules found to save!")


def main():

    print(f"🚀正在加载模型: HDZOOM")
    model_path = "./black-forest-labs/FLUX.1-dev"

    ckpt_repo = "HuiZhang0812/CreatiDesign" # huggingface repo of ckpt

    ckpt_path = snapshot_download(
        repo_id=ckpt_repo,
        repo_type="model",
        local_dir="./CreatiDesign_checkpoint",
        local_dir_use_symlinks=False
    )

    # Load transformer config from checkpoint
    with open(os.path.join(ckpt_path, "transformer", "config.json"), 'r') as f:
        transformer_config = json.load(f)
    
    transformer = FluxTransformer2DModel(**transformer_config)
    transformer = load_checkpoint_and_dispatch(transformer, checkpoint=os.path.join(model_path,"transformer"), device_map=None)

    # Load lora parameters using safetensors
    state_dict = load_file(os.path.join(ckpt_path, "transformer","model.safetensors"))

    # Load parameters, allow partial loading
    missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded parameters: {len(state_dict)}",state_dict.keys())
    print(f"Missing keys: {len(missing_keys)}",missing_keys)
    print(f"Unexpected keys: {len(unexpected_keys)}",unexpected_keys)

    transformer = transformer.to(dtype=torch.bfloat16)

    pipe = FluxPipeline.from_pretrained(model_path, transformer=transformer,torch_dtype=torch.bfloat16)
    
    # Latent 分辨率 (Flux 经过 VAE 8x 和 Patch 2x，总共缩小 16 倍)
    latent_resolution = (config.resolution // 16, config.resolution // 16)  # 结果为 (64, 64)
    
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
    pipe = pipe.to("cuda")

    print("❄️ 正在冻结基础模型，仅解冻 HDZOOM 模块...")
    
    # 全局冻结
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    
    # 局部解冻
    trainable_params = []
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, FluxInvertedSwinPostProcessor):
            # 仅解冻 swin_module
            module.swin_module.requires_grad_(True)
            trainable_params.extend(module.swin_module.parameters())

    print(f"🔥 可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=1e-2
    )

    print("📦 准备数据...")
    # benchmark_repo = 'HuiZhang0812/CreatiDesign_benchmark' #  huggingface repo of benchmark
    benchmark_repo = '/home/fength/.cache/huggingface/datasets/HuiZhang0812___creati_design_benchmark/default/0.0.0/63fb381622f01b2f3ee11e56f0a1a017d52a843d/'
    # benchmark_repo = 'HuiZhang0812/CreatiDesign_benchmark' #  huggingface repo of benchmark
    train_datasets = DesignDataset(dataset_name=benchmark_repo,
                             resolution=config.resolution,
                             condition_resolution=config.condition_resolution,
                             neg_condition_image =config.neg_condition_image,
                             background_color=config.background_color,
                             use_bucket=config.use_bucket,
                             condition_resolution_scale_ratio=config.condition_resolution_scale_ratio,
                             split="test",
                             )
    train_dataloader = DataLoader(train_datasets, batch_size=config.batch_size, shuffle=True, num_workers=4,collate_fn=collate_fn)
    test_datasets = DesignDataset(dataset_name=benchmark_repo,
                                resolution=config.resolution,
                                condition_resolution=config.condition_resolution,
                                neg_condition_image =config.neg_condition_image,
                                background_color=config.background_color,
                                use_bucket=config.use_bucket,
                                condition_resolution_scale_ratio=config.condition_resolution_scale_ratio,
                                split="test",
                                )
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4,collate_fn=collate_fn)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    max_train_steps = config.num_epochs * num_update_steps_per_epoch

    print(f"🧮 预计总更新步数: {max_train_steps} (预热: {config.lr_warmup_steps})")
    
    
    
    # 初始化 Accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=os.path.join(config.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision, # 读取 config.py 中的 "bf16"
        log_with="tensorboard",
        project_config=accelerator_project_config
    )
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes, # 如果是多卡，预热步数可能需要调整，通常保持原值即可
        num_training_steps=max_train_steps,
    )
    
    pipe.vae.to(accelerator.device)
    pipe.text_encoder.to(accelerator.device)
    pipe.text_encoder_2.to(accelerator.device)
    # 初始化日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(accelerator.state)

    # 准备模型和优化器
    
    pipe.transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipe.transformer, optimizer, train_dataloader, lr_scheduler
    )


    # 开始训练循环
    if accelerator.is_main_process:
        print("🚀 Accelerate 训练环境已启动！")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Grad Accumulation: {config.gradient_accumulation_steps}")
        print(f"   Mixed Precision: {config.mixed_precision}")

    global_step = 0
    pipe.transformer.train()

    for epoch in range(config.num_epochs):
        # 进度条
        progress_bar = tqdm(
            total=len(train_dataloader), 
            disable=not accelerator.is_local_main_process, 
            desc=f"Epoch {epoch}"
        )
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(pipe.transformer):
                
                with torch.no_grad():
                    imgs = batch['img'].to(accelerator.device, dtype=pipe.dtype) 
                
                    prompts = batch["caption"] 
                    imgs_id = batch['id']
                    objects_boxes = batch["objects_boxes"]
                    objects_caption = batch['objects_caption'] 
                    objects_masks = batch['objects_masks']
                    condition_img = batch['condition_img']
                    neg_condition_img = batch['neg_condtion_img']
                    objects_masks_maps= batch['objects_masks_maps']
                    subject_masks_maps = batch['condition_img_masks_maps']
                    target_width=batch['target_width'][0]
                    target_height=batch['target_height'][0]

                    img_info = batch["img_info"][0] 
                    filename = img_info["img_id"]+'.jpg'
                    start_time = time.time()

                    # <--- 新增: CFG Dropout 逻辑 (随机丢弃条件)
                    # 假设 10% 的概率丢弃条件 (训练 unconditional 分支)
                    if config.cfg_dropout_prob > 0:
                        # 生成一个 mask, True 表示丢弃条件
                        dropout_mask = torch.rand(B, device=accelerator.device) < config.cfg_dropout_prob
                        
                        # 1. 替换 Caption 为空字符串
                        captions = ["" if drop else cap for drop, cap in zip(dropout_mask, captions)]
                        
                        # 2. 替换 Condition Image 为 Negative Image (通常是全黑/全白/全灰)
                        # dropout_mask reshape for broadcast: [B] -> [B, 1, 1, 1]
                        mask_broadcast = dropout_mask[:, None, None, None].to(dtype=pipe.dtype)
                        final_condition_imgs = (1 - mask_broadcast) * condition_img + mask_broadcast * neg_condition_img
                    else:
                        final_condition_imgs = condition_img
                    # VAE 编码 (Pixel -> Latent)
                    latents = pipe.vae.encode(imgs).latent_dist.sample()
                    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                    
                    condition_latents = pipe.vae.encode(final_condition_imgs).latent_dist.sample()
                    condition_latents = (condition_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                    # Latent 变形 (Packing)
                    # Flux 需要 (B, L, C) 格式
                    B, C, H, W = latents.shape
                    # 简单展平 (注意：Flux 官方有更复杂的 patch 打包，这里使用简化版通过 Reshape)
                    # 如果这步报错，说明需要引入官方的 _pack_latents 函数
                    latents = latents.view(B, C, -1).permute(0, 2, 1) # (B, H*W, C)

                    # 文本编码 (T5 + CLIP)
                    # 使用 Pipeline 的 encode_prompt 方法
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                        tokenizer, text_encoder, tokenizer_2, text_encoder_2, 
                        captions, max_sequence_length=512, device=accelerator.device
                    )
                    
                    # 生成噪声和时间步
                    noise = torch.randn_like(latents)
                    # 随机时间步
                    timesteps = torch.rand((B,), device=accelerator.device)
                    
                    # 加噪 (Flow Matching: x_t = (1-t)x_0 + t*noise)
                    noisy_latents = (1 - timesteps.view(B, 1, 1)) * latents + timesteps.view(B, 1, 1) * noise

                    # 准备 img_ids
                    img_ids = pipe._prepare_latent_image_ids(B, config.resolution, config.resolution, accelerator.device, config.torch_dtype)

                # 前向传播与反向传播
                noisy_latents.requires_grad_(True)
                
                # Predict Noise / Velocity
                model_pred = pipe.transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps, # Flux transformer 接受 float timesteps
                    guidance=torch.tensor([1.0]*B, device=accelerator.device), # 训练时通常 guidance=1
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False
                )[0]

                target = noise - latents
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipe.transformer.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step() 
                optimizer.zero_grad()

            # 日志与保存
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # 打印日志
                if global_step % 10 == 0:
                    logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")
                    accelerator.log({"loss": loss.item()}, step=global_step)

                # 保存权重
                if global_step % config.save_steps == 0:
                    if accelerator.is_main_process:
                        save_adapter_weights(accelerator, pipe.transformer, config.output_dir, global_step)

    # 训练结束保存
    save_adapter_weights(accelerator, pipe.transformer, config.output_dir, "final")
    accelerator.end_training()
    
    print("🎉 训练结束！")

if __name__ == "__main__":
    main()