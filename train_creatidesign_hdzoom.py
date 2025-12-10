import os
import time
import math
import json
import logging
import random
from random import uniform

# 设置环境变量
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download

# Accelerate & Diffusers
from accelerate import Accelerator, load_checkpoint_and_dispatch
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler

# 自定义模块
from modules.flux.transformer_flux_creatidesign import FluxTransformer2DModel
from pipeline.pipeline_flux_creatidesign import FluxPipeline
from modules.flux.attention_processor_flux_creatidesign import (
    FluxInvertedSwinPostProcessor, 
    Attention
)

# 本地模块
from dataloader.creatidesign_dataset_benchmark import DesignDataset, collate_fn
from config import Config

logger = get_logger(__name__)
config = Config()

def save_adapter_weights(accelerator, model, output_dir, step):
    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = {}
        count = 0
        
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, FluxInvertedSwinPostProcessor):
                prefix = f"{name}.swin_module"
                for param_name, param in module.swin_module.state_dict().items():
                    state_dict[f"{prefix}.{param_name}"] = param.cpu()
                count += 1
        
        if count > 0:
            torch.save(state_dict, os.path.join(save_path, "swin_adapter.pt"))
            logger.info(f"💾 [Step {step}] Saved {count} adapter modules to {save_path}")

def main():
    # 1. 初始化 Accelerate
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, 
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_main_process:
        print(f"🚀 正在加载模型: HDZOOM")
        os.makedirs(config.output_dir, exist_ok=True)

    # 2. 准备模型
    model_path = config.model
    ckpt_repo = "HuiZhang0812/CreatiDesign" 

    if accelerator.is_main_process:
        ckpt_path = snapshot_download(
            repo_id=ckpt_repo,
            repo_type="model",
            local_dir="./CreatiDesign_checkpoint",
            local_dir_use_symlinks=False
        )
    accelerator.wait_for_everyone()
    ckpt_path = "./CreatiDesign_checkpoint"

    with open(os.path.join(ckpt_path, "transformer", "config.json"), 'r') as f:
        transformer_config = json.load(f)
    
    transformer = FluxTransformer2DModel(**transformer_config)
    
    # 加载权重
    state_dict = load_file(os.path.join(ckpt_path, "transformer", "model.safetensors"))
    transformer.load_state_dict(state_dict, strict=False)
    transformer = transformer.to(dtype=config.torch_dtype)

    # 加载 Pipeline
    pipe = FluxPipeline.from_pretrained(
        model_path, 
        transformer=transformer, 
        torch_dtype=config.torch_dtype
    )
    
    # 3. 注入 Swin Adapter
    latent_resolution = config.latent_resolution
    
    for name, module in pipe.transformer.named_modules():
        if "single_transformer_blocks" in name and isinstance(module, Attention):
            current_processor = module.processor
            dim = module.out_dim if module.out_dim is not None else module.query_dim
            dim = min(dim, 192)
            
            swin_wrapper = FluxInvertedSwinPostProcessor(
                base_processor=current_processor,
                in_dim=dim,
                input_resolution=latent_resolution,
                depths=config.swin_depths,
                num_heads=config.swin_num_heads,
                window_size=config.window_size
            ).to(dtype=config.torch_dtype)
            
            module.set_processor(swin_wrapper)

    # 4. 冻结与解冻
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    
    trainable_params = []
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, FluxInvertedSwinPostProcessor):
            module.swin_module.requires_grad_(True)
            trainable_params.extend(module.swin_module.parameters())

    if accelerator.is_main_process:
        print(f"🔥 可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 5. 数据准备
    train_datasets = DesignDataset(
        dataset_name=config.dataset_name,
        resolution=config.resolution,
        condition_resolution=config.condition_resolution,
        neg_condition_image=config.neg_condition_image,
        background_color=config.background_color,
        use_bucket=config.use_bucket,
        condition_resolution_scale_ratio=config.condition_resolution_scale_ratio,
        split="test",
    )
    
    train_dataloader = DataLoader(
        train_datasets, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    max_train_steps = config.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps,
    )
    
    pipe.vae.to(accelerator.device)
    pipe.text_encoder.to(accelerator.device)
    pipe.text_encoder_2.to(accelerator.device)

    pipe.transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipe.transformer, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        print(f"🚀 开始训练! Total Steps: {max_train_steps}")

    global_step = 0
    pipe.transformer.train()

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), 
            disable=not accelerator.is_local_main_process, 
            desc=f"Epoch {epoch}"
        )
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(pipe.transformer):
                # 获取数据
                imgs = batch['img'].to(accelerator.device, dtype=config.torch_dtype) 
                prompts = batch["caption"]
                condition_img = batch['condition_img'].to(accelerator.device, dtype=config.torch_dtype)
                neg_condition_img = batch['neg_condtion_img'].to(accelerator.device, dtype=config.torch_dtype)
                
                # Layout related
                objects_boxes = batch["objects_boxes"].to(accelerator.device, dtype=config.torch_dtype)
                objects_masks = batch["objects_masks"].to(accelerator.device, dtype=config.torch_dtype)
                objects_masks_maps = batch["objects_masks_maps"].to(accelerator.device, dtype=config.torch_dtype)
                subject_masks_maps = batch["condition_img_masks_maps"].to(accelerator.device, dtype=config.torch_dtype)
                objects_caption = batch['objects_caption'] # List of lists

                B = imgs.shape[0]

                # --- CFG Dropout ---
                if config.cfg_dropout_prob > 0:
                    dropout_mask = torch.rand(B, device=accelerator.device) < config.cfg_dropout_prob
                    final_prompts = ["" if drop else p for drop, p in zip(dropout_mask.tolist(), prompts)]
                    mask_broadcast = dropout_mask[:, None, None, None].to(dtype=config.torch_dtype)
                    final_condition_imgs = (1 - mask_broadcast) * condition_img + mask_broadcast * neg_condition_img
                    # 注意：这里我们仅简单处理 Image Condition 的 Dropout，
                    # 完整的 Layout Dropout 逻辑比较复杂，这里暂时保持 Layout 输入不变
                else:
                    final_prompts = prompts
                    final_condition_imgs = condition_img

                with torch.no_grad():
                    # 1. VAE Encode Images (Target) -> [B, 16, H, W]
                    latents = pipe.vae.encode(imgs).latent_dist.sample()
                    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                    
                    # 2. VAE Encode Condition -> [B, 16, H_cond, W_cond]
                    condition_latents = pipe.vae.encode(final_condition_imgs).latent_dist.sample()
                    condition_latents = (condition_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

                    # 3. Pack Latents (Crucial Fix! 16 -> 64 Channels)
                    # Target Latents
                    h, w = latents.shape[2], latents.shape[3]
                    latents_packed = pipe._pack_latents(latents, B, latents.shape[1], h, w)
                    
                    # Condition Latents
                    h_c, w_c = condition_latents.shape[2], condition_latents.shape[3]
                    condition_packed = pipe._pack_latents(condition_latents, B, condition_latents.shape[1], h_c, w_c)

                    # 4. Prepare Image IDs (Needs packed dimensions)
                    img_ids = pipe._prepare_latent_image_ids(B, h // 2, w // 2, accelerator.device, config.torch_dtype)
                    
                    # Prepare Condition Image IDs
                    condition_ids = pipe._prepare_latent_image_ids(B, h_c // 2, w_c // 2, accelerator.device, config.torch_dtype)
                    # Shift ID (Simplified for batch)
                    if config.use_bucket:
                        condition_ids[:, 2] += -1 * (w_c // 2)
                    else:
                        condition_ids[:, 1] += 0 # position_delta logic if needed
                        condition_ids[:, 2] += -64

                    # 5. Text Encode
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=final_prompts,
                        prompt_2=final_prompts,
                        device=accelerator.device,
                        max_sequence_length=512
                    )

                    # 6. Encode Object Captions (For Layout Control)
                    # objects_caption is a list of B lists. We need to encode them.
                    max_boxes = config.max_boxes_per_image if hasattr(config, 'max_boxes_per_image') else 10
                    bbox_text_embeddings = torch.zeros(B, max_boxes, 512, 4096, device=accelerator.device, dtype=config.torch_dtype)
                    
                    for i in range(B):
                        caps = objects_caption[i] # List of strings
                        if len(caps) > 0:
                            # Encode list of strings
                            # 注意：pipe.encode_prompt 返回 (embeds, pooled, ids)
                            # embeds shape: [N_boxes, Seq_len, 4096]
                            b_embeds, _, _ = pipe.encode_prompt(
                                prompt=caps, 
                                prompt_2=caps, 
                                device=accelerator.device, 
                                max_sequence_length=512
                            )
                            num_boxes = min(len(caps), max_boxes)
                            bbox_text_embeddings[i, :num_boxes] = b_embeds[:num_boxes]

                    # 7. Add Noise (Flow Matching)
                    noise = torch.randn_like(latents_packed)
                    timesteps = torch.rand((B,), device=accelerator.device)
                    # Expand timesteps for broadcasting
                    t_expand = timesteps.view(B, 1, 1)
                    noisy_latents_packed = (1 - t_expand) * latents_packed + t_expand * noise

                # --- Construct Design Kwargs ---
                design_kwargs = {
                    "object_layout": {
                        "objects_boxes": objects_boxes, 
                        "bbox_text_embeddings": bbox_text_embeddings, 
                        "bbox_masks": objects_masks,
                        "objects_masks_maps": objects_masks_maps,
                        "img_token_h": h // 2, 
                        "img_token_w": w // 2
                    },
                    "subject_contion": {
                        "condition_img": condition_packed,
                        "subject_masks_maps": subject_masks_maps,
                        "condition_img_ids": condition_ids,
                        "subject_token_h": h_c // 2, 
                        "subject_token_w": w_c // 2
                    },
                }

                # --- Training Forward ---
                noisy_latents_packed.requires_grad_(True)
                
                model_pred = pipe.transformer(
                    hidden_states=noisy_latents_packed,
                    timestep=timesteps, 
                    guidance=torch.tensor([1.0]*B, device=accelerator.device),
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                    design_kwargs=design_kwargs # 传入 Control 参数
                )[0]

                # Loss Target: noise - data
                target = noise - latents_packed
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipe.transformer.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step() 
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % 10 == 0:
                    accelerator.log({"loss": loss.item()}, step=global_step)
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(loss=loss.item())

                if global_step % config.save_steps == 0:
                    save_adapter_weights(accelerator, pipe.transformer, config.output_dir, global_step)

    save_adapter_weights(accelerator, pipe.transformer, config.output_dir, "final")
    accelerator.end_training()
    
    if accelerator.is_main_process:
        print("🎉 训练结束！")

if __name__ == "__main__":
    main()