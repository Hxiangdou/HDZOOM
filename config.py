# config.py
import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:

    model: str = "./black-forest-labs/FLUX.1-dev"
    output_dir: str = "/data/fength/swin_adapter_v1/outputs/"
    mixed_precision: str = "bf16"
    # dataset_name: str = "HuiZhang0812/CreatiDesign_dataset"
    dataset_name: str = "/home/fength/.cache/huggingface/datasets/HuiZhang0812___creati_design_benchmark/default/0.0.0/63fb381622f01b2f3ee11e56f0a1a017d52a843d/"
    
    # Flux 推荐 1024
    resolution: int = 1024
    condition_resolution: int = 512
    neg_condition_image: str = "same"
    background_color: str = "gray"
    use_bucket: bool = True
    condition_resolution_scale_ratio: float = 0.5
    
    num_workers: int = 4
    
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    save_steps: int = 500
    seed: int = 42
    cfg_dropout_prob: float = 0.1

    window_size: int = 8
    swin_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    swin_num_heads: int = 4
    
    # 调度器类型: 
    # "constant" (保持不变)
    # "constant_with_warmup" (推荐: 线性预热后保持不变)
    # "cosine" (余弦退火)
    # "linear" (线性下降)
    lr_scheduler: str = "constant_with_warmup"
    
    lr_warmup_steps: int = 500
    @property
    def latent_resolution(self):
        """计算 VAE 压缩后的 Latent 分辨率 (Flux 为 1/16)"""
        h = self.resolution // 16
        w = self.resolution // 16
        return (h, w)
    
    @property
    def torch_dtype(self):
        """获取 Torch 数据类型"""
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.mixed_precision == "fp16":
            return torch.float16
        return torch.float32
