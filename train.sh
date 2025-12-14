#!/bin/bash


export CUDA_VISIBLE_DEVICES="2,1"


NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


# 2. Accelerate 启动参数
# ------------------------------------------------------------------------------
# 混合精度: no, fp16, bf16 (Flux 强烈推荐使用 bf16)
MIXED_PRECISION="bf16"

# 3. 启动命令
# ------------------------------------------------------------------------------
echo "🚀 Starting training on $NUM_GPUS GPUs ..."
echo "   Mixed Precision: $MIXED_PRECISION"
echo "   Config File: config.py"

# 使用 accelerate launch 启动
# --multi_gpu: 启用多卡
# --num_processes: GPU 总数
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=$MIXED_PRECISION \
    --multi_gpu \
    --num_machines=1 \
    --dynamo_backend=no \
    train_creatidesign_hdzoom.py

# 注意：
# 如果遇到显存不足 (OOM)，请尝试：
# 1. 在 config.py 中减少 batch_size (设为 1)
# 2. 在 config.py 中增加 gradient_accumulation_steps
# 3. 使用 DeepSpeed (需要生成 deepspeed_config.yaml 并添加 --use_deepspeed 参数)