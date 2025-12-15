#!/bin/bash

lr=6e-5
dropout=0.3
weight_decay=2e-4
batchsize=64
dataset_path="/data/LATEX-EnxText-new/val/"
output_dir="/data/fength/hdlayoutv2/test/"
device="cuda:0"
epochs=100
# resume="/data/fength/hdlayoutv2/20250909_232505/checkpoint.pth" # catblock
resume="/data/fength/hdlayoutv2/20250911_150952/checkpoint.pth"

python inference.py \
    --lr "${lr}" \
    --dropout "${dropout}" \
    --batch_size "${batchsize}" \
    --weight_decay "${weight_decay}" \
    --no_aux_loss \
    --device "${device}" \
    --epochs "${epochs}" \
    --dataset_path "${dataset_path}" \
    --output_dir "${output_dir}" \
    --resume "${resume}"