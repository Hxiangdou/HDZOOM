#!/bin/bash

lr=1e-4
dropout=0.3
weight_decay=2e-4
batchsize=32
dataset_path="/data/LATEX-EnxText-new/"
output_dir="/data1/fength/HDDiffuser-Text/outputs/"
device="cuda:0"
epochs=300

python main.py \
    --lr "${lr}" \
    --dropout "${dropout}" \
    --batch_size "${batchsize}" \
    --weight_decay "${weight_decay}" \
    --no_aux_loss \
    --device "${device}" \
    --epochs "${epochs}" \
    --dataset_path "${dataset_path}" \
    --output_dir "${output_dir}" 