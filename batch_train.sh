#!/bin/bash

vec_path="/data/ztj_data/rlphy/data/trainset/k_mer_6.pt"
log_dir="/data/ztj_data/rlphy/kmer6_logs"
model_dir="/data/ztj_data/rlphy/kmer6_models"
k_mer=6
embed_dims=(128 256 512)
n_layers=(2 3 4 5)
node_dim=4096
n_epoch=100
for dim in $embed_dims
do
    for n_layer in $n_layers
    do
        python run.py --vec_path $vec_path --embedding_dim $dim --n_encode_layers $n_layer \
        --log_dir $log_dir --output_dir $model_dir --n_epochs $n_epoch --node_dim $node_dim \
        --k_mer $k_mer
    done
done