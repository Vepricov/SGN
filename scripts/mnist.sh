#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
echo "Running experiment with MNIST dataset and CNN model..."
python main.py \
    --dataset "mnist" \
    --model "cnn" \
    --epochs 3 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --optimizer "sgd" \
    --analyze_every 10 \

echo "MNIST experiment completed!"