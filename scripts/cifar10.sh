#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
echo "Running experiment with CIFAR-10 dataset and ResNet18 model..."
python main.py \
    --dataset "cifar10" \
    --model "resnet18" \
    --epochs 10 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --optimizer "sgd" \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --scheduler \
    --analyze_every 10 \
    --run_name "cifar10_resnet18_example"

echo "CIFAR-10 experiment completed!"