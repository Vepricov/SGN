#!/bin/bash

# MNIST with CNN model (classification)
echo "Running experiment with MNIST dataset and CNN model..."
python main.py \
    --dataset "mnist" \
    --model "cnn" \
    --epochs 1 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --optimizer "sgd" \
    --analyze_every 10 \
    --run_name "mnist_cnn_example"

echo "MNIST experiment completed!"