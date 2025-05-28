#!/bin/bash

# Synthetic data with simple model (regression)
echo "Running experiment with synthetic data and simple model..."
python main.py \
    --dataset "synthetic" \
    --model "simple" \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --hidden_size 64 \
    --data_size 1000 \
    --input_dim 10 \
    --output_dim 1 \
    --analyze_every 1 \
    --run_name "synthetic_simple_example"

echo "Synthetic data experiment completed!"