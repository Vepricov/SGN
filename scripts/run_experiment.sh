#!/bin/bash

# Run all experiments sequentially
echo "Starting all experiments..."
echo "=================================="

# Run synthetic data experiment
chmod +x run_synthetic.sh
./run_synthetic.sh

echo "=================================="

# Run MNIST experiment
chmod +x run_mnist.sh
./run_mnist.sh

echo "=================================="

# Run CIFAR-10 experiment
chmod +x run_cifar10.sh
./run_cifar10.sh

echo "=================================="
echo "All experiments completed!"