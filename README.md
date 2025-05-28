# Loss Distribution Analyzer

This project analyzes the distribution of loss values during the training of a machine learning model. It helps understand how the loss distribution evolves throughout the training process.

## Features

- Tracks loss distribution at specified training steps
- Generates histograms and density plots for each analyzed step
- Calculates statistical metrics (mean, median, standard deviation, etc.)
- Creates a final report with the evolution of loss statistics
- Supports comparison of distributions across different training steps
- Organizes results by experiment runs with detailed configuration

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SciPy

## Usage

### Basic Usage

```bash
python main.py
```

### Command Line Arguments

- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate for optimizer (default: 0.01)
- `--hidden_size`: Size of hidden layer in the model (default: 64)
- `--data_size`: Size of synthetic dataset (default: 1000)
- `--input_dim`: Input dimension (default: 10)
- `--output_dim`: Output dimension (default: 1)
- `--save_dir`: Base directory to save results (default: 'results')
- `--analyze_every`: Analyze loss distribution every N steps (default: 10)
- `--dataset`: Dataset name to use (default: 'synthetic')
- `--model`: Model type to use (default: 'simple')
- `--run_name`: Custom name for this experiment run (optional)

Example:

```bash
python main.py --epochs 20 --batch_size 64 --learning_rate 0.001 --analyze_every 5 --model mlp
```

## Output

The program generates the following outputs in a directory named after your experiment run (e.g., `results/synthetic_simple_e10_b32_lr0.01_h64_20230415-123456/`):

1. Configuration files (`config.json` and `config.txt`) with all experiment parameters
2. Loss distribution data files (`.npy`) for each analyzed step
3. Histogram plots (`.png`) for each analyzed step
4. A CSV file with loss statistics for all analyzed steps
5. Plots showing the evolution of loss statistics throughout training
6. Comparison plots between epochs

## Project Structure

- `main.py`: Main script for training and analysis
- `model.py`: Neural network model definitions
- `loss_analyzer.py`: Class for analyzing loss distributions
- `utils.py`: Utility functions for saving and plotting

## Extending the Project

You can extend this project by:

1. Adding more complex models in `model.py`
2. Implementing additional statistical analyses in `loss_analyzer.py`
3. Creating more visualization types in `utils.py`
4. Adding support for real datasets