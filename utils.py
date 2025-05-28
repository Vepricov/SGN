import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def save_loss_distribution(losses, global_step, epoch, step_in_epoch, save_dir):
    """
    Save loss distribution data to a file.
    
    Args:
        losses: Array of loss values
        global_step: Unique identifier for the step across all epochs
        epoch: Current epoch number
        step_in_epoch: Step number within the current epoch
        save_dir: Directory to save the data
    """
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'loss_dist_e{epoch}_s{step_in_epoch}_g{global_step}.npy'), losses)

def plot_loss_distribution(losses, global_step, epoch, step_in_epoch, save_dir):
    """
    Plot and save the loss distribution histogram.
    
    Args:
        losses: Array of loss values
        global_step: Unique identifier for the step across all epochs
        epoch: Current epoch number
        step_in_epoch: Step number within the current epoch
        save_dir: Directory to save the plot
    """
    os.makedirs(os.path.join(save_dir, 'histograms'), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = plt.hist(losses, bins=30, alpha=0.7, color='skyblue', density=True)
    
    # Add a kernel density estimate
    kde = stats.gaussian_kde(losses)
    x = np.linspace(min(losses), max(losses), 1000)
    plt.plot(x, kde(x), 'r-', linewidth=2)
    
    # Add vertical lines for mean and median
    plt.axvline(np.mean(losses), color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(losses):.4f}')
    plt.axvline(np.median(losses), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(losses):.4f}')
    
    # Add text with statistics
    stats_text = f"Mean: {np.mean(losses):.4f}\n" \
                f"Median: {np.median(losses):.4f}\n" \
                f"Std Dev: {np.std(losses):.4f}\n" \
                f"Min: {np.min(losses):.4f}\n" \
                f"Max: {np.max(losses):.4f}\n" \
                f"Skewness: {stats.skew(losses):.4f}\n" \
                f"Kurtosis: {stats.kurtosis(losses):.4f}"
    
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f'Loss Distribution - Epoch {epoch+1}, Step {step_in_epoch}')
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    #plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'histograms', f'loss_dist_e{epoch}_s{step_in_epoch}_g{global_step}.png'))
    plt.close()

def compare_distributions(step_info_list, save_dir):
    """
    Compare loss distributions from different steps.
    
    Args:
        step_info_list: List of tuples (global_step, epoch, step_in_epoch) to compare
        save_dir: Directory where loss distributions are saved
    """
    plt.figure(figsize=(12, 8))
    
    for global_step, epoch, step_in_epoch in step_info_list:
        try:
            losses = np.load(os.path.join(save_dir, f'loss_dist_e{epoch}_s{step_in_epoch}_g{global_step}.npy'))
            kde = stats.gaussian_kde(losses)
            x = np.linspace(min(losses), max(losses), 1000)
            plt.plot(x, kde(x), linewidth=2, label=f'Epoch {epoch+1}, Step {step_in_epoch}')
        except FileNotFoundError:
            print(f"Data for epoch {epoch+1}, step {step_in_epoch} not found.")
    
    plt.title('Comparison of Loss Distributions')
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'loss_distribution_comparison.png'))
    plt.close()

def compare_epochs(epochs, save_dir):
    """
    Compare average loss distributions across different epochs.
    
    Args:
        epochs: List of epoch numbers to compare
        save_dir: Directory where loss distributions are saved
    """
    plt.figure(figsize=(12, 8))
    
    # Get all distribution files
    all_files = os.listdir(save_dir)
    dist_files = [f for f in all_files if f.startswith('loss_dist_e') and f.endswith('.npy')]
    
    for epoch in epochs:
        # Get all files for this epoch
        epoch_files = [f for f in dist_files if f.startswith(f'loss_dist_e{epoch}_')]
        
        if not epoch_files:
            print(f"No data found for epoch {epoch+1}")
            continue
        
        # Combine all distributions for this epoch
        combined_losses = []
        for file in epoch_files:
            losses = np.load(os.path.join(save_dir, file))
            combined_losses.extend(losses)
        
        if combined_losses:
            kde = stats.gaussian_kde(combined_losses)
            x = np.linspace(min(combined_losses), max(combined_losses), 1000)
            plt.plot(x, kde(x), linewidth=2, label=f'Epoch {epoch+1}')
    
    plt.title('Comparison of Loss Distributions Across Epochs')
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'epoch_loss_distribution_comparison.png'))
    plt.close()