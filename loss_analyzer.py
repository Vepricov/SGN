import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from utils import save_loss_distribution, plot_loss_distribution

class LossAnalyzer:
    """
    Class for analyzing the distribution of loss values during model training.
    """
    def __init__(self, dataloader, save_dir='results', analysis_batch_size=100):
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.step_distributions = {}
        self.analysis_batch_size = analysis_batch_size
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def compute_all_losses(self, model, criterion, device):
        """
        Compute loss for all samples in the dataset.
        
        Args:
            model: The neural network model
            criterion: Loss function
            device: Device to run computations on
            
        Returns:
            numpy array of loss values for each sample
        """
        model.eval()  # Set model to evaluation mode
        all_losses = []
        
        with torch.no_grad():  # No need to track gradients
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Compute individual losses for each sample in the batch
                for i in range(len(inputs)):
                    loss = criterion(outputs[i:i+1], targets[i:i+1])
                    all_losses.append(loss.item())
                
                if len(all_losses) > self.analysis_batch_size:
                    break
        
        return np.array(all_losses)
    
    def analyze_step(self, model, criterion, device, global_step, epoch, step_in_epoch):
        """
        Analyze the loss distribution at a specific training step.
        
        Args:
            model: The neural network model
            criterion: Loss function
            device: Device to run computations on
            global_step: Unique identifier for the step across all epochs
            epoch: Current epoch number
            step_in_epoch: Step number within the current epoch
        """
        # Compute losses for all samples
        losses = self.compute_all_losses(model, criterion, device)
        
        # Store the distribution
        self.step_distributions[global_step] = {
            'losses': losses,
            'mean': np.mean(losses),
            'median': np.median(losses),
            'std': np.std(losses),
            'min': np.min(losses),
            'max': np.max(losses),
            'skewness': stats.skew(losses),
            'kurtosis': stats.kurtosis(losses),
            'epoch': epoch,
            'step_in_epoch': step_in_epoch
        }
        
        # Save the distribution
        # save_loss_distribution(losses, global_step, epoch, step_in_epoch, self.save_dir)
        
        # Plot the distribution
        plot_loss_distribution(losses, global_step, epoch, step_in_epoch, self.save_dir)
        
        print(f"Epoch {epoch+1}, Step {step_in_epoch}: Mean Loss = {np.mean(losses):.4f}, Std = {np.std(losses):.4f}")
    
    def generate_final_report(self):
        """
        Generate a final report with statistics and visualizations of loss distributions.
        """
        if not self.step_distributions:
            print("No data to generate report. Did you run analyze_step during training?")
            return
        
        # Create a DataFrame with statistics for each step
        stats_df = pd.DataFrame({
            'global_step': [],
            'epoch': [],
            'step_in_epoch': [],
            'mean': [],
            'median': [],
            'std': [],
            'min': [],
            'max': [],
            'skewness': [],
            'kurtosis': []
        })
        
        for step, dist in self.step_distributions.items():
            stats_df = pd.concat([stats_df, pd.DataFrame({
                'global_step': [step],
                'epoch': [dist['epoch']],
                'step_in_epoch': [dist['step_in_epoch']],
                'mean': [dist['mean']],
                'median': [dist['median']],
                'std': [dist['std']],
                'min': [dist['min']],
                'max': [dist['max']],
                'skewness': [dist['skewness']],
                'kurtosis': [dist['kurtosis']]
            })], ignore_index=True)
        
        # Sort by global_step to ensure correct order
        stats_df = stats_df.sort_values('global_step')
        
        # Save statistics to CSV
        stats_df.to_csv(os.path.join(self.save_dir, 'loss_statistics.csv'), index=False)
        
        # Plot evolution of statistics over steps
        plt.figure(figsize=(15, 10))
        
        # Combined plot for mean, median, min, max
        plt.subplot(2, 1, 1)
        plt.plot(stats_df['global_step'], stats_df['mean'], 'b-', label='Mean')
        plt.plot(stats_df['global_step'], stats_df['median'], 'r--', label='Median')
        #plt.plot(stats_df['global_step'], stats_df['min'], 'g-.', label='Min')
        #plt.plot(stats_df['global_step'], stats_df['max'], 'm:', label='Max')
        plt.xlabel('Global Step')
        plt.ylabel('Loss Value')
        plt.title('Evolution of Loss Statistics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add epoch markers
        unique_epochs = stats_df['epoch'].unique()
        for epoch in unique_epochs:
            first_step = stats_df[stats_df['epoch'] == epoch]['global_step'].min()
            plt.axvline(x=first_step, color='k', linestyle='--', alpha=0.3)
            plt.text(first_step, plt.ylim()[1] * 0.9, f'Epoch {epoch+1}', 
                     rotation=90, verticalalignment='top')
        
        # Standard deviation
        plt.subplot(2, 2, 3)
        plt.plot(stats_df['global_step'], stats_df['std'], 'g-')
        plt.xlabel('Global Step')
        plt.ylabel('Standard Deviation')
        plt.title('Evolution of Loss Standard Deviation')
        plt.grid(True, alpha=0.3)
        
        # Skewness and kurtosis
        plt.subplot(2, 2, 4)
        plt.plot(stats_df['global_step'], stats_df['skewness'], 'y-', label='Skewness')
        plt.plot(stats_df['global_step'], stats_df['kurtosis'], 'k--', label='Kurtosis')
        plt.xlabel('Global Step')
        plt.ylabel('Value')
        plt.title('Evolution of Loss Distribution Shape')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_evolution.png'))
        plt.close()
        
        # Create a plot showing loss statistics by epoch
        epoch_stats = stats_df.groupby('epoch').agg({
            'mean': 'mean',
            'median': 'mean',
            'std': 'mean',
            'min': 'min',
            'max': 'max'
        }).reset_index()
        
        print(f"Final report generated and saved to {self.save_dir}")