import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import json
from model import SimpleModel, MLPModel, CNNModel, CIFAR10CNN, ResNet18
from loss_analyzer import LossAnalyzer
from utils import save_loss_distribution, plot_loss_distribution

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model and analyze loss distribution')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--max_steps', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--data_size', type=int, default=1000, help='Size of synthetic dataset')
    parser.add_argument('--input_dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--save_dir', type=str, default='results', help='Base directory to save results')
    parser.add_argument('--analyze_every', type=int, default=10, help='Analyze loss every N steps')
    parser.add_argument('--dataset', type=str, default='synthetic', 
                        choices=['synthetic', 'mnist', 'fashion_mnist', 'cifar10'],
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='simple', 
                        choices=['simple', 'mlp', 'cnn', 'cifar_cnn', 'resnet18'],
                        help='Model type')
    parser.add_argument('--run_name', type=str, default='', help='Custom name for this run (optional)')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        choices=['sgd', 'adam', 'rmsprop'],
                        help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler') 
    parser.add_argument('--analysis_batch_size', type=int, default=100, help='Batch size to compute loss')
    return parser.parse_args()

def generate_synthetic_data(data_size, input_dim, output_dim):
    # Generate synthetic data for demonstration
    X = torch.randn(data_size, input_dim)
    # Create a synthetic relationship
    W = torch.randn(input_dim, output_dim)
    b = torch.randn(output_dim)
    y = torch.matmul(X, W) + b + 0.1 * torch.randn(data_size, output_dim)  # Add some noise
    return X, y

def load_mnist(batch_size):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_fashion_mnist(batch_size):
    """Load Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_cifar10(batch_size):
    """Load CIFAR-10 dataset"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, dataloader, criterion, optimizer, device, loss_analyzer, analyze_every, epoch):
    model.train()
    running_loss = 0.0
    step_counter = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update step counter
        step_counter += 1
        
        # Analyze loss distribution if needed
        if step_counter % analyze_every == 0:
            # Create a unique identifier that includes both epoch and step
            global_step = (epoch * len(dataloader)) + step_counter
            loss_analyzer.analyze_step(model, criterion, device, global_step, epoch, step_counter)
        
        running_loss += loss.item()
        
        # Calculate accuracy for classification tasks
        if not isinstance(criterion, nn.MSELoss):
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total if total > 0 else 0
    return running_loss / len(dataloader), accuracy

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Handle different output formats
            if isinstance(criterion, nn.MSELoss):
                loss = criterion(outputs, targets)
            else:  # Classification task
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            
            total_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def save_run_config(args, save_dir):
    """
    Save the run configuration to a JSON file.
    
    Args:
        args: Command line arguments
        save_dir: Directory to save the configuration
    """
    config = vars(args)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Also save as a readable text file
    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def main():
    args = parse_args()
    
    # Create a unique run name if not provided
    if not args.run_name:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{args.dataset}_{args.model}_e{args.epochs}_b{args.batch_size}_lr{args.learning_rate}_time{timestamp}"
    
    # Create full save directory path
    full_save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(full_save_dir, exist_ok=True)
    
    # Save run configuration
    save_run_config(args, full_save_dir)
    
    # Display run information
    print("\nRunning experiment with the following parameters:")
    print(f"  Dataset:               {args.dataset}")
    print(f"  Model:                 {args.model}")
    print(f"  Optimizer:             {args.optimizer}")
    print(f"  Epochs:                {args.epochs}")
    print(f"  Batch size:            {args.batch_size}")
    print(f"  Learning rate:         {args.learning_rate}")
    print(f"  Weight decay:          {args.weight_decay}")
    print(f"  Momentum:              {args.momentum}")
    print(f"  Scheduler:             {args.scheduler}")
    print(f"  Save directory:        {full_save_dir}")
    print(f"  Analyze every:         {args.analyze_every} steps")
    print(f"  Analysis batch size:   {args.analysis_batch_size}")
    print(f"  Run name:              {args.run_name}")
    print()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("~~~~~~~~~~~ GPU ~~~~~~~~~~~")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))
    else:
        print("~~~~~~~~~~~ CPU ~~~~~~~~~~~")
    
    # Load dataset
    if args.dataset == 'synthetic':
        X, y = generate_synthetic_data(args.data_size, args.input_dim, args.output_dim)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = None  # No test set for synthetic data
        input_dim = args.input_dim
        output_dim = args.output_dim
        is_image_data = False
        criterion = nn.MSELoss()
    elif args.dataset == 'mnist':
        train_loader, test_loader = load_mnist(args.batch_size)
        input_dim = 28 * 28  # MNIST image size
        output_dim = 10      # 10 classes
        is_image_data = True
        criterion = nn.CrossEntropyLoss()
    elif args.dataset == 'fashion_mnist':
        train_loader, test_loader = load_fashion_mnist(args.batch_size)
        input_dim = 28 * 28  # Fashion MNIST image size
        output_dim = 10      # 10 classes
        is_image_data = True
        criterion = nn.CrossEntropyLoss()
    elif args.dataset == 'cifar10':
        train_loader, test_loader = load_cifar10(args.batch_size)
        input_dim = 3 * 32 * 32  # CIFAR-10 image size (3 channels, 32x32 pixels)
        output_dim = 10          # 10 classes
        is_image_data = True
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Dataset '{args.dataset}' not implemented.")
    
    # Initialize model based on model type
    if args.model == 'simple':
        if is_image_data and args.dataset != 'synthetic':
            print("Warning: Simple model may not be suitable for image data.")
        model = SimpleModel(input_dim, args.hidden_size, output_dim).to(device)
    elif args.model == 'mlp':
        if is_image_data:
            # Flatten images for MLP
            model = MLPModel(input_dim, [args.hidden_size, args.hidden_size//2], output_dim).to(device)
        else:
            model = MLPModel(input_dim, [args.hidden_size, args.hidden_size//2], output_dim).to(device)
    elif args.model == 'cnn':
        if not is_image_data:
            raise ValueError("CNN model requires image data.")
        if args.dataset in ['mnist', 'fashion_mnist']:
            model = CNNModel(num_classes=output_dim).to(device)
        else:
            raise ValueError(f"CNN model not configured for dataset '{args.dataset}'.")
    elif args.model == 'cifar_cnn':
        if not is_image_data or args.dataset != 'cifar10':
            raise ValueError("CIFAR CNN model requires CIFAR-10 dataset.")
        model = CIFAR10CNN(num_classes=output_dim).to(device)
    elif args.model == 'resnet18':
        if not is_image_data or args.dataset != 'cifar10':
            raise ValueError("ResNet18 model requires CIFAR-10 dataset.")
        model = ResNet18(num_classes=output_dim).to(device)
    else:
        raise ValueError(f"Model '{args.model}' not implemented.")
    
    # Define optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                             momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer '{args.optimizer}' not implemented.")
    
    # Learning rate scheduler
    scheduler = None
    if args.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Initialize loss analyzer
    loss_analyzer = LossAnalyzer(train_loader, full_save_dir, args.analysis_batch_size)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Results will be saved to: {full_save_dir}")
    
    # Create a DataFrame to store training metrics
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, 
                                      loss_analyzer, args.analyze_every, epoch)
        
        # Evaluate on test set if available
        if test_loader is not None:
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
            
            # Update learning rate scheduler if used
            if scheduler is not None:
                scheduler.step(test_loss)
        else:
            test_loss, test_acc = float('nan'), float('nan')
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")
        
        # Store metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(full_save_dir, 'model.pth'))
    
    # Save training metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(full_save_dir, 'training_metrics.csv'), index=False)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Train Loss')
    if test_loader is not None:
        plt.plot(metrics['epoch'], metrics['test_loss'], 'r--', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot (only for classification tasks)
    if not isinstance(criterion, nn.MSELoss):
        plt.subplot(1, 2, 2)
        plt.plot(metrics['epoch'], metrics['train_acc'], 'b-', label='Train Accuracy')
        if test_loader is not None:
            plt.plot(metrics['epoch'], metrics['test_acc'], 'r--', label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(full_save_dir, 'training_curves.png'))
    plt.close()
    
    # Final analysis and visualization
    loss_analyzer.generate_final_report()
    print(f"Training completed. Results saved to {full_save_dir}")

if __name__ == "__main__":
    main()