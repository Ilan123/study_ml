import torch
from torch import nn, Tensor
from typing import Callable
from matplotlib import pyplot as plt

def save_model_with_train_state(
    checkpoint_path: str,
    model: nn.Module,
    epoch: int,
    optimizer,
    scheduler,
    train_avg_loss: float,
    val_avg_loss: float,
    train_losses: list,
    val_losses: list,
):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_avg_loss,
        'val_loss': val_avg_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, checkpoint_path)


def load_best_model(
        model: nn.Module,
        checkpoint_path: str,
        device: str
) -> nn.Module:
    """Load the best saved model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
    
    return model
    

def visualize_samples(samples: Tensor, epoch=None):
    """Generate and display sample images from the trained model."""
    # Sample
    samples = torch.clamp(samples, 0, 1)
    
    # Plot samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    
    title = f'Generated Samples - Epoch {epoch}' if epoch else 'Generated Samples'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()