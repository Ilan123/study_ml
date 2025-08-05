import torch
from torch import Tensor
from matplotlib import pyplot as plt

def visualize_samples(samples: Tensor, nrows=4, ncols=4, figsize=(8, 8), title=''):
    """Generate and display sample images from the trained model."""
    # Sample
    samples = torch.clamp(samples, 0, 1)
    
    # Plot samples
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_losses(train_losses: list, val_losses: list):
    """
    Plots training and validation loss curves.

    Args:
        train_losses (List[float]): Training loss values per epoch.
        val_losses (List[float]): Validation loss values per epoch.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()