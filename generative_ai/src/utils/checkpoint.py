import torch
from torch import nn

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


def load_model(
        model: nn.Module,
        checkpoint_path: str,
        device: str
) -> nn.Module:
    """Load the best saved model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
    
    return model
    