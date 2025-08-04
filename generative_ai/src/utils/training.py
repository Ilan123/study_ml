import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable, Tuple
from tqdm import tqdm
from .checkpoint import save_model_with_train_state, load_model
from .visualization import visualize_samples

def train_generative_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: Callable[[nn.Module, Tensor], Tensor],
    device: str,
    num_epochs: int = 50,
    lr: float = 1e-4,
    patience: int = 10,
    save_path: str = '../models/best_model.pth'
) -> Tuple[list, list]:
    """
        General training loop for generative models.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            loss_fn: Loss function that takes (model, data) and returns a scalar loss
            device: Device to train on ('cuda' or 'cpu')
            num_epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            save_path: Path to save best model
            
        Returns:
            Tuple of (train_losses, val_losses) - loss histories for each epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_avg_loss = run_epoch(
            model, train_dataloader, loss_fn, device, optimizer, True)

        
        model.eval()
        with torch.no_grad():
            val_avg_loss = run_epoch(
                model, val_dataloader, loss_fn, device, optimizer, False)
        
        scheduler.step()
        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)


        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            save_model_with_train_state(
                save_path, model, epoch, optimizer, scheduler,
                train_avg_loss, val_avg_loss, train_losses, val_losses,
            )
            
            print(f'Epoch {epoch+1} completed. Train Loss: {train_avg_loss:.4f}, '
                  f'Val Loss: {val_avg_loss:.4f} # NEW BEST MODEL #')
        else:
            epochs_without_improvement += 1
            print(f'Epoch {epoch+1} completed. Train Loss: {train_avg_loss:.4f}, '
                  f'Val Loss: {val_avg_loss:.4f} (Best: {best_val_loss:.4f} at epoch {best_epoch})')
        

        if (epoch + 1) % 10 == 0:
            samples = model.sample(16, device).cpu()
            visualize_samples(samples, epoch+1)

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f'\nEarly stopping triggered after {patience} epochs without improvement.')
            print(f'Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}')
            break

    return train_losses, val_losses



def run_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: Callable[[nn.Module, Tensor], Tensor],
        device: str,
        optimizer: Optimizer,
        training_mode: bool = True,
    ) -> float:
    """
    Run a single epoch in either training or evaluation mode.
    
    Args:
        model: The model to train/evaluate
        dataloader: DataLoader for the dataset
        device: Device to run on (cuda/cpu)
        optimizer: Optimizer for training (required if train_mode=True)
        train_mode: If True, run in training mode; if False, run in evaluation mode
        max_grad_norm: Maximum gradient norm for clipping (only used in training)
    
    Returns:
        average_loss: Average loss for the epoch
    """
    total_loss, num_batches = 0, 0
    
    for data, _ in tqdm(data_loader, total=len(data_loader)):
        data = data.to(device)
        if training_mode:
            optimizer.zero_grad()
        
        # Compute negative log likelihood
        loss = loss_fn(model, data)
        
        if training_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss
    

def flow_nll_loss(model: nn.Module, data: Tensor) -> Tensor:
    """Negative log-likelihood loss for normalizing flows."""
    log_prob = model.log_prob(data)
    return -log_prob.mean()


def train_flow(model, train_dataloader, val_dataloader, device, **kwargs):
    """Convenience wrapper for training flows."""
    return train_generative_model(
        model, train_dataloader, val_dataloader, 
        flow_nll_loss, device, **kwargs
    )