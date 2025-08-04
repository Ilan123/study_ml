from dataclasses import dataclass, field
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

from utils.checkpoint import save_model_with_train_state
from utils.visualization import visualize_samples


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_losses: List[float] = field(default_factory=list)
    train_losses_cd: List[float] = field(default_factory=list)
    train_losses_reg: List[float] = field(default_factory=list)
    train_energies_real: List[float] = field(default_factory=list)
    train_energies_fake: List[float] = field(default_factory=list)
    train_energies_rand: List[float] = field(default_factory=list)
    
    val_losses: List[float] = field(default_factory=list)
    val_losses_cd: List[float] = field(default_factory=list)
    val_losses_reg: List[float] = field(default_factory=list)
    val_energies_real: List[float] = field(default_factory=list)
    val_energies_fake: List[float] = field(default_factory=list)
    val_energies_rand: List[float] = field(default_factory=list)
    
    def add_train_metrics(self, loss, loss_cd, loss_reg, e_real, e_fake, e_rand):
        """Add training metrics for current epoch."""
        self.train_losses.append(loss)
        self.train_losses_cd.append(loss_cd)
        self.train_losses_reg.append(loss_reg)
        self.train_energies_real.append(e_real)
        self.train_energies_fake.append(e_fake)
        self.train_energies_rand.append(e_rand)
    
    def add_val_metrics(self, loss, loss_cd, loss_reg, e_real, e_fake, e_rand):
        """Add validation metrics for current epoch."""
        self.val_losses.append(loss)
        self.val_losses_cd.append(loss_cd)
        self.val_losses_reg.append(loss_reg)
        self.val_energies_real.append(e_real)
        self.val_energies_fake.append(e_fake)
        self.val_energies_rand.append(e_rand)


class EnergyModelTrainer:
    """Trainer for energy-based models with contrastive divergence."""
    
    def __init__(
        self, 
        model: nn.Module,
        alpha: float,
        noise_scale: float,
        sampler,
        device: str,
        lr: float = 1e-4,
        patience: int = 10,
        save_path: str = "../models/best_model.pth"
    ):
        self.model = model
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.sampler = sampler
        self.device = device
        self.patience = patience
        self.save_path = save_path
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)
        
        # Training state
        self.metrics = TrainingMetrics()
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        

    def prepare_batch_data(self, data_real, data_fake):
        """Prepare and preprocess batch data."""
        data_real = data_real.to(self.device)
        data_real.data.add_(self.noise_scale * torch.randn_like(data_real)).clamp(-1, 1)
        
        # Match batch sizes
        if len(data_real) < len(data_fake):
            data_fake = data_fake[:len(data_real)]
            
        data_rand = torch.rand_like(data_real, device=self.device)
        return data_real, data_fake, data_rand


    def compute_loss(self, data_real, data_fake, data_rand):
        """Compute contrastive divergence and regularization losses."""
        data = torch.cat([data_real, data_fake, data_rand], dim=0)
        e_real, e_fake, e_rand = self.model(data).chunk(3, dim=0)
        avg_e_real = torch.mean(e_real)
        avg_e_fake = torch.mean(e_fake)
        avg_e_rand = torch.mean(e_rand)
        
        loss_cd = avg_e_real - avg_e_fake
        loss_reg = self.alpha * torch.mean(e_real**2 + e_fake**2)
        loss = loss_cd + loss_reg
        
        return loss, loss_cd, loss_reg, avg_e_real, avg_e_fake, avg_e_rand


    def process_batch(self, data_real, training_mode: bool) -> Tuple[float, ...]:
        """Process a single batch and return loss components and energies."""
        # Generate negative samples
        with torch.enable_grad():
            data_fake = self.sampler.persist_sample(
                len(data_real)).to(self.device).detach()
        
        # Prepare batch data
        data_real, data_fake, data_rand = self.prepare_batch_data(data_real, data_fake)
        
        if training_mode:
            self.optimizer.zero_grad()
        
        # Compute loss
        loss, loss_cd, loss_reg, e_real, e_fake, e_rand = self.compute_loss(
            data_real, data_fake, data_rand
        )
        
        if training_mode:
            self._backward_and_step(loss)
        
        return (
            loss.item(), loss_cd.item(), loss_reg.item(),
            e_real.item(), e_fake.item(), e_rand.item()
        )
    

    def _backward_and_step(self, loss):
        """Perform backward pass and optimizer step."""
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
    

    def _accumulate_metrics(self, batch_metrics: dict, batch_results: Tuple[float, ...]):
        """Accumulate batch results into running metrics."""
        loss, loss_cd, loss_reg, e_real, e_fake, e_rand = batch_results
        batch_metrics['total_loss'] += loss
        batch_metrics['total_loss_cd'] += loss_cd
        batch_metrics['total_loss_reg'] += loss_reg
        batch_metrics['total_energy_real'] += e_real
        batch_metrics['total_energy_fake'] += e_fake
        batch_metrics['total_energy_rand'] += e_rand
        batch_metrics['num_batches'] += 1
    

    def _finalize_epoch_metrics(self, batch_metrics: dict) -> Tuple[float, ...]:
        """Convert accumulated metrics to averages."""
        num_batches = batch_metrics['num_batches']
        return (
            batch_metrics['total_loss'] / num_batches,
            batch_metrics['total_loss_cd'] / num_batches,
            batch_metrics['total_loss_reg'] / num_batches,
            batch_metrics['total_energy_real'] / num_batches,
            batch_metrics['total_energy_fake'] / num_batches,
            batch_metrics['total_energy_rand'] / num_batches,
        )

    def run_epoch(self, data_loader: DataLoader, training_mode: bool = True) -> Tuple[float, ...]:
        """Run a single epoch in either training or evaluation mode."""
        batch_metrics = {
            'total_loss': 0, 'total_loss_cd': 0, 'total_loss_reg': 0,
            'total_energy_real': 0, 'total_energy_fake': 0, 'total_energy_rand': 0,
            'num_batches': 0
        }
        
        for data_real, _ in tqdm(data_loader, total=len(data_loader)):
            batch_results = self.process_batch(data_real, training_mode)
            self._accumulate_metrics(batch_metrics, batch_results)
        
        return self._finalize_epoch_metrics(batch_metrics)

    def save_best_model(self, epoch: int, train_loss: float, val_loss: float):
        """Save the best model checkpoint."""
        # You'll need to implement save_model_with_train_state or adapt this
        save_model_with_train_state(
            self.save_path, self.model, epoch, self.optimizer, self.scheduler,
            train_loss, val_loss, self.metrics.train_losses, self.metrics.val_losses,
        )

    def print_epoch_results(self, epoch: int, train_results: Tuple, val_results: Tuple, is_best: bool = False):
        """Print epoch results in a formatted way."""
        train_loss, train_cd, train_reg, train_e_real, train_e_fake, train_e_rand = train_results
        val_loss, val_cd, val_reg, val_e_real, val_e_fake, val_e_rand = val_results
        
        base_msg = (
            f'Epoch {epoch+1} completed. Train: Loss: {train_loss:.5f}, '
            f'CD Loss: {train_cd:.5f}, Reg Loss: {train_reg:.5f}, '
            f'E_real: {train_e_real:.4f}, E_fake: {train_e_fake:.4f},\n'
            f'Val: Loss: {val_loss:.5f}, '
            f'CD Loss: {val_cd:.5f}, Reg Loss: {val_reg:.5f}, '
            f'E_real: {val_e_real:.4f}, E_fake: {val_e_fake:.4f}, '
            f'E_rand: {val_e_rand:.4f}, '
        )
        
        if is_best:
            print(base_msg + '# NEW BEST MODEL #')
        else:
            print(base_msg + f'(Best: {self.best_val_loss:.5f} at epoch {self.best_epoch})')

    def should_stop_early(self) -> bool:
        """Check if early stopping criteria is met."""
        return self.epochs_without_improvement >= self.patience

    def _run_training_epoch(self, train_dataloader: DataLoader) -> Tuple[float, ...]:
        """Run a single training epoch."""
        self.model.train()
        return self.run_epoch(train_dataloader, training_mode=True)
    
    def _run_validation_epoch(self, val_dataloader: DataLoader) -> Tuple[float, ...]:
        """Run a single validation epoch."""
        self.model.eval()
        with torch.no_grad():
            return self.run_epoch(val_dataloader, training_mode=False)
    
    def _update_best_model(self, epoch: int, train_loss: float, val_loss: float) -> bool:
        """Update best model tracking. Returns True if this is a new best."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.epochs_without_improvement = 0
            self.save_best_model(epoch, train_loss, val_loss)
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def _handle_epoch_end(self, epoch: int, train_results: Tuple, val_results: Tuple):
        """Handle end-of-epoch tasks: metrics, logging, visualization."""
        # Update metrics
        self.metrics.add_train_metrics(*train_results)
        self.metrics.add_val_metrics(*val_results)
        
        # Check for best model
        train_loss = train_results[0]
        val_loss = val_results[0]
        is_best = self._update_best_model(epoch, train_loss, val_loss)
        
        # Print results
        self.print_epoch_results(epoch, train_results, val_results, is_best)
        
        # Visualization every 10 epochs
        if (epoch + 1) % 10 == 0:
            samples = self.sampler.sample(n=16).detach().cpu()[:,0]
            visualize_samples(samples, epoch + 1)

    def train(
        self, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        num_epochs: int = 50
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Maximum number of epochs
            
        Returns:
            Training and validation loss histories
        """
        for epoch in range(num_epochs):
            # Run training and validation
            train_results = self._run_training_epoch(train_dataloader)
            val_results = self._run_validation_epoch(val_dataloader)
            
            self.scheduler.step()
            
            # Handle all end-of-epoch tasks
            self._handle_epoch_end(epoch, train_results, val_results)
            
            # Early stopping check
            if self.should_stop_early():
                print(f'\nEarly stopping triggered after {self.patience} epochs without improvement.')
                print(f'Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}')
                break
        
        return self.metrics.train_losses, self.metrics.val_losses


def train_model(model, train_dataloader, val_dataloader, sampler, device, **kwargs):
    trainer = EnergyModelTrainer(
        model=model,
        sampler=sampler,
        device=device,
        **kwargs
    )
    return trainer.train(train_dataloader, val_dataloader)
