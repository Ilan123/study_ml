from typing import List, Tuple, Callable
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

from utils.checkpoint import save_model_with_train_state
from utils.visualization import visualize_samples


class EnergyModelTrainer:
    """Trainer for energy-based models with contrastive divergence."""
    
    def __init__(
        self, 
        model: nn.Module,
        noise_scale: float,
        sampler,
        loss_fn: Callable[[nn.Module, Tensor, Tensor, Tensor, Tensor], Tuple[float, dict]],
        device: str,
        lr: float = 1e-4,
        patience: int = 10,
        visualize_every_n_epochs: int = 10,
        save_path: str = "../models/best_model.pth"
    ):
        self.model = model
        self.noise_scale = noise_scale
        self.sampler = sampler
        self.loss_fn = loss_fn
        self.device = device
        self.patience = patience
        self.visualize_every_n_epochs = visualize_every_n_epochs
        self.save_path = save_path
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)
        
        # Training state
        self.metrics = defaultdict(list)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        

    def prepare_batch_data(self, data_real, data_real_labels, data_fake):
        """Prepare and preprocess batch data."""
        data_real = data_real.to(self.device)
        data_real_labels = data_real_labels.to(self.device)
        data_real.data.add_(self.noise_scale * torch.randn_like(data_real)).clamp(-1, 1)
        
        # Match batch sizes
        if len(data_real) < len(data_fake):
            data_fake = data_fake[:len(data_real)]
            
        data_rand = torch.rand_like(data_real, device=self.device)
        return data_real, data_real_labels, data_fake, data_rand


    def process_batch(self, data_real, data_real_labels, training_mode: bool) -> Tuple[float, ...]:
        """Process a single batch and return loss components and energies."""
        # Generate negative samples
        with torch.enable_grad():
            data_fake = self.sampler.persist_sample(
                len(data_real)).to(self.device).detach()
        
        # Prepare batch data
        data_real, data_real_labels, data_fake, data_rand = self.prepare_batch_data(
            data_real, data_real_labels, data_fake)
        
        if training_mode:
            self.optimizer.zero_grad()
        
        # Compute loss
        loss, metrics = self.loss_fn(self.model, data_real, data_real_labels, data_fake, data_rand)
        
        if training_mode:
            self._backward_and_step(loss)
        
        return metrics
    

    def _backward_and_step(self, loss: Tensor):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
    

    def _accumulate_metrics(self, batch_metrics: dict, batch_results: Tuple[float, ...]):
        for metric, value in batch_results.items():
            batch_metrics[metric] += value
    

    def _finalize_epoch_metrics(self, batch_metrics: dict, n_batches: int) -> Tuple[float, ...]:
        """Convert accumulated metrics to averages."""
        for metric in batch_metrics.keys():
            batch_metrics[metric] /= n_batches
        return batch_metrics

    def run_epoch(self, data_loader: DataLoader, training_mode: bool = True) -> Tuple[float, ...]:
        batch_metrics = defaultdict(float)
        n_batches = 0
        
        for data_real, data_real_labels in tqdm(data_loader, total=len(data_loader)):
            n_batches += 1
            batch_results = self.process_batch(data_real, data_real_labels, training_mode)
            self._accumulate_metrics(batch_metrics, batch_results)
        
        return self._finalize_epoch_metrics(batch_metrics, n_batches)

    def save_best_model(self, epoch: int, train_loss: float, val_loss: float):
        """Save the best model checkpoint."""
        # You'll need to implement save_model_with_train_state or adapt this
        save_model_with_train_state(
            self.save_path, self.model, epoch, self.optimizer, self.scheduler,
            train_loss, val_loss, self.metrics['train_losses'], self.metrics['val_losses'],
        )

    def print_epoch_results(self, epoch: int, train_results: Tuple, val_results: Tuple, is_best: bool = False):
        """Print epoch results in a formatted way."""
        dict_to_str = lambda metric_value: f'{metric_value[0]}: {metric_value[1]:.4f}'
        train_str = ', '.join(map(dict_to_str, train_results.items()))
        val_str = ', '.join(map(dict_to_str, val_results.items()))

        base_msg = f'Epoch {epoch+1} completed. Train: {train_str}\nVal: {val_str}'
        
        if is_best:
            print(base_msg + '# NEW BEST MODEL #')
        else:
            print(base_msg + f'(Best: {self.best_val_loss:.5f} at epoch {self.best_epoch})')

    def should_stop_early(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def _run_training_epoch(self, train_dataloader: DataLoader) -> Tuple[float, ...]:
        self.model.train()
        return self.run_epoch(train_dataloader, training_mode=True)
    
    def _run_validation_epoch(self, val_dataloader: DataLoader) -> Tuple[float, ...]:
        self.model.eval()
        with torch.no_grad():
            return self.run_epoch(val_dataloader, training_mode=False)
        
    def _append_metrics(self, metrics: dict, prefix=''):
        for metric, value in metrics.items():
            self.metrics[f'{prefix}{metric}'].append(value)
    
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
        self._append_metrics(train_results, 'train_')
        self._append_metrics(val_results, 'val_')
        
        # Check for best model
        train_loss = train_results['loss']
        val_loss = val_results['loss']
        is_best = self._update_best_model(epoch, train_loss, val_loss)
        
        # Print results
        self.print_epoch_results(epoch, train_results, val_results, is_best)
        
        # Visualization every `self.visualize_every_n_epochs` epochs
        if (epoch + 1) % self.visualize_every_n_epochs == 0:
            samples = self.sampler.sample(n=16).detach().cpu()[:,0]
            title = f'Generated Samples - Epoch {epoch+1}'
            visualize_samples(samples, title=title)

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
        
        return self.metrics['train_loss'], self.metrics['val_loss']


def train_model(model, train_dataloader, val_dataloader, sampler, device, **kwargs):
    trainer = EnergyModelTrainer(
        model=model,
        sampler=sampler,
        device=device,
        **kwargs
    )
    return trainer.train(train_dataloader, val_dataloader)


def get_cd_loss(alpha: float):
    """Returns contrastive divergence and regularization loss function"""
    def cd_loss(
            model: nn.modules,
            data_real: Tensor,
            data_real_labels: Tensor,
            data_fake: Tensor,
            data_rand: Tensor
        ) -> Tuple[float, dict]:
        
        data = torch.cat([data_real, data_fake, data_rand], dim=0)
        e_real, e_fake, e_rand = model(data).chunk(3, dim=0)
        avg_e_real = torch.mean(e_real)
        avg_e_fake = torch.mean(e_fake)
        avg_e_rand = torch.mean(e_rand)
        
        loss_cd = avg_e_real - avg_e_fake
        loss_reg = alpha * torch.mean(e_real**2 + e_fake**2)
        loss = loss_cd + loss_reg

        metrics = {
            'loss': loss.item(),
            'loss_cd': loss_cd.item(),
            'loss_reg': loss_reg.item(),
            'avg_energy_real': avg_e_real.item(),
            'avg_energy_fake': avg_e_fake.item(),
            'avg_energy_rand': avg_e_rand.item(),
        }
        
        return loss, metrics
    
    return cd_loss


def get_multi_class_cd_loss(alpha: float):
    """Returns contrastive divergence and cross entropy loss function"""
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    def multi_class_cd_loss(
            model: nn.modules,
            data_real: Tensor,
            data_real_labels: Tensor,
            data_fake: Tensor,
            data_rand: Tensor
        ) -> Tuple[float, dict]:
        data = torch.cat([data_real, data_fake, data_rand], dim=0)
        logit_real, logit_fake, logit_rand = model(data).chunk(3, dim=0)
        e_real = -logit_real.exp().sum(axis=1).log()
        e_fake = -logit_fake.exp().sum(axis=1).log()
        e_rand = -logit_rand.exp().sum(axis=1).log()
        avg_e_real = torch.mean(e_real)
        avg_e_fake = torch.mean(e_fake)
        avg_e_rand = torch.mean(e_rand)
        
        loss_cd = avg_e_real - avg_e_fake
        loss_clf = cross_entropy_loss(logit_real, data_real_labels)
        if alpha != 0:
            loss_reg = alpha * torch.mean(e_real**2 + e_fake**2)
        else:
            loss_reg = torch.tensor(0)
        loss = loss_cd + loss_clf + loss_reg

        metrics = {
            'loss': loss.item(),
            'loss_cd': loss_cd.item(),
            'loss_clf': loss_clf.item(),
            'loss_reg': loss_reg.item(),
            'avg_energy_real': avg_e_real.item(),
            'avg_energy_fake': avg_e_fake.item(),
            'avg_energy_rand': avg_e_rand.item(),
        }
        
        return loss, metrics
    
    return multi_class_cd_loss