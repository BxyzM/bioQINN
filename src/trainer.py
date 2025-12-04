"""
Quantum Circuit Trainer for classification tasks.

Framework for training the quantum classifier or regressor with integrated W&B support.

Author: Dr. Aritra Bal (ETP)
Date: December 01, 2025
"""

import numpy as np
import time
import os
import json
import pathlib
from typing import Callable, Optional, Any, Dict, List, Tuple, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class QuantumTrainer:
    """
    Trainer for quantum classification circuits.
    
    Handles the complete training loop including validation, early stopping,
    learning rate decay, checkpointing, and metrics logging to W&B.
    
    Args:
        model: QuantumClassifier instance to train
        cfg: Configuration object containing all training parameters
        optimizer: Optimizer instance with step_and_cost method
        loss_fn: Loss function with signature (weights, inputs, labels, quantum_circuit, ...)
        wandb: Weights & Biases run instance for logging (optional)
        init_weights: Initial circuit weights (optional, can be in cfg or passed directly)
    """
    
    def __init__(
        self,
        model: Any,
        cfg: Any,
        optimizer: Callable,
        loss_fn: Callable,
        wandb: Optional[Any] = None,
        init_weights: Optional[np.ndarray] = None
    ) -> None:
        self.model = model
        self.circuit = model.fetch_circuit()
        self.backend = model.backend
        self.cfg = cfg
        
        # Extract training parameters from config
        self.batch_size = cfg.training.batch_size
        self.lr_decay = cfg.training.lr_decay
        self.epochs = cfg.training.epochs
        self.patience = cfg.training.patience
        self.saving = cfg.training.save_checkpoints
        self.loss_type = cfg.loss.type
        self.improv = cfg.training.improv
        
        # Early stopping configuration
        self.min_epochs = cfg.training.get('min_epochs', 4)
        self.improvement_window = cfg.training.get('improvement_window', 2)
        
        # Learning rate
        self.lr = cfg.optimizer.learning_rate
        
        # Training state
        self.init_weights = init_weights
        self.current_weights = self.init_weights
        self.current_epoch = 0
        self.optim = optimizer
        self.quantum_loss = loss_fn
        self.wandb = wandb
        self.history: Dict[str, List[float]] = {'train': [], 'val': [], 'auc': []}
        
        # Directories
        self.save_dir: Optional[str] = None
        self.checkpoint_dir: Optional[str] = None
        
        logger.info(f'Optimizer: {self.optim.__class__.__name__} | Learning rate: {self.lr}')
        logger.info(f'Backend: {self.backend}')
        logger.info(f'Training config: epochs={self.epochs}, batch_size={self.batch_size}, '
                   f'patience={self.patience}, lr_decay={self.lr_decay}')
        logger.info(f'Early stopping: min_epochs={self.min_epochs}, '
                   f'improvement_window={self.improvement_window}, improv_threshold={self.improv}')
    
    def iteration(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        train: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Execute single training or validation iteration.
        
        Args:
            data: Input batch
            labels: Corresponding labels
            train: If True, perform gradient update
            
        Returns:
            If train=True: loss value
            If train=False: (loss value, prediction scores)
        """
        if train:
            self.current_weights, cost = self.optim.step_and_cost(
                self.quantum_loss,
                self.current_weights,
                inputs=data,
                labels=labels,
                quantum_circuit=self.circuit,
                loss_type=self.loss_type
            )
            return float(cost)
        else:
            cost, scores = self.quantum_loss(
                self.current_weights,
                inputs=data,
                labels=labels,
                quantum_circuit=self.circuit,
                return_scores=True,
                loss_type=self.loss_type
            )
            return float(cost), scores
    
    def _plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epoch: int
    ) -> plt.Figure:
        """
        Create visualization of true vs predicted labels.
        
        Generates a scatter plot with histograms showing the distribution
        of predictions for each class.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities/scores
            epoch: Current epoch number
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        class_0_mask = y_true == 0
        class_1_mask = y_true == 1
        
        ax1.scatter(
            np.arange(len(y_true))[class_0_mask],
            y_pred[class_0_mask],
            alpha=0.5,
            label='Class 0',
            s=10
        )
        ax1.scatter(
            np.arange(len(y_true))[class_1_mask],
            y_pred[class_1_mask],
            alpha=0.5,
            label='Class 1',
            s=10
        )
        ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Decision boundary')
        ax1.set_xlabel('Sample index')
        ax1.set_ylabel('Predicted score')
        ax1.set_title(f'Predictions vs True Labels - Epoch {epoch}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions by class
        ax2.hist(
            y_pred[class_0_mask],
            bins=30,
            alpha=0.6,
            label='Class 0',
            color='blue',
            edgecolor='black'
        )
        ax2.hist(
            y_pred[class_1_mask],
            bins=30,
            alpha=0.6,
            label='Class 1',
            color='orange',
            edgecolor='black'
        )
        ax2.axvline(x=0.5, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Decision boundary')
        ax2.set_xlabel('Predicted score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Predictions by Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _check_early_stopping(self, n_epoch: int, n_decays: int, last_decay: int) -> Tuple[bool, int, int]:
        """
        Check early stopping criteria based on validation metrics.
        
        Args:
            n_epoch: Current epoch number
            n_decays: Number of learning rate decays performed
            last_decay: Epoch at which last decay occurred
            
        Returns:
            Tuple of (should_stop, updated_n_decays, updated_last_decay)
        """
        # Only check after minimum epochs have passed
        if n_epoch <= self.min_epochs:
            return False, n_decays, last_decay
        
        # Ensure we have enough history for comparison
        if len(self.history['auc']) < (self.improvement_window + 1):
            return False, n_decays, last_decay
        
        # Calculate improvement over the improvement window
        recent_val_metrics = self.history['auc'][-self.improvement_window:]
        previous_val_metric = self.history['auc'][-(self.improvement_window + 1)]
        improvement = np.mean(recent_val_metrics) - previous_val_metric
        
        if improvement < self.improv:
            if self.lr_decay:
                # Check if we should decay learning rate
                epochs_since_decay = n_epoch - last_decay
                if (n_decays < self.patience) and (epochs_since_decay >= self.improvement_window):
                    last_decay = n_epoch
                    n_decays += 1
                    self.optim.stepsize *= 0.5
                    logger.info(
                        f'No improvement observed. Learning rate decayed to '
                        f'{self.optim.stepsize:.6f} at epoch {n_epoch} '
                        f'(decay {n_decays}/{self.patience})'
                    )
                    return False, n_decays, last_decay
                elif n_decays >= self.patience:
                    logger.info(
                        f"Early stopping after {self.patience} decay steps with no improvement"
                    )
                    return True, n_decays, last_decay
            else:
                logger.info(
                    f"Early stopping: no improvement over last {self.improvement_window} epochs "
                    f"(threshold: {self.improv})"
                )
                return True, n_decays, last_decay
        
        return False, n_decays, last_decay
    
    def run_training_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Execute complete training loop with validation and early stopping.
        
        Args:
            train_loader: DataLoader for training data (yields data, labels, scores, tags)
            val_loader: DataLoader for validation data (yields data, labels, scores, tags)
            
        Returns:
            Dictionary containing training history
        """
        self.print_params('Initial weights: ')
        n_decays = 0
        last_decay = 0
        complete = False
        
        for n_epoch in tqdm(range(self.epochs + 1), desc="Training epochs"):
            sample_counter = 0
            batch_yield = 0
            self.current_epoch = n_epoch
            losses = 0.0
            
            # Early stopping check
            should_stop, n_decays, last_decay = self._check_early_stopping(
                n_epoch, n_decays, last_decay
            )
            
            if should_stop:
                self.save(self.save_dir, name='trained_model.pickle')
                complete = True
                break
            
            # Training phase
            if n_epoch > 0:
                logger.info("Starting training phase")
                start_time = time.time()
                
                for data, labels, _, _ in tqdm(
                    train_loader,
                    desc=f"Epoch {n_epoch} training",
                    total=int(len(train_loader.dataset) / self.batch_size)
                ):
                    sample_counter += data.shape[0]
                    batch_yield += 1
                    loss = self.iteration(data, labels=labels, train=True)
                    losses += loss
                    
                    if self.wandb is not None:
                        self.wandb.log({'train_loss_batch': loss, 'epoch': n_epoch})
                
                end_time = time.time()
                train_loss = losses / batch_yield
                self.print_params('Updated weights:')
                logger.info('Starting validation phase')
                
                if self.wandb is not None:
                    self.wandb.log({'train_loss_epoch': train_loss, 'epoch': n_epoch})
            else:
                logger.info('Running initial validation pass')
                start_time = time.time()
            
            # Validation phase - accumulate all predictions and labels
            val_loss = 0.0
            val_batch_yield = 0
            all_val_scores = []
            all_val_labels = []
            
            for data, labels, _, _ in tqdm(
                val_loader,
                desc=f"Epoch {n_epoch} validation",
                total=int(len(val_loader.dataset) / self.batch_size)
            ):
                loss, scores = self.iteration(data, labels=labels, train=False)
                val_loss += loss
                
                # Accumulate predictions and labels
                if isinstance(scores, np.ndarray):
                    all_val_scores.extend(scores.flatten().tolist())
                else:
                    all_val_scores.append(float(scores))
                
                if isinstance(labels, np.ndarray):
                    all_val_labels.extend(labels.flatten().tolist())
                else:
                    all_val_labels.append(float(labels))
                
                val_batch_yield += 1
            
            # Compute validation metrics
            val_loss = val_loss / val_batch_yield
            val_labels_array = np.array(all_val_labels)
            val_scores_array = np.array(all_val_scores)
            val_auc = roc_auc_score(val_labels_array, val_scores_array)
            val_std = np.std(val_scores_array)
            val_score_mean = np.mean(val_scores_array)
            
            # Create and log prediction plot
            if self.wandb is not None:
                fig = self._plot_predictions(val_labels_array, val_scores_array, n_epoch)
                self.wandb.log({
                    'val_predictions_plot': self.wandb.Image(fig),
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_score_mean': val_score_mean,
                    'val_score_std': val_std,
                    'epoch': n_epoch
                })
                plt.close(fig)
            
            # Logging to console
            elapsed_time = time.time() - start_time
            
            if n_epoch > 0:
                log_msg = (
                    f'Epoch {n_epoch}: Network with {len(self.model.all_wires)} qubits trained on '
                    f'{sample_counter} samples in {batch_yield} batches | '
                    f'Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | '
                    f'Val AUC: {val_auc:.3f} | Val pred (mean±std): {val_score_mean:.3f}±{val_std:.3f} | '
                    f'Time: {elapsed_time:.1f}s'
                )
                logger.info(log_msg)
                self.history['train'].append(train_loss)
            else:
                log_msg = (
                    f'Epoch {n_epoch} (initial validation): Val loss: {val_loss:.3f} | '
                    f'Val AUC: {val_auc:.3f} | Val pred (mean±std): {val_score_mean:.3f}±{val_std:.3f}'
                )
                logger.info(log_msg)
            
            self.history['val'].append(val_loss)
            self.history['auc'].append(val_auc)
            
            # Save checkpoints
            if self.saving:
                name = None
                checkpoint_frequency = self.cfg.training.get('checkpoint_frequency', 5)
                
                if n_epoch == self.epochs:
                    name = 'trained_model.pickle'
                elif n_epoch == 0:
                    name = 'init_weights.pickle'
                elif n_epoch % checkpoint_frequency == 0:
                    # Save periodic checkpoints
                    name = None  # Will use epoch-based naming
                
                if name is not None or n_epoch % checkpoint_frequency == 0:
                    self.save(self.save_dir, name=name)
        
        # Save final history
        if not complete and self.save_dir is not None:
            history_path = os.path.join(self.save_dir, 'history.pickle')
            import pickle
            with open(history_path, 'wb') as f:
                pickle.dump(self.history, f)
            logger.info(f"Training history saved to {history_path}")
        
        return self.history
    
    def print_params(self, prefix: Optional[str] = None) -> None:
        """
        Print current circuit parameters.
        
        Args:
            prefix: Optional text to print before parameters
        """
        if prefix is not None:
            logger.info(prefix)
        logger.info(f'Weights shape: {self.current_weights.shape}')
        logger.info(f'Weights (first 5): {self.current_weights.flatten()[:5]}...')
    
    def save(self, save_dir: str, name: Optional[str] = None) -> None:
        """
        Save model weights and optimizer state.
        
        Args:
            save_dir: Directory for saving
            name: Filename for weights. If None, uses epoch-based naming
        """
        if save_dir is None:
            logger.warning("Save directory not set, skipping save")
            return
        
        opt_name = None
        
        if name is None:
            if self.current_epoch > 100:
                name = f'ep{self.current_epoch:03}.pickle'
                opt_name = f'optimizer_ep{self.current_epoch:03}.json'
            else:
                name = f'ep{self.current_epoch:02}.pickle'
                opt_name = f'optimizer_ep{self.current_epoch:02}.json'
        
        # Determine save location
        if 'trained' not in name and 'init' not in name:
            target_dir = self.checkpoint_dir
        else:
            target_dir = save_dir
            opt_name = 'optimizer.json'
        
        # Save weights
        import pickle
        weights_path = os.path.join(target_dir, name)
        with open(weights_path, 'wb') as f:
            pickle.dump({'weights': self.current_weights}, f)
        logger.info(f"Weights saved to {weights_path}")
        
        # Save optimizer state
        try:
            optim_dict = {
                'stepsize': self.optim.stepsize,
                'beta1': getattr(self.optim, 'beta1', None),
                'beta2': getattr(self.optim, 'beta2', None),
                'epsilon': getattr(self.optim, 'eps', None),
                'fm': getattr(self.optim, 'fm', None),
                'sm': getattr(self.optim, 'sm', None),
                't': getattr(self.optim, 't', None)
            }
            
            opt_path = os.path.join(target_dir, opt_name)
            with open(opt_path, 'w') as f:
                json.dump(optim_dict, f, indent=2, default=str)
            logger.info(f"Optimizer state saved to {opt_path}")
        except Exception as e:
            logger.error(f"Failed to save optimizer state: {str(e)}")
    
    def get_current_epoch(self) -> int:
        """
        Get current epoch number.
        
        Returns:
            Current epoch
        """
        return self.current_epoch
    
    def set_current_epoch(self, epoch: int) -> None:
        """
        Set epoch number for resuming training.
        
        Args:
            epoch: Epoch number to resume from
        """
        self.current_epoch = epoch + 1
        logger.info(f"Resuming training from epoch {epoch + 1}")
    
    def set_directories(self, save_dir: str) -> None:
        """
        Configure save directories for checkpoints and models.
        
        Args:
            save_dir: Base directory for saving
        """
        self.save_dir = save_dir
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        pathlib.Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory: {save_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def fetch_history(self) -> Dict[str, List[float]]:
        """
        Retrieve training history.
        
        Returns:
            Dictionary with train/val losses and metrics
        """
        return self.history