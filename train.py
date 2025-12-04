"""
End-to-end training script for quantum circuit classifiers on protein-complex data.

This script orchestrates the complete training pipeline:
- Configuration management from YAML
- Data loading from PISToN framework outputs
- Quantum circuit initialization with custom operations
- Training loop with validation and checkpointing
- W&B experiment tracking

Author: Dr. Aritra Bal (ETP)
Date: December 01, 2025
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
from loguru import logger
import pennylane as qml
from pennylane import numpy as pnp
import src.losses as losses

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from data_handlers.file_paths import get_output_paths, fetch_subfolders
from data_handlers.dataloader import create_dataloader
from src.architectures import QuantumClassifier
from src.trainer import QuantumTrainer
from configs.config import load_config
from models.operations import encoding, entanglement, operation


def setup_wandb(cfg):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        cfg: Configuration object with wandb settings
        
    Returns:
        W&B run object if enabled, None otherwise
    """
    if not cfg.wandb.enabled:
        logger.info("W&B tracking disabled")
        return None
    
    try:
        import wandb
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config={
                'circuit': cfg.circuit.__dict__,
                'training': cfg.training.__dict__,
                'optimizer': cfg.optimizer.__dict__,
                'loss': cfg.loss.__dict__
            }
        )
        logger.info(f"W&B initialized: {cfg.wandb.project}/{cfg.wandb.name}")
        return run
    except ImportError:
        logger.warning("wandb not installed, proceeding without tracking")
        return None
    except Exception as e:
        logger.error(f"W&B initialization failed: {e}")
        return None

def create_optimizer(cfg):
    """
    Create PennyLane optimizer based on config.
    
    Args:
        cfg: Configuration object with optimizer settings
    
    Returns:
        Configured optimizer instance
    """
    opt_name = cfg.optimizer.name.lower()
    lr = cfg.optimizer.learning_rate
    
    if opt_name == 'adam':
        optimizer = qml.AdamOptimizer(
            stepsize=lr,
            beta1=cfg.optimizer.beta1,
            beta2=cfg.optimizer.beta2,
            eps=cfg.optimizer.eps
        )
    elif opt_name == 'gradientdescent' or opt_name == 'gd':
        optimizer = qml.GradientDescentOptimizer(stepsize=lr)
    elif opt_name == 'nesterovmomentum':
        optimizer = qml.NesterovMomentumOptimizer(
            stepsize=lr,
            momentum=cfg.optimizer.momentum
        )
    elif opt_name == 'adagrad':
        optimizer = qml.AdagradOptimizer(stepsize=lr)
    elif opt_name == 'rmsprop':
        optimizer = qml.RMSPropOptimizer(
            stepsize=lr,
            decay=cfg.optimizer.decay,
            eps=cfg.optimizer.eps
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    logger.info(f"Optimizer created: {opt_name} with lr={lr}")
    return optimizer


def initialize_weights(cfg, seed=None):
    """
    Initialize circuit weights based on configuration.
    
    Args:
        cfg: Configuration object with circuit settings
        seed: Random seed for reproducibility
        
    Returns:
        Initialized weight array
    """
    if seed is not None:
        np.random.seed(int(seed) if seed.isdigit() else hash(seed) % (2**32))
    
    # Weight shape depends on circuit architecture
    # For now, use a reasonable default based on number of qubits
    n_qubits = cfg.circuit.qubits
    n_layers = cfg.circuit.get('layers', 1)
    
    # Typical parameterized circuit has 3 parameters per qubit per layer
    weight_shape = (n_layers, n_qubits, 3)
    
    weights = pnp.random.uniform(
        low=-np.pi,
        high=np.pi,
        size=weight_shape,
        requires_grad=True
    )
    
    logger.info(f"Weights initialized with shape {weight_shape}")
    return weights


def create_dataloaders(cfg):
    """
    Create training and validation dataloaders.
    
    Args:
        cfg: Configuration object with data settings
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get data paths
    subfolders = fetch_subfolders(cfg.directories.base_data)
    data_path = subfolders['grid_maps']
    
    logger.info(f"Loading data from: {data_path}")
    
    # Determine train/val split paths
    # Assuming data is organized in subdirectories or separate paths
    if hasattr(cfg.data, 'train_path') and hasattr(cfg.data, 'val_path'):
        train_paths = [cfg.data.train_path]
        val_paths = [cfg.data.val_path]
    else:
        # Use single path for both, DataLoader will handle splitting
        train_paths = [data_path]
        val_paths = [data_path]
    
    # Create train dataloader
    train_loader = create_dataloader(
        paths=train_paths,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers
    )
    
    # Create validation dataloader
    val_loader = create_dataloader(
        paths=val_paths,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    return train_loader, val_loader


def main():
    """Main training pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train quantum classifier on protein-complex data"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    cfg = load_config(args.config)
    
    # Create output directories
    output_dir = get_output_paths(
        output_dir=cfg.directories.models,
        seed=cfg.seed # not the determinism seed, this one is just for tracking
    )
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize W&B
    wandb_run = setup_wandb(cfg)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg)
    
    # Initialize circuit weights
    logger.info("Initializing circuit weights...")
    initial_weights = initialize_weights(cfg, seed=cfg.determinism.seed_value)
    
    # Create quantum classifier
    logger.info("Creating quantum classifier...")
    model = QuantumClassifier(
        wires=cfg.circuit.qubits,
        device_name=cfg.circuit.device_name,
        backend=cfg.circuit.backend,
        shots=cfg.circuit.shots,
        encoding=encoding,
        entanglement=entanglement,
        operation=operation,
        initial_weights=initial_weights,
        approx=cfg.circuit.approx,
        use_aux_wire=cfg.circuit.use_aux_wire,
        total_batches=min(
            cfg.training.train_max_n // cfg.training.batch_size,
            len(train_loader)
        )
    )
    
    # Set up the circuit QNodes
    model.set_circuit()
    model.print_circuit_info()
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(cfg, initial_weights)
    
    # Create loss function
    logger.info("Creating loss function...")
    loss_fn = losses.quantum_loss
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = QuantumTrainer(
        model=model,
        lr=cfg.optimizer.learning_rate,
        optimizer=optimizer,
        loss_fn=loss_fn,
        save=cfg.training.save_checkpoints,
        train_max_n=cfg.training.train_max_n,
        valid_max_n=cfg.training.valid_max_n,
        epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        improv=cfg.training.improv,
        wandb=wandb_run,
        lr_decay=cfg.training.lr_decay,
        loss_type=cfg.loss.type,
        batch_size=cfg.training.batch_size,
        init_weights=initial_weights
    )
    
    # Set save directories
    trainer.set_directories(output_dir)
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.model.load_weights_from_file(args.resume)
        # Extract epoch number from filename if possible
        try:
            epoch_num = int(args.resume.split('ep')[-1].split('.')[0])
            trainer.set_current_epoch(epoch_num)
        except:
            logger.warning("Could not determine epoch from checkpoint filename")
    
    # Run training
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    history = trainer.run_training_loop(train_loader, val_loader)
    
    # Training summary
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final training loss: {history['train'][-1]:.4f}")
    logger.info(f"Final validation loss: {history['val'][-1]:.4f}")
    logger.info(f"Final validation AUC: {history['auc'][-1]:.4f}")
    logger.info(f"Best validation AUC: {max(history['auc']):.4f} at epoch {np.argmax(history['auc'])}")
    logger.info(f"Model saved to: {output_dir}")
    
    # Close W&B run
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("W&B run finished")
    
    logger.info("All operations completed successfully")


if __name__ == "__main__":
    main()