"""
Orchestrates the QM9 quantum regression training pipeline.

Usage:
    python train.py --config configs/YAML/my_run.yml

Author: Dr. Aritra Bal (ETP)
Date: March 03, 2026
"""

import argparse
import pathlib
import sys

import pennylane as qml
from loguru import logger
from pathlib import Path
from configs.configuration import Config
from data_handlers.qm9_h5_dataloader import build_loaders_from_config
from quantum.architectures import QuantumCircuit
from quantum.trainer import QuantumTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QML regressor on QM9")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g. configs/YAML/run_001.yml)",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    config = Config(args.config)

    # Save resolved config immediately so the exact run parameters are recorded.
    model_dir = Path(config.paths.model_dir) / Path(config.setup.run_id)
    log_dir = model_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "train.log", rotation="10 MB")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Model directory: {model_dir}")
    config.save(model_dir / "config.yaml")
    
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader = build_loaders_from_config(config)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    # Wires = n_qubits; shots=None uses exact statevector simulation.
    device = qml.device(
        config.model.device,
        wires=config.model.n_qubits,
        shots=config.model.shots,
    )
    logger.info(f"Device: {config.model.device} | wires={config.model.n_qubits}")

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------
    circuit = QuantumCircuit(config, device)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    # Only Adam is supported for now; the name field is informational.
    if config.optimizer.name != "Adam":
        logger.warning(
            f"Optimizer '{config.optimizer.name}' is not implemented; "
            f"falling back to AdamOptimizer."
        )
    optimizer = qml.AdamOptimizer(stepsize=config.optimizer.lr)
    logger.info(f"Optimizer: AdamOptimizer | lr={config.optimizer.lr}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = QuantumTrainer(config, circuit, optimizer, train_loader, val_loader, model_dir)
    history = trainer.train()

    logger.info(
        f"Run '{config.setup.run_id}' complete | "
        f"best val_mae={min(history['val_mae']):.4f} (norm)"
    )


if __name__ == "__main__":
    main()
