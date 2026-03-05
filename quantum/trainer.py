"""
Training and validation loop for the variational quantum circuit regressor.

Author: Dr. Aritra Bal (ETP)
Date: March 03, 2026
"""

import pathlib
import numpy as np
import pennylane.numpy as pnp
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from typing import Any, Dict, List

from quantum.architectures import QuantumCircuit, _HARTREE_TO_EV


class QuantumTrainer:
    """
    Manages the training loop, validation, LR decay, and early stopping
    for a QuantumCircuit regressor.

    LR decay logic:
        If validation MAE does not improve by more than decay_threshold for
        decay_patience consecutive epochs, multiply the learning rate by
        decay_factor. After patience such decays with no improvement,
        training stops.
    """

    def __init__(
        self,
        config: Any,
        circuit: QuantumCircuit,
        optimizer: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: pathlib.Path,
    ) -> None:
        """
        Args:
            config       : Config namespace object.
            circuit      : Initialised QuantumCircuit instance.
            optimizer    : PennyLane optimizer (e.g. AdamOptimizer).
            train_loader : DataLoader for training split.
            val_loader   : DataLoader for validation split.
        """
        self.config       = config
        self.circuit      = circuit
        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.epochs          = config.setup.epochs
        self.lr_decay        = config.optimizer.lr_decay
        self.decay_factor    = config.optimizer.decay_factor
        self.decay_patience  = config.optimizer.decay_patience
        self.decay_threshold = config.optimizer.decay_threshold
        self.patience        = config.optimizer.patience

        self.save_dir = save_dir

        self.history: Dict[str, List[float]] = {
            "train_mae": [], "val_mae": []
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(t: Any) -> np.ndarray:
        """Convert torch tensor or pnp array to plain numpy."""
        if hasattr(t, "numpy"):
            return t.numpy()
        return np.asarray(t)

    @staticmethod
    def _mae(pred: np.ndarray, target_norm: np.ndarray) -> float:
        """Mean absolute error in normalised space."""
        return float(np.mean(np.abs(pred - target_norm)))

    @staticmethod
    def _mse(pred: np.ndarray, target_norm: np.ndarray) -> float:
        """Mean squared error in normalised space."""
        return float(np.mean((pred - target_norm) ** 2))

    def _normalise_target(self, target: np.ndarray) -> np.ndarray:
        """Convert Hartree gap to normalised [-1, 1] range."""
        return target * _HARTREE_TO_EV / 10.0 - 1.0
    def _normalise_pred(self, pred: np.ndarray) -> np.ndarray:
        """Convert normalised prediction back to eV."""
        return (1 - pred) * 10.0
    # ------------------------------------------------------------------
    # Single epoch loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """
        Run one training epoch.

        Args:
            epoch: Current epoch index (for tqdm label).

        Returns:
            Mean training MSE over all batches.
        """
        accumulated_loss_fn = 0.0
        total_mae = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for nodes, edges, targets, _ in pbar:
            node_np   = self._to_numpy(nodes)
            edge_np   = self._to_numpy(edges)
            target_np = self._to_numpy(targets)

            def batch_loss(w: pnp.ndarray) -> pnp.ndarray:
                return self.circuit.loss(w, node_np, edge_np, target_np)

            self.circuit.weights, self.circuit.extra_weights, cost = self.optimizer.step_and_cost(
                batch_loss, self.circuit.weights, self.circuit.extra_weights
            )

            batch_loss_fn = float(cost)
            accumulated_loss_fn += batch_loss_fn
            #total_mae += float(np.sqrt(batch_loss_fn))
            n_batches += 1
            pbar.set_postfix(mae_ev=f"{accumulated_loss_fn / n_batches:.4f}")#, mae_ev=f"{total_mae*10. / n_batches:.4f}
        return accumulated_loss_fn / max(n_batches, 1)

    def _val_epoch(self, epoch: int):
        """
        Run one validation epoch.

        Args:
            epoch: Current epoch index (for tqdm label).

        Returns:
            Tuple of (mean_mse, mean_mae) over all validation batches.
        """
        all_pred  = []
        all_norm  = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)
        for nodes, edges, targets, _ in pbar:
            node_np   = self._to_numpy(nodes)
            edge_np   = self._to_numpy(edges)
            target_np = self._to_numpy(targets)
            
            pred = self.circuit.qnode(self.circuit.weights, node_np, edge_np)
            norm_pred = self._normalise_pred(pred)
            pred_np = np.atleast_1d(np.array(norm_pred))
            target_np  = np.atleast_1d(target_np)

            all_pred.append(pred_np)
            all_norm.append(target_np)

        all_pred = np.concatenate(all_pred)
        all_norm = np.concatenate(all_norm)

        return self._mae(all_pred, all_norm) # also averages it out

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop with LR decay and early stopping.

        Returns:
            Training history dictionary with keys:
                train_mae, val_mse, val_mae
        """
        best_mae     = float("inf")
        no_improve   = 0   # consecutive epochs without improvement
        decay_count  = 0   # number of LR decays performed

        epoch_bar = tqdm(range(1, self.epochs + 1), desc="Epochs", unit="epoch")

        for epoch in epoch_bar:
            # --- training ---
            train_mae = self._train_epoch(epoch)

            # --- validation ---
            val_mae = self._val_epoch(epoch)

            # --- logging ---
            # Report MAE in physical units (eV).
            logger.info(
                f"Epoch {epoch:03d} | "
                f"train_mae={train_mae:.4f} | "
                f"val_mae={val_mae:.4f} eV | "
                f"lr={self.optimizer.stepsize:.6f}"
            )

            self.history["train_mae"].append(train_mae)
            self.history["val_mae"].append(val_mae)

            epoch_bar.set_postfix(
                val_mae_ev=f"{val_mae:.4f}"
            )

            # --- LR decay / early stopping ---
            if self.lr_decay:
                if best_mae - val_mae > self.decay_threshold:
                    best_mae   = val_mae
                    no_improve = 0
                    self._save_weights("weights_best.npy")
                else:
                    no_improve += 1

                if no_improve >= self.decay_patience:
                    self.optimizer.stepsize *= self.decay_factor
                    decay_count += 1
                    no_improve  = 0
                    logger.info(
                        f"LR decayed to {self.optimizer.stepsize:.6f} "
                        f"(decay {decay_count}/{self.patience})"
                    )

                    if decay_count >= self.patience:
                        logger.info(
                            f"Early stopping: {self.patience} decays with no improvement."
                        )
                        break

        self._save_weights("weights_final.npy")
        logger.info(f"Training complete. Weights saved to {self.save_dir}")
        return self.history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_weights(self, filename: str) -> None:
        """
        Save current circuit weights as a numpy file.

        Args:
            filename: File name (placed inside config.paths.model_dir).
        """
        path = self.save_dir / filename
        np.save(str(path), np.array(self.circuit.weights))
        logger.info(f"Weights saved to {path}")
