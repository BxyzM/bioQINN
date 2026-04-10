"""
Inference and evaluation for the QM9 quantum regression model.

Loads a trained QuantumCircuit from saved weights, runs inference on the
test split, reports MAE, and optionally computes the QFIM.

Usage:
    python test.py --config configs/YAML/my_run.yml --weights path/to/run/trained_model/weights_best.npy
"""

import argparse
import pathlib

import h5py
import matplotlib
import numpy as np
import pennylane as qml
from loguru import logger
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.configuration import Config
from data_handlers.qm9_h5_dataloader import build_loaders_from_config
from quantum.architectures import QuantumCircuit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained QML regressor on QM9 test set")
    parser.add_argument("--config",  type=str, required=True,
                        help="Path to YAML config used during training")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to weights_best.npy (or weights_final.npy)")
    parser.add_argument("--test-n",  type=int, default=None,
                        help="Optional cap on the number of test samples to evaluate")
    parser.add_argument("--qfim",    action="store_true", default=False,
                        help="Compute Quantum Fisher Information Matrices")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for outputs. Defaults to <model_dir>/<run_id>/eval/")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_weights(circuit: QuantumCircuit, weights_path: str) -> None:
    """Load weights and extra_weights from .npy or .npz files saved by the trainer."""
    weights_path = pathlib.Path(weights_path)
    extra_path   = weights_path.parent / ("extra_" + weights_path.name)

    def _load_array(path: pathlib.Path) -> np.ndarray:
        data = np.load(str(path))
        if isinstance(data, np.lib.npyio.NpzFile):
            # .npz archive: extract the first (and typically only) stored array
            key = list(data.keys())[0]
            arr = np.array(data[key])
            data.close()
            logger.warning(f"{path} is a .npz archive; extracted key '{key}'")
            return arr
        return np.array(data)

    loaded_w  = _load_array(weights_path)
    loaded_ew = _load_array(extra_path)

    expected_w_shape  = circuit.weights.shape
    expected_ew_shape = circuit.extra_weights.shape
    if loaded_w.shape != expected_w_shape:
        raise ValueError(
            f"Weight shape mismatch: file={loaded_w.shape}, circuit expects={expected_w_shape}"
        )
    if loaded_ew.shape != expected_ew_shape:
        raise ValueError(
            f"Extra-weight shape mismatch: file={loaded_ew.shape}, circuit expects={expected_ew_shape}"
        )

    circuit.weights       = qml.numpy.array(loaded_w,  requires_grad=False)
    circuit.extra_weights = qml.numpy.array(loaded_ew, requires_grad=False)
    logger.info(f"Weights loaded from {weights_path}  shape={loaded_w.shape}")
    logger.info(f"Extra weights loaded from {extra_path}  shape={loaded_ew.shape}")


def _run_inference(circuit: QuantumCircuit, test_loader) -> tuple:
    """Run forward pass over the test set. Returns (all_pred, all_true)."""
    all_pred, all_true = [], []

    for nodes, edges, targets, _ in tqdm(test_loader, desc="Inference"):
        node_np   = np.asarray(nodes)
        edge_np   = np.asarray(edges)
        target_np = np.asarray(targets)

        raw_pred     = circuit.qnode(circuit.weights, circuit.extra_weights, node_np, edge_np)
        coeff_values = np.array([
            float(circuit.extra_weights[circuit.extra_weights_IDX[f"coeff{j+1}"]])
            for j in range(circuit.read_qubits)
        ])
        bias      = np.sum(np.abs(coeff_values))
        affine    = float(circuit.extra_weights[circuit.extra_weights_IDX["out_bias"]])
        raw_scalar = float(np.atleast_1d(np.array(raw_pred))[0])
        pred      = float(bias + affine - raw_scalar)

        all_pred.append(pred)
        all_true.append(float(np.atleast_1d(target_np)[0]))

    return np.array(all_pred), np.array(all_true)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_results(pred: np.ndarray, true: np.ndarray, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histogram overlay
    fig, ax = plt.subplots()
    bins_t, edges_t = np.histogram(true, bins=100, range=(0, 20))
    bins_p, edges_p = np.histogram(pred, bins=100, range=(0, 20))
    ax.stairs(bins_t, edges_t, label="True",      fill=False, edgecolor="blue")
    ax.stairs(bins_p, edges_p, label="Predicted", fill=False, edgecolor="red")
    ax.set_xlabel("HOMO-LUMO gap (eV)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(out_dir / "gap_histogram.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    # Scatter plot
    r = float(np.corrcoef(true, pred)[0, 1])
    fig, ax = plt.subplots()
    ax.plot(true, pred, "o", alpha=0.3, markersize=2)
    ax.plot([0, 12], [0, 12], "k--")
    ax.set_xlabel("True gap (eV)")
    ax.set_ylabel("Predicted gap (eV)")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_title(f"Predicted vs True  |  r = {r:.3f}")
    fig.savefig(out_dir / "scatter.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot_qfim(fisher_matrices: np.ndarray, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    avg_qfi = np.mean(fisher_matrices, axis=0)

    colors = ["#0066FF", "white", "#FF0066"]
    cmap   = matplotlib.colors.LinearSegmentedColormap.from_list("bwr", colors, N=256)
    norm   = matplotlib.colors.Normalize(vmin=-0.25, vmax=0.25)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.matshow(avg_qfi, cmap=cmap, norm=norm)
    ax.set_xlabel("Parameter index")
    ax.set_ylabel("Parameter index")
    ax.set_title("Average QFIM")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="QFI value")
    plt.tight_layout()
    fig.savefig(out_dir / "qfim_average.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"QFIM plot saved to {out_dir / 'qfim_average.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = Config(args.config)

    # Override train flag so build_loaders_from_config returns the test loader.
    config.setup.train = False
    if args.test_n is not None:
        config.setup.test_n = args.test_n

    model_dir = pathlib.Path(config.paths.model_dir) / config.setup.run_id
    out_dir   = pathlib.Path(args.out_dir) if args.out_dir else model_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "test.log", rotation="10 MB")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    test_loader = build_loaders_from_config(config)
    logger.info(f"Test set: {len(test_loader)} batches")

    # ------------------------------------------------------------------
    # Circuit + weights
    # ------------------------------------------------------------------
    device = qml.device(
        config.model.device,
        wires=config.model.n_qubits,
        shots=config.model.shots,
    )
    circuit = QuantumCircuit(config, device)
    _load_weights(circuit, args.weights)

    # DEBUG: inspect loaded weights before inference
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    pred, true = _run_inference(circuit, test_loader)

    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    r   = float(np.corrcoef(true, pred)[0, 1])
    logger.info(f"Test MAE  = {mae:.4f} eV")
    logger.info(f"Test MSE  = {mse:.4f} eV²")
    logger.info(f"Pearson r = {r:.4f}")

    _plot_results(pred, true, out_dir / "plots")

    # ------------------------------------------------------------------
    # QFIM (optional)
    # ------------------------------------------------------------------
    fisher_matrices = None
    if args.qfim:
        logger.info("Computing QFIM...")
        fisher_matrices, _ = circuit.run_fisher_computation(test_loader)
        logger.info(f"QFIM shape: {fisher_matrices.shape}")
        _plot_qfim(fisher_matrices, out_dir / "plots")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = out_dir / "results.h5"
    with h5py.File(results_path, "w") as f:
        f.create_dataset("pred",  data=pred)
        f.create_dataset("true",  data=true)
        f.attrs["mae"] = mae
        f.attrs["mse"] = mse
        f.attrs["r"]   = r
        if fisher_matrices is not None:
            f.create_dataset("fisher_matrices", data=fisher_matrices)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
