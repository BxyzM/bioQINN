# bioQINN: Regressing Molecular Structure with Quantum Machine Learning

**Authors:** [Aritra Bal](https://etpwww.etp.kit.edu/~abal/) 

**Contact:** [Email](mailto:aritra.bal@do-not-spam.kit.edu)

Dataset description paper: [Ramakrishnan et al., *Scientific Data* (2014)](https://www.nature.com/articles/sdata201422)

> Work in Progress. This README is updated over time.

---

## Overview

**bioQINN** is a quantum graph neural network for molecular property prediction on the [QM9](https://www.nature.com/articles/sdata201422) dataset. The primary regression target is the **HOMO-LUMO gap** — the energy difference Δε between the **Highest Occupied Molecular Orbital (HOMO)** and the **Lowest Unoccupied Molecular Orbital (LUMO)** — a quantity of central importance in computational chemistry, governing optical absorption, reactivity, and charge transport properties.

The data embedding strategy is inspired by the **1P1Q architecture** developed for jet tagging and anomaly detection in high-energy physics [Bal et al., Phys. Rev. D 112, 2025](https://link.aps.org/doi/10.1103/l8y2-87vq). Details of the encoding, entanglement, and measurement schemes are documented directly in `quantum/architectures.py`.

---

## Repository Structure

```
qm9/
├── configs/
│   ├── YAML/
│   │   └── qm9.yaml           # User-facing hyperparameter configuration
│   ├── configuration.py       # Config loader: merges defaults + YAML overrides
│   └── defaults.py            # Default parameter values
├── data_handlers/
│   ├── qm9_dataloader.py      # PyG-based dense dataloader
│   └── qm9_h5_dataloader.py   # HDF5-backed dataloader (used in training)
├── data_processors/
│   └── h5_maker_qm9.py        # One-time HDF5 dataset creation script
├── quantum/
│   ├── architectures.py       # Variational quantum circuit definition
│   └── trainer.py             # Training loop, validation, LR decay, checkpointing
└── train.py                   # Main training entrypoint
```

---

## Dependencies

See requirements.txt

---

## Usage

### Step 1: Prepare the Dataset (run once)

The `h5_maker_qm9.py` script fetches the QM9 dataset from PyTorch Geometric, filters molecules by heavy-atom count, computes pairwise geometric edge features (bond type, polar angle $\theta$, azimuthal angle $\phi$, interatomic distance), and writes train/val/test splits to separate HDF5 files.

```bash
cd qm9
python3 data_processors/h5_maker_qm9.py
```

This script must be run **exactly once** prior to training. Output HDF5 files are written to the paths configured in the script (`SAVE_ROOT`). The default split ratio is 50/10/40 (train/val/test), filtered to molecules with 5–9 heavy atoms.

**HDF5 schema per split:**

| Dataset            | Shape                        | Description                              |
|--------------------|------------------------------|------------------------------------------|
| `node_features`    | `(N, MAX_NODES, 9)`          | Atomic features + atom count scalars     |
| `edge_features`    | `(N, MAX_NODES, MAX_NODES, 4)` | Bond type, θ, φ, distance (Å)          |
| `targets`          | `(N, 19)`                    | QM9 molecular properties                 |
| `n_atoms`          | `(N, 2)`                     | Total and heavy atom counts              |

---

### Step 2: Configure the Run

Default hyperparameters are defined in `configs/defaults.py`. Any parameter can be overridden by providing a YAML file. An annotated example is provided at `configs/YAML/qm9.yaml`.

Key configuration fields:

```yaml
setup:
  run_id: "run_001"       # Identifier for output directory
  epochs: 50
  batch_size: 1           # Quantum circuit executes one sample at a time
  train_n: 4000           # Number of training samples (null = full split)
  val_n: 2000
  targets: ["gap"]        # Regression target(s) from the 19 QM9 properties

model:
  n_qubits: 10            # Number of qubits; should match or exceed heavy-atom count
  num_layers: 2           # Variational ansatz depth
  device: "lightning.qubit"
  backend: "autograd"

optimizer:
  name: "adam"
  lr: 0.01
  lr_decay: true

loss:
  name: "huber"           # Options: mae, mse, huber

paths:
  train: "/path/to/qm9_train.h5"
  val:   "/path/to/qm9_val.h5"
  test:  "/path/to/qm9_test.h5"
  model_dir: "/path/to/save/models"
```

---

### Step 3: Train

```bash
cd qm9
python3 train.py --config configs/YAML/qm9.yaml
```

Outputs are written to `<model_dir>/<run_id>/` and include:

- `config.yaml` — fully resolved run configuration
- `logs/train.log` — loguru training log
- `circuit.png` — quantum circuit diagram (drawn at epoch 0)
- `checkpoints/weights_epoch_NNN.npy` — per-epoch weight snapshots
- `trained_model/weights_best.npy` — weights at best validation MAE
- `trained_model/weights_final.npy` — weights at end of training
- `predictions.png` / `scatter_plotpredictions.png` — validation diagnostics

---

## Circuit Architecture

The variational circuit follows an **encode → entangle → variational layers → measurement** structure:

1. **Encoding** (`_encode_atomID`): Atom identity is encoded via trainable element-specific RY rotations applied through `AngleEmbedding`.
2. **Entanglement** (`_entangle`): Pairwise IsingXX, IsingYY, and IsingZZ gates are applied over bonded heavy-atom pairs, parameterised by interatomic distance and bond type.
3. **Variational layers** (`_trainable_layers`): Each layer applies a full `Rot(α, β, γ)` (ZYZ decomposition) on every qubit.
4. **Measurement** (`_trainable_measurement` + Hamiltonian): A trainable basis rotation precedes measurement of a weighted sum Hamiltonian over `read_qubits` qubits.

---

## Reference

If you use this work, please cite the 1P1Q architecture paper from which the embedding strategy is derived:

```bibtex
@article{bal2025,
  title = {One particle - one qubit: Particle physics data encoding for quantum machine learning},
  author = {Bal, Aritra and Klute, Markus and Maier, Benedikt and Oughton, Melik and Pezone, Eric and Spannowsky, Michael},
  journal = {Phys. Rev. D},
  volume = {112},
  issue = {7},
  pages = {076004},
  numpages = {9},
  year = {2025},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/l8y2-87vq},
  url = {https://link.aps.org/doi/10.1103/l8y2-87vq}
}
```