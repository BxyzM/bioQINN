"""
Filters QM9 to molecules with 7 or 8 heavy atoms, splits 50-10-40
train/val/test, and writes each split to a separate HDF5 file.

Author: Dr. Aritra Bal (ETP)
Date: March 03, 2026
"""

import pathlib
import numpy as np
import torch
import h5py
from torch.utils.data import Subset
from torch_geometric.utils import to_dense_adj
from loguru import logger

from data_handlers.qm9_dataloader import QM9DenseDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAVE_ROOT   = pathlib.Path("/ceph/abal/BIO/QM9")
QM9_ROOT    = "./data/qm9"
MAX_NODES   = 36
SEED        = 42
SPLIT_RATIO = (0.50, 0.10, 0.40)   # train, val, test

NODE_FEATURE_NAMES = np.array(
    [
        "atomic_number",
        "aromatic_flag",
        "hybridisation_scalar",
        "num_attached_hydrogens",
        "x_coord_angstrom",
        "y_coord_angstrom",
        "z_coord_angstrom",
    ],
    dtype=h5py.special_dtype(vlen=str),
)

ATOM_NUMBER_INFO = np.array(
    [
        "dim0: total atom count including hydrogens",
        "dim1: heavy atom count (atomic number > 1)",
    ],
    dtype=h5py.special_dtype(vlen=str),
)
TARGET_INFO = np.array(
    [
        "mu: dipole moment (Debye)",
        "alpha: isotropic polarizability (a0^3)",
        "homo: HOMO energy (Hartree)",
        "lumo: LUMO energy (Hartree)",
        "gap: HOMO-LUMO gap (Hartree)",
        "r2: electronic spatial extent (a0^2)",
        "zpve: zero-point vibrational energy (Hartree)",
        "U0: internal energy at 0K (Hartree)",
        "U: internal energy at 298.15K (Hartree)",
        "H: enthalpy at 298.15K (Hartree)",
        "G: free energy at 298.15K (Hartree)",
        "Cv: heat capacity at 298.15K (cal/mol/K)",
        "U0_atom: atomisation energy at 0K (Hartree)",
        "U_atom: atomisation energy at 298.15K (Hartree)",
        "H_atom: atomisation enthalpy at 298.15K (Hartree)",
        "G_atom: atomisation free energy at 298.15K (Hartree)",
        "A: rotational constant A (GHz)",
        "B: rotational constant B (GHz)",
        "C: rotational constant C (GHz)",
    ],
    dtype=h5py.special_dtype(vlen=str),
)
TARGET_IDX = np.array(
    [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
        "U0", "U", "H", "G", "Cv",
        "U0_atom", "U_atom", "H_atom", "G_atom",
        "rot_A", "rot_B", "rot_C",
    ],
    dtype=h5py.special_dtype(vlen=str),
)
# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def get_heavy_atom_filtered_indices(
    dataset: QM9DenseDataset,
    heavy_atom_counts: tuple = (7, 8),
) -> list:
    """
    Return dataset indices where heavy atom count is in heavy_atom_counts.

    Args:
        dataset          : QM9DenseDataset instance.
        heavy_atom_counts: Tuple of accepted heavy atom counts.

    Returns:
        List of integer indices into dataset.
    """
    indices = []
    for i in range(len(dataset.dataset)):
        data = dataset.dataset[i]
        n_heavy = int((data.x[:, 5] > 1).sum())
        if n_heavy in heavy_atom_counts:
            indices.append(i)
    logger.info(
        f"Filtered to {len(indices)} molecules with "
        f"{heavy_atom_counts} heavy atoms"
    )
    return indices


# ---------------------------------------------------------------------------
# Dense conversion (max_nodes overridden to MAX_NODES)
# ---------------------------------------------------------------------------
def sample_to_dense(
    data,
) -> tuple:
    """
    Convert a single PyG Data sample to dense numpy arrays.

    Args:
        data: PyG Data object for one molecule.

    Returns:
        node_feat : (MAX_NODES, 7)   float32
        edge_feat : (MAX_NODES, MAX_NODES)  float32
        targets   : (19,)  float32
        n_atoms   : (2,)   float32  [total, heavy]
    """
    n = data.x.shape[0]

    hyb      = data.x[:, 7:10]
    hyb_any  = hyb.sum(dim=-1, keepdim=True).bool()
    hyb_s    = (hyb.argmax(dim=-1, keepdim=True).float() + 1.0) * hyb_any.float()
    node_raw = torch.cat(
        [data.x[:, 5:6], data.x[:, 6:7], hyb_s, data.x[:, 10:11], data.pos],
        dim=-1,
    )                                           # (n, 7)

    node_feat = torch.zeros(MAX_NODES, 7)
    node_feat[:n] = node_raw

    bond_types = data.edge_attr.argmax(dim=-1).float() + 1
    edge_feat  = to_dense_adj(
        data.edge_index,
        edge_attr=bond_types,
        max_num_nodes=MAX_NODES,
    ).squeeze(0)                                # (MAX_NODES, MAX_NODES)

    n_heavy = int((data.x[:, 5] > 1).sum())
    n_atoms = torch.tensor([n, n_heavy], dtype=torch.float32)

    return (
        node_feat.numpy().astype(np.float32),
        edge_feat.numpy().astype(np.float32),
        data.y.squeeze(0).numpy().astype(np.float32),
        n_atoms.numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------
def write_split(
    path: pathlib.Path,
    dataset: QM9DenseDataset,
    indices: list,
) -> None:
    """
    Write one split to an HDF5 file at path.

    Datasets are written sample-by-sample into pre-allocated arrays to avoid
    RAM spikes. Gzip compression is applied to all numeric datasets.

    Args:
        path    : Full path to the output .h5 file.
        dataset : QM9DenseDataset instance.
        indices : Molecule indices for this split.
    """
    n = len(indices)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {n} samples to {path}")

    with h5py.File(path, "w") as f:
        # Pre-allocate datasets
        ds_node   = f.create_dataset("node_features",shape=(n, MAX_NODES, 7),dtype=np.float32)
        ds_edge   = f.create_dataset("edge_features",shape=(n, MAX_NODES, MAX_NODES),dtype=np.float32)
        ds_target = f.create_dataset("targets",shape=(n, 19),dtype=np.float32)
        ds_natoms = f.create_dataset("n_atoms",shape=(n, 2),dtype=np.float32)

        # String metadata
        f.create_dataset("nodeFeatureNames",data=NODE_FEATURE_NAMES)
        f.create_dataset("atomNumberInfo",data=ATOM_NUMBER_INFO)
        f.create_dataset("targetInfo", data=TARGET_INFO)
        f.create_dataset("targetIndex", data=TARGET_IDX)
        # Sample-by-sample write
        for out_idx, src_idx in enumerate(indices):
            node, edge, tgt, nat = sample_to_dense(dataset.dataset[src_idx])
            ds_node[out_idx]   = node
            ds_edge[out_idx]   = edge
            ds_target[out_idx] = tgt
            ds_natoms[out_idx] = nat

        # Split-level metadata attributes
        f.attrs["n_samples"]        = n
        f.attrs["max_nodes"]        = MAX_NODES
        f.attrs["seed"]             = SEED
        f.attrs["split_ratio"]      = str(SPLIT_RATIO)
        f.attrs["heavy_atom_filter"]= str((7, 8))

    logger.info(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = QM9DenseDataset(root=QM9_ROOT)

    # Filter to 7 and 8 heavy atom molecules
    filtered = get_heavy_atom_filtered_indices(dataset, heavy_atom_counts=(7, 8))
    total    = len(filtered)

    # Deterministic shuffle then split
    rng     = np.random.default_rng(SEED)
    shuffled = rng.permutation(filtered).tolist()

    n_train = int(SPLIT_RATIO[0] * total)
    n_val   = int(SPLIT_RATIO[1] * total)
    # test gets the remainder to avoid rounding loss
    train_idx = shuffled[:n_train]
    val_idx   = shuffled[n_train:n_train + n_val]
    test_idx  = shuffled[n_train + n_val:]

    logger.info(
        f"Split sizes | train={len(train_idx)} | "
        f"val={len(val_idx)} | test={len(test_idx)}"
    )

    write_split(SAVE_ROOT / "train" / "qm9_train.h5", dataset, train_idx)
    write_split(SAVE_ROOT / "val"   / "qm9_val.h5",   dataset, val_idx)
    write_split(SAVE_ROOT / "test"  / "qm9_test.h5",  dataset, test_idx)

    logger.info("All splits written successfully")