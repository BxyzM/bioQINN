"""
Minimal QM9 dataloader. Converts PyG graph samples to dense arrays on-the-fly
per sample inside __getitem__, avoiding full dataset pre-allocation.

Node features (7 columns):
    0: atomic number
    1: aromatic flag
    2: hybridisation {0=none, 1=sp, 2=sp2, 3=sp3}
    3: hydrogen count
    4-6: x, y, z coordinates (Angstrom)

Edge features: (max_nodes, max_nodes) integer matrix, bond type {0,1,2,3,4}.

Author: Dr. Aritra Bal (ETP)
Date: March 03, 2026
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_adj
from loguru import logger
from typing import Tuple, Union

try:
    import pennylane.numpy as pnp
    _PNP_AVAILABLE = True
except ImportError:
    _PNP_AVAILABLE = False

_QM9_MAX_NODES: int = 36
_BOND_OFFSET: int = 1


class QM9DenseDataset(Dataset):
    """
    Wraps PyG QM9, converting each sample to dense arrays on demand.
    No full dataset is loaded into memory.
    """

    def __init__(self, root: str = "./data/qm9") -> None:
        """
        Args:
            root: Cache directory for QM9 raw and processed files.
        """
        self.dataset = QM9(root=root)
        logger.info(f"QM9 dataset ready: {len(self.dataset)} molecules")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns dense tensors for a single molecule.

        Returns:
            node_features : (max_nodes, 7)   float32
            edge_features : (max_nodes, max_nodes)  float32
            targets       : (19,)  float32
        """
        data = self.dataset[idx]
        n = data.x.shape[0]
        N = _QM9_MAX_NODES

        # --- node features: compress + pad ---
        hyb   = data.x[:, 7:10]                                # (n, 3)
        hyb_any = hyb.sum(dim=-1, keepdim=True).bool()
        hyb_scalar = (hyb.argmax(dim=-1, keepdim=True).float() + 1.0) * hyb_any.float()

        # node_raw = torch.cat(
        #     [data.x[:, 5:6], data.x[:, 6:7], hyb_scalar, data.x[:, 10:11], data.pos],
        #     dim=-1,
        # )                                                        # (n, 7)
        node_raw = torch.cat(
        [data.x[:, 5:6], data.x[:, 6:7], hyb_scalar, data.x[:, 10:11], data.pos, data.x[:, 7:10]],
        dim=-1,
)  # (n, 10)
        node_feat = torch.zeros(N, node_raw.shape[-1], dtype=torch.float32)
        node_feat[:n] = node_raw
        # --- edge features: dense adjacency ---
        bond_types = data.edge_attr.argmax(dim=-1).float() + _BOND_OFFSET
        edge_feat = to_dense_adj(
            data.edge_index,
            edge_attr=bond_types,
            max_num_nodes=N,
        ).squeeze(0)                                            # (N, N)

        # --- targets ---
        targets = data.y.squeeze(0).float()                    # (19,)
        n_heavy = int((data.x[:, 5] > 1).sum())
        n_atoms = torch.tensor([n, n_heavy], dtype=torch.float32)  # (2,)
        # return hydrogen and total atom numbers as B,2 tensor for convenience in some models
        return node_feat, edge_feat, targets, n_atoms


def build_dataloader(
    root: str = "./data/qm9",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    convert_pnp: bool = False,
) -> DataLoader:
    """
    Build a DataLoader over QM9DenseDataset.

    Args:
        root        : QM9 cache directory.
        batch_size  : Molecules per batch.
        shuffle     : Shuffle each epoch.
        num_workers : Parallel workers. Set 0 when convert_pnp=True
                      (pnp is not picklable; see note below).
        convert_pnp : If True, batches are returned as pennylane.numpy arrays.
                      Requires num_workers=0; enforced automatically.

    Returns:
        DataLoader yielding (node_features, edge_features, targets) per batch.
        Shapes: (B, 29, 7), (B, 29, 29), (B, 19).
    """
    if convert_pnp and not _PNP_AVAILABLE:
        raise ImportError("pennylane is not installed; cannot set convert_pnp=True")

    if convert_pnp and num_workers > 0:
        logger.warning(
            "convert_pnp=True requires num_workers=0; overriding."
        )
        num_workers = 0

    collate_fn = _pnp_collate if convert_pnp else None

    loader = DataLoader(
        QM9DenseDataset(root=root),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    logger.info(
        f"DataLoader ready | batch_size={batch_size} | "
        f"batches={len(loader)} | convert_pnp={convert_pnp}"
    )
    return loader


def _pnp_collate(
    batch: list,
) -> Tuple["pnp.ndarray", "pnp.ndarray", "pnp.ndarray"]:
    """
    Collate function that stacks samples into pennylane.numpy arrays.
    Gradients are off by default; enable per-array at the call site if needed.
    """
    nodes, edges, targets, n_atoms = torch.utils.data.default_collate(batch)
    return (
        pnp.array(nodes.numpy(),   requires_grad=False),
        pnp.array(edges.numpy(),   requires_grad=False),
        pnp.array(targets.numpy(), requires_grad=False),
        pnp.array(n_atoms.numpy(), requires_grad=False),
    )


if __name__ == "__main__":
    loader = build_dataloader(
        root="./data/qm9",
        batch_size=10000,
        shuffle=False,
        num_workers=4,
        convert_pnp=False,
    )

    nodes, edges, targets, n_atoms = next(iter(loader))
    logger.info(f"nodes   : {tuple(nodes.shape)}  {nodes.dtype}")
    logger.info(f"edges   : {tuple(edges.shape)}  {edges.dtype}")
    logger.info(f"targets : {tuple(targets.shape)}  {targets.dtype}")
    logger.info(f"n_atoms : {tuple(n_atoms.shape)}  {n_atoms.dtype}")

    assert set(np.unique(edges.numpy()).tolist()).issubset({0,1,2,3,4})
    assert set(np.unique(nodes[:, :, 2].numpy()).tolist()).issubset({0,1,2,3})
    logger.info("Sanity checks passed")
    logger.info("Press q to quit, or continue with interactive session...")
    import pdb; pdb.set_trace()