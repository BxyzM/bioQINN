"""
PyTorch DataLoader for protein-complex data from PISToN framework with CAPRI labels.

Loads docked protein configurations, extracts CAPRI labels and DockQ scores from irmsd.csv,
applies negative sampling for class balance up to a maximum ratio (1:5 in OG PISToN paper), and creates train/validation splits.

Author: Dr. Aritra Bal (ETP)
Date: December 04, 2025
"""

import os
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from loguru import logger
from pennylane import numpy as pnp

from bioqinn.data_handlers.proteins import CAPRI_label, DockQ_score


class ProteinComplexDataset(Dataset):
    """
    Dataset for protein-complex interaction classification with CAPRI labels.
    
    Loads docked configurations from multiple PPIs, applies negative sampling
    for class balance, and provides data, labels, scores, and PPI tags.
    """
    
    def __init__(
        self, 
        cfg,
        data: Optional[pnp.ndarray] = None,
        labels: Optional[pnp.ndarray] = None,
        scores: Optional[pnp.ndarray] = None,
        ppi_tags: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset either by loading from config or from pre-split arrays.
        
        Args:
            cfg: Configuration object
            data: Pre-loaded data array (for train/val splits)
            labels: Pre-loaded label array
            scores: Pre-loaded score array
            ppi_tags: Pre-loaded PPI tag array
        """
        self.cfg = cfg
        
        if data is not None:
            # Use pre-loaded data (for train/val splits)
            self.data = data
            self.labels = labels
            self.scores = scores
            self.ppi_tags = ppi_tags
            logger.info(f"Dataset initialized with {len(self.data)} pre-loaded samples")
        else:
            # Load and process data from scratch
            self.data = None
            self.labels = None
            self.scores = None
            self.ppi_tags = None
            self._load_all_data()
    
    def _load_single_ppi(
        self,
        ppi_name: str,
        ppi_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data for a single protein-protein interaction.
        
        Args:
            ppi_name: Name of the PPI (e.g., '1TM1_E_I')
            ppi_path: Base path for this PPI's docked directory
            
        Returns:
            Tuple of (data_array, labels_array, scores_array)
        """
        # Construct path to grid maps
        grid_path = Path(os.path.join(ppi_path, '07-grid-maps'))
        
        if not grid_path.exists():
            logger.warning(f"Grid maps directory not found: {grid_path}")
            return None, None, None
        
        # Get all .npy files excluding pattern
        exclude_pattern = self.cfg.data.get('exclude_pattern', 'resname')
        npy_files = sorted([
            f for f in grid_path.glob("*.npy")
            if exclude_pattern not in f.name
        ])
        
        if not npy_files:
            logger.warning(f"No valid .npy files found in {grid_path}")
            return None, None, None
        
        # Load CSV file with labels and scores
        csv_path = Path(os.path.join(ppi_path, 'irmsd.csv'))
        if not csv_path.exists():
            logger.error(f"irmsd.csv not found: {csv_path}")
            return None, None, None
        
        df = pd.read_csv(csv_path)
        
        # Extract CAPRI labels and DockQ scores
        capri_labels = CAPRI_label(df)
        dockq_scores = DockQ_score(df)
        
        # Load all npy files
        data_arrays = []
        for npy_file in npy_files:
            arr = np.load(npy_file)
            data_arrays.append(arr)
        
        # Stack into single array
        data_array = np.stack(data_arrays, axis=0)
        
        # Verify dimensions match
        n_configs = len(data_arrays)
        assert data_array.shape[0] == len(capri_labels), \
            f"Data configs ({data_array.shape[0]}) != labels ({len(capri_labels)}) for {ppi_name}"
        assert data_array.shape[0] == len(dockq_scores), \
            f"Data configs ({data_array.shape[0]}) != scores ({len(dockq_scores)}) for {ppi_name}"
        
        n_positive = np.sum(capri_labels == 1.0)
        logger.info(
            f"PPI {ppi_name}: Loaded {n_configs} configs | "
            f"Positive: {n_positive} ({100*n_positive/n_configs:.1f}%) | "
            f"Shape: {data_array.shape}"
        )
        
        return data_array, capri_labels, dockq_scores
    
    def _apply_negative_sampling(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        ppi_name: str,
        seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply negative sampling to balance positive and negative classes.
        
        Args:
            data: Data array for single PPI
            labels: Label array
            scores: Score array
            ppi_name: Name of PPI (for logging)
            seed: Random seed for sampling
            
        Returns:
            Tuple of balanced (data, labels, scores)
        """
        # Create masks for positive and negative samples
        pos_mask = labels == 1.0
        neg_mask = labels == 0.0
        
        n_positive = np.sum(pos_mask)
        n_negative = np.sum(neg_mask)
        
        if n_positive == 0:
            logger.warning(f"PPI {ppi_name} has no positive samples, skipping")
            return None, None, None
        
        # Determine number of negatives to sample
        neg_pos_ratio = self.cfg.data.get('neg_pos_ratio', 1)
        n_neg_to_sample = int(neg_pos_ratio * n_positive)
        
        if n_negative < n_neg_to_sample:
            logger.warning(
                f"PPI {ppi_name}: Requested {n_neg_to_sample} negatives but only "
                f"{n_negative} available. Using all negatives."
            )
            n_neg_to_sample = n_negative
        
        # Get positive samples
        pos_data = data[pos_mask]
        pos_labels = labels[pos_mask]
        pos_scores = scores[pos_mask]
        
        # Randomly sample negative samples
        np.random.seed(seed)
        neg_indices = np.where(neg_mask)[0]
        sampled_neg_indices = np.random.choice(
            neg_indices,
            size=n_neg_to_sample,
            replace=False
        )
        
        neg_data = data[sampled_neg_indices]
        neg_labels = labels[sampled_neg_indices]
        neg_scores = scores[sampled_neg_indices]
        
        # Combine positive and negative samples
        balanced_data = np.concatenate([pos_data, neg_data], axis=0)
        balanced_labels = np.concatenate([pos_labels, neg_labels], axis=0)
        balanced_scores = np.concatenate([pos_scores, neg_scores], axis=0)
        
        logger.info(
            f"PPI {ppi_name}: Balanced dataset created | "
            f"Positive: {n_positive} | Negative: {n_neg_to_sample} | "
            f"Total: {len(balanced_data)}"
        )
        
        return balanced_data, balanced_labels, balanced_scores
    
    def _load_all_data(self) -> None:
        """Load and process data from all PPIs specified in config."""
        protein_pairs = self.cfg.data.protein_pairs
        docked_path_template = self.cfg.directories.docked_path
        seed = self.cfg.reproducibility.get('seed_value', 42)
        
        logger.info(f"Loading data for {len(protein_pairs)} protein pairs")
        
        all_data = []
        all_labels = []
        all_scores = []
        all_ppi_tags = []
        
        for ppi_name in protein_pairs:
            # Construct path for this PPI
            ppi_path = docked_path_template.replace('PROTEIN_NAME', ppi_name)
            
            logger.info(f"Processing PPI: {ppi_name}")
            logger.info(f"Path: {ppi_path}")
            
            # Load single PPI data
            data, labels, scores = self._load_single_ppi(ppi_name, ppi_path)
            
            if data is None:
                logger.warning(f"Skipping PPI {ppi_name} due to loading errors")
                continue
            
            # Apply negative sampling for class balance
            balanced_data, balanced_labels, balanced_scores = self._apply_negative_sampling(
                data, labels, scores, ppi_name, seed
            )
            
            if balanced_data is None:
                logger.warning(f"Skipping PPI {ppi_name} due to sampling errors")
                continue
            
            # Create PPI tags for this batch
            ppi_tags = np.array([ppi_name] * len(balanced_data))
            
            # Accumulate
            all_data.append(balanced_data)
            all_labels.append(balanced_labels)
            all_scores.append(balanced_scores)
            all_ppi_tags.append(ppi_tags)
        
        if not all_data:
            logger.error("No data loaded from any PPI")
            raise ValueError("No valid data found in provided protein pairs")
        
        # Concatenate all PPIs
        self.data = pnp.array(np.concatenate(all_data, axis=0), requires_grad=True)
        self.labels = pnp.array(np.concatenate(all_labels, axis=0), requires_grad=True)
        self.scores = pnp.array(np.concatenate(all_scores, axis=0), requires_grad=True)
        self.ppi_tags = np.concatenate(all_ppi_tags, axis=0)
        
        logger.info(f"Global dataset created: {self.data.shape}")
        logger.info(f"Total samples: {len(self.data)}")
        logger.info(f"Positive samples: {np.sum(self.labels == 1.0)}")
        logger.info(f"Negative samples: {np.sum(self.labels == 0.0)}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get single sample with label, score, and PPI tag.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (data, label, score, ppi_tag)
        """
        return (
            self.data[idx],
            self.labels[idx],
            self.scores[idx],
            self.ppi_tags[idx]
        )


def create_dataloader(
    cfg,
    train: bool = True,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Create DataLoader(s) for protein-complex dataset.
    
    Args:
        cfg: Configuration object
        train: If True, split data and return train/val loaders. If False, return single loader.
        batch_size: Batch size (overrides cfg if provided)
        shuffle: Whether to shuffle (overrides cfg if provided)
        num_workers: Number of workers (overrides cfg if provided)
        
    Returns:
        If train=True: Tuple of (train_loader, val_loader)
        If train=False: Single DataLoader
    """
    # Get dataloader parameters from config or arguments
    batch_size = batch_size or cfg.training.batch_size
    num_workers = num_workers or cfg.data.get('num_workers', 0)
    
    if train:
        # Load full dataset
        logger.info("Creating training and validation dataloaders")
        full_dataset = ProteinComplexDataset(cfg)
        
        # Get train/val split ratio
        train_split = cfg.data.get('train_val_split', 0.8)
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        logger.info(f"Splitting dataset: train={train_size}, val={val_size}")
        
        # Create indices for splitting
        seed = cfg.reproducibility.get('seed_value', 42)
        np.random.seed(seed)
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split data, labels, scores, and tags
        train_data = full_dataset.data[train_indices]
        train_labels = full_dataset.labels[train_indices]
        train_scores = full_dataset.scores[train_indices]
        train_tags = full_dataset.ppi_tags[train_indices]
        
        val_data = full_dataset.data[val_indices]
        val_labels = full_dataset.labels[val_indices]
        val_scores = full_dataset.scores[val_indices]
        val_tags = full_dataset.ppi_tags[val_indices]
        
        # Create separate datasets
        train_dataset = ProteinComplexDataset(
            cfg,
            data=train_data,
            labels=train_labels,
            scores=train_scores,
            ppi_tags=train_tags
        )
        
        val_dataset = ProteinComplexDataset(
            cfg,
            data=val_data,
            labels=val_labels,
            scores=val_scores,
            ppi_tags=val_tags
        )
        
        # Create dataloaders
        train_shuffle = shuffle if shuffle is not None else cfg.data.get('shuffle', True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        logger.info(f"Train loader: {len(train_loader)} batches")
        logger.info(f"Val loader: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    else:
        # Create single dataloader (e.g., for inference)
        logger.info("Creating single dataloader")
        dataset = ProteinComplexDataset(cfg)
        
        shuffle = shuffle if shuffle is not None else cfg.data.get('shuffle', False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        logger.info(f"Dataloader: {len(dataloader)} batches")
        return dataloader


# Testing block
if __name__ == "__main__":
    import argparse
    import yaml
    from configs.config import Config
    
    parser = argparse.ArgumentParser(description="Test protein complex dataloader")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--test-split",
        action='store_true',
        help="Test train/val split functionality"
    )
    args = parser.parse_args()
    
    logger.info("Starting dataloader test")
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    cfg = Config(config_dict)
    
    try:
        if args.test_split:
            # Test train/val split
            train_loader, val_loader = create_dataloader(cfg, train=True)
            
            logger.info("Testing train loader:")
            for batch_idx, (data, labels, scores, tags) in enumerate(train_loader):
                logger.info(
                    f"Train batch {batch_idx}: data shape={data.shape}, "
                    f"labels shape={labels.shape}, "
                    f"scores shape={scores.shape}, "
                    f"tags sample={tags[0]}"
                )
                if batch_idx == 2:
                    break
            
            logger.info("Testing val loader:")
            for batch_idx, (data, labels, scores, tags) in enumerate(val_loader):
                logger.info(
                    f"Val batch {batch_idx}: data shape={data.shape}, "
                    f"labels shape={labels.shape}, "
                    f"scores shape={scores.shape}, "
                    f"tags sample={tags[0]}"
                )
                if batch_idx == 2:
                    break
        else:
            # Test single loader
            dataloader = create_dataloader(cfg, train=False)
            
            for batch_idx, (data, labels, scores, tags) in enumerate(dataloader):
                logger.info(
                    f"Batch {batch_idx}: data shape={data.shape}, "
                    f"labels shape={labels.shape}, "
                    f"scores shape={scores.shape}, "
                    f"tags sample={tags[0]}"
                )
                if batch_idx == 2:
                    break
        
        logger.info("Dataloader test completed successfully")
    
    except Exception as e:
        logger.error(f"Error during dataloader test: {e}")
        raise