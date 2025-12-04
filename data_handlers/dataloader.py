"""
Minimal PyTorch DataLoader for loading protein-complex image data from PISToN framework.
Loads all .npy files (excluding resname files) from specified paths and creates batched tensors.

Author: Dr. Aritra Bal (ETP)
Date: 27 November 2025
"""

from pathlib import Path
from typing import Union, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class ProteinComplexDataset(Dataset):
    """
    Dataset for loading protein-complex interaction images from .npy files.
    Pre-loads all data into memory for maximum speed.
    """
    
    def __init__(
        self, 
        paths: Union[List[str], Dict[str, str]],
        exclude_pattern: str = "resname"
    ):
        """
        Initialize dataset by loading all .npy files from specified paths.
        
        Args:
            paths: List of directory paths or dict mapping interaction names to paths
            exclude_pattern: String pattern to exclude files (default: "resname")
        """
        self.paths = paths
        self.exclude_pattern = exclude_pattern
        self.interaction_names = []
        self.data = None
        
        self._load_all_data()
    
    def _load_all_data(self) -> None:
        """Load all .npy files from paths into a single tensor."""
        path_dict = self._normalize_paths()
        
        all_arrays = []
        for interaction_name, path_str in path_dict.items():
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"Path does not exist: {path_str}")
                continue
            
            # Get all .npy files excluding those with exclude_pattern
            npy_files = [
                f for f in path.glob("*.npy") 
                if self.exclude_pattern not in f.name
            ]
            
            if not npy_files:
                logger.warning(f"No valid .npy files found in {path_str}")
                continue
            
            # Load all files from this path
            for npy_file in npy_files:
                arr = np.load(npy_file)
                all_arrays.append(arr)
                self.interaction_names.append(interaction_name)
        
        if not all_arrays:
            logger.error("No data loaded from any path")
            raise ValueError("No valid .npy files found in provided paths")
        
        # Stack all arrays into single tensor: (num_files, N, N, C)
        self.data = torch.from_numpy(np.stack(all_arrays, axis=0)).float()
        logger.info(f"Loaded {len(all_arrays)} files with shape {self.data.shape}")
    
    def _normalize_paths(self) -> Dict[str, str]:
        """Convert paths to dict format for uniform handling."""
        
        for path in self.paths if isinstance(self.paths, list) else self.paths.values():
            if Path(path).suffix == '.csv':
                raise ValueError("CSV files are not supported in this dataloader.")
        if isinstance(self.paths, dict):
            return self.paths
        elif isinstance(self.paths, list):
            return {f"path_{i}": p for i, p in enumerate(self.paths)}
        else:
            raise TypeError("paths must be list or dict")
        # check if at least one path points to a csv file
        
            
            
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tensor of shape (N, N, C)
        """
        return self.data[idx]


def create_dataloader(
    paths: Union[List[str], Dict[str, str]],
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for protein-complex dataset.
    
    Args:
        paths: List of directory paths or dict mapping interaction names to paths
        batch_size: Batch size for loading
        shuffle: Whether to shuffle data
        num_workers: Number of workers for parallel loading
        
    Returns:
        DataLoader yielding batches of shape (batch_size, N, N, C)
    """
    dataset = ProteinComplexDataset(paths)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )


# Testing block
if __name__ == "__main__":
    import tempfile
    import shutil
    import argparse
    
    parser = argparse.ArgumentParser(description="Test protein complex dataloader")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to directory containing .npy files. If not provided, creates dummy data."
    )
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size for testing")
    args = parser.parse_args()
    
    logger.info("Starting dataloader test")
    
    if args.path is None:
        # Create temporary directory with dummy data
        temp_base = Path("/tmp/abal")
        temp_base.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(dir=temp_base))
        cleanup_required = True
        
        # Create dummy .npy files
        N, C = 64, 5
        num_files = 10
        
        for i in range(num_files):
            dummy_data = np.random.randn(N, N, C).astype(np.float32)
            np.save(temp_dir / f"sample_{i}.npy", dummy_data)
        
        # Create resname file (should be excluded)
        dummy_resname = np.random.randn(N, N, C).astype(np.float32)
        np.save(temp_dir / "resname_sample.npy", dummy_resname)
        
        logger.info(f"Created {num_files} dummy files in {temp_dir}")
        test_path = str(temp_dir)
    else:
        test_path = args.path
        cleanup_required = False
        logger.info(f"Using provided path: {test_path}")
    
    try:
        # Test with list of paths
        dataloader = create_dataloader(
            paths=[test_path],
            batch_size=args.batch_size,
            shuffle=False
        )
        
        logger.info(f"Dataset size: {len(dataloader.dataset)}")
        logger.info(f"Number of batches: {len(dataloader)}")
        
        # Iterate and print batch shapes
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Batch {batch_idx}: shape {batch.shape}, dtype {batch.dtype}")
            #if batch_idx == 2:
            #    break
        
        # Test with dict of paths
        dataloader_dict = create_dataloader(
            paths={"test_interaction": test_path},
            batch_size=args.batch_size,
            shuffle=False
        )
        
        first_batch = next(iter(dataloader_dict))
        logger.info(f"Single batch shape: {first_batch.shape}")
        
        logger.info("Dataloader test completed successfully")
    except Exception as e:
        logger.error(f"Error during dataloader test: {e}")    
    finally:
        # Cleanup only if dummy data was created
        if cleanup_required:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary directory")