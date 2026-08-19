"""
scratchml/data.py

A production-grade, memory-efficient Data Pipeline engine built from scratch using NumPy and Pandas.
This module mimics the PyTorch `Dataset` and `DataLoader` classes, allowing larger-than-memory
datasets to be streamed, shuffled, and batched efficiently using Python generators.

Mathematical/CS Concepts Covered:
1. Generator-based Streaming: Prevents Out-Of-Memory (OOM) errors by yielding mini-batches on-the-fly.
2. Index Shuffling: Shuffles index arrays instead of copying large datasets in memory.
3. Collation: Combines individual samples into unified mini-batch tensors.

Generally Used Library: PyTorch (torch.utils.data.Dataset, torch.utils.data.DataLoader).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Iterator, Union

class Dataset:
    """
    Abstract Base Class for datasets.
    All custom datasets should inherit from this and override __len__ and __getitem__.
    """
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SimpleDataset(Dataset):
    """
    A concrete implementation of Dataset that wraps features and targets.
    Highly educational for students to see how raw arrays map to a Dataset interface.
    """
    def __init__(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        # Line importance: Ensure internal storage uses memory-efficient, contiguous NumPy arrays.
        self.X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        self.y = y.values if isinstance(y, pd.Series) else np.array(y)

    def __len__(self) -> int:
        # Line importance: Returns total number of samples in the dataset.
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # Line importance: Retrieves a single sample-target pair.
        return self.X[idx], self.y[idx]


class DataLoader:
    """
    Dataloader that batches, shuffles, and streams samples from a Dataset.
    """
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a Python generator that yields mini-batches of data.
        """
        # Line importance: Shuffle indices instead of the actual data to save CPU cycles and RAM.
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        # Reset batch pointer
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the next mini-batch.
        """
        if self.current_idx >= len(self.dataset):
            # Line importance: Signals the end of an epoch to the training loop.
            raise StopIteration

        # Get indices for the current batch
        batch_end = min(self.current_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.current_idx:batch_end]
        
        # Line importance: Increment index pointer for the next batch request.
        self.current_idx = batch_end

        # Collate individual samples into batch tensors
        batch_x = []
        batch_y = []
        for idx in batch_indices:
            x, y = self.dataset[idx]
            batch_x.append(x)
            batch_y.append(y)

        # Line importance: Returns batched NumPy arrays ready for forward propagation.
        return np.array(batch_x), np.array(batch_y)

    def __len__(self) -> int:
        """
        Returns the total number of batches in an epoch.
        """
        return int(np.ceil(len(self.dataset) / self.batch_size))


# =====================================================================
# STUDENT EXAMPLE & VERIFICATION
# =====================================================================
if __name__ == "__main__":
    print("--- Running Dataset & DataLoader Student Example ---")
    # 1. Create a dummy dataset of 10 samples
    X_dummy = np.arange(20).reshape(10, 2)  # 10 samples, 2 features
    y_dummy = np.arange(10)                 # 10 targets

    # 2. Instantiate SimpleDataset
    dataset = SimpleDataset(X_dummy, y_dummy)
    print(f"Total dataset size: {len(dataset)}")
    print(f"Sample at index 3: {dataset[3]}")

    # 3. Instantiate DataLoader with batch size 3
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    print(f"Total number of batches: {len(dataloader)}")

    # 4. Iterate over batches
    for batch_idx, (bx, by) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  X: {bx.tolist()}")
        print(f"  y: {by.tolist()}")
