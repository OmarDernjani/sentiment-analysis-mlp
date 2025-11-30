"""
Dataset class for sentiment analysis.
"""

import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    """
    Custom PyTorch Dataset for sentiment analysis.
    
    Args:
        x (np.ndarray): Feature vectors (vectorized text)
        y (np.ndarray): Target labels (0 or 1)
    """
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (features, label) as torch tensors
        """
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )