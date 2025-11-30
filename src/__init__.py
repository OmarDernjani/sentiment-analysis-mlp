"""
Sentiment Analysis MLP
A PyTorch Lightning implementation for binary sentiment classification.

Author: Omar Dernjani
License: MIT
"""

from .model import MLP
from .dataset import SentimentDataset

__version__ = "1.0.0"
__all__ = ["MLP", "SentimentDataset"]