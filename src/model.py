"""
MLP model for sentiment analysis using PyTorch Lightning.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import accuracy, precision, recall


class MLP(pl.LightningModule):
    """
    Multi-Layer Perceptron for binary sentiment classification.
    
    Architecture:
        - Input layer: input_dim features
        - Hidden layers: 256 → 128 → 64 neurons
        - Output layer: 1 neuron (sigmoid activation)
        - Dropout: 0.3 after each hidden layer
    
    Args:
        input_dim (int): Number of input features
    """
    
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x).view(-1)
    
    def configure_optimizers(self):
        """Configure Adam optimizer with learning rate 0.01."""
        return optim.Adam(self.parameters(), lr=0.01)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.
        
        Args:
            batch: Tuple of (features, labels)
            batch_idx: Index of the batch
            
        Returns:
            loss: Training loss for the batch
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.
        
        Args:
            batch: Tuple of (features, labels)
            batch_idx: Index of the batch
            
        Returns:
            loss: Validation loss for the batch
        """
        x, y = batch
        y_hat = self(x)
        pred = torch.round(y_hat)
        loss = nn.BCELoss()(y_hat, y.float())
        
        # Calculate metrics
        acc = accuracy(pred, y, task='binary')
        prec = precision(pred, y, task='binary')
        rec = recall(pred, y, task='binary')
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_prec', prec)
        self.log('val_rec', rec)
        
        return loss