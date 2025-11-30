"""
Training script for sentiment analysis MLP.
"""

import os
import kagglehub
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import MLP
from dataset import SentimentDataset


def load_data():
    """
    Download and load the Sentiment140 dataset.
    
    Returns:
        pd.DataFrame: Dataset with text and target columns
    """
    print("Downloading Sentiment140 dataset...")
    path = kagglehub.dataset_download("kazanova/sentiment140")
    print(f"Dataset downloaded to: {path}")
    
    dataset = pd.read_csv(
        path + f"/{os.listdir(path)[0]}", 
        names=['target', 'ids', 'date', 'flag', 'user', 'text'], 
        encoding='latin-1'
    )
    
    return dataset


def preprocess_data(dataset, n_features=2**12, train_size=40000, val_size=4000, test_size=4000):
    """
    Preprocess text data and split into train/val/test sets.
    
    Args:
        dataset (pd.DataFrame): Raw dataset
        n_features (int): Number of features for HashingVectorizer
        train_size (int): Number of training samples
        val_size (int): Number of validation samples
        test_size (int): Number of test samples
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
    """
    print("Preprocessing data...")
    
    # Initialize vectorizer
    vectorizer = HashingVectorizer(n_features=n_features)
    
    # Extract features and targets
    X_dataset = dataset['text']
    target = dataset['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, target, random_state=42, test_size=0.2
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_train, y_train, random_state=42, test_size=0.5
    )
    
    # Limit dataset size
    X_train = X_train[:train_size]
    X_test = X_test[:test_size]
    X_val = X_val[:val_size]
    
    # Vectorize text
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    X_val = vectorizer.transform(X_val).toarray()
    
    # Normalize targets (0, 4) -> (0, 1)
    y_train = (y_train[:train_size] / 4).astype(int).values
    y_test = (y_test[:test_size] / 4).astype(int).values
    y_val = (y_val[:val_size] / 4).astype(int).values
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train)
    val_dataset = SentimentDataset(X_val, y_val)
    test_dataset = SentimentDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train.shape[1]


def train_model(train_loader, val_loader, input_dim, max_epochs=1000):
    """
    Train the MLP model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim (int): Number of input features
        max_epochs (int): Maximum number of training epochs
        
    Returns:
        pl.Trainer: Trained PyTorch Lightning trainer
    """
    print("Initializing model...")
    
    # Initialize model
    model = MLP(input_dim)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        save_top_k=2,
        monitor="val_acc",
        mode="max",
        filename="mlp-{epoch:02d}-{val_acc:.2f}"
    )
    
    early_stopping = EarlyStopping(
        monitor="val_acc",
        min_delta=0.01,
        patience=5,
        verbose=True,
        mode="max"
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        log_every_n_steps=10,
        deterministic=True,
        logger=True,
        enable_progress_bar=True
    )
    
    print("Starting training...")
    print(model)
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed! Best model saved at: {checkpoint_callback.best_model_path}")
    
    return trainer


def main():
    """Main training pipeline."""
    # Load data
    dataset = load_data()
    
    # Preprocess
    train_loader, val_loader, test_loader, input_dim = preprocess_data(dataset)
    
    # Train
    trainer = train_model(train_loader, val_loader, input_dim)
    
    print("\n" + "="*50)
    print("Training pipeline completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()