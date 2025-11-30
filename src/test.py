"""
Testing script for trained sentiment analysis model.
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import numpy as np

from model import MLP
from dataset import SentimentDataset


def load_checkpoint(checkpoint_path, input_dim):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        input_dim (int): Number of input features
        
    Returns:
        MLP: Loaded model
    """
    model = MLP.load_from_checkpoint(checkpoint_path, input_dim=input_dim)
    model.eval()
    return model


def test_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Test the model and compute metrics.
    
    Args:
        model: Trained MLP model
        test_loader: Test data loader
        device (str): Device to run on
        
    Returns:
        tuple: (predictions, true_labels, probabilities)
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Get predictions
            probs = model(batch_x)
            preds = torch.round(probs)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_prediction_distribution(probs, save_path='results/prediction_distribution.png'):
    """
    Plot distribution of prediction probabilities.
    
    Args:
        probs: Prediction probabilities
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction distribution saved to {save_path}")
    plt.close()


def save_classification_report(y_true, y_pred, save_path='results/classification_report.txt'):
    """
    Save classification report to file.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str): Path to save the report
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=['Negative', 'Positive'],
                                   digits=4)
    
    os.makedirs('results', exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("SENTIMENT ANALYSIS - CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    
    print(f"Classification report saved to {save_path}")
    print("\n" + report)


def predict_sentiment(model, text, vectorizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predict sentiment for a single text.
    
    Args:
        model: Trained model
        text (str): Input text
        vectorizer: HashingVectorizer fitted on training data
        device (str): Device to run on
        
    Returns:
        tuple: (prediction, probability)
    """
    model = model.to(device)
    model.eval()
    
    # Vectorize text
    X = vectorizer.transform([text]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prob = model(X_tensor).item()
        pred = 1 if prob > 0.5 else 0
    
    sentiment = "Positive" if pred == 1 else "Negative"
    return sentiment, prob


def main():
    """Main testing pipeline."""
    print("=" * 50)
    print("SENTIMENT ANALYSIS MODEL TESTING")
    print("=" * 50 + "\n")
    
    # Configuration
    CHECKPOINT_PATH = "checkpoints/best_model.ckpt"  # Modify with your checkpoint
    INPUT_DIM = 4096
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please train the model first or provide the correct checkpoint path.")
        return
    
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = load_checkpoint(CHECKPOINT_PATH, INPUT_DIM)
    print("Model loaded successfully!\n")
    
    # Note: You need to load test data here
    # This is a placeholder - adapt based on your data loading
    print("Note: Please load your test data using the same preprocessing as training.")
    print("Then call test_model(model, test_loader) to evaluate.\n")
    
    # Example usage (uncomment when you have test_loader):
    # preds, labels, probs = test_model(model, test_loader)
    # plot_confusion_matrix(labels, preds)
    # plot_prediction_distribution(probs)
    # save_classification_report(labels, preds)
    
    print("\nTesting functions available:")
    print("- test_model(model, test_loader)")
    print("- plot_confusion_matrix(y_true, y_pred)")
    print("- plot_prediction_distribution(probs)")
    print("- save_classification_report(y_true, y_pred)")
    print("- predict_sentiment(model, text, vectorizer)")


if __name__ == "__main__":
    main()