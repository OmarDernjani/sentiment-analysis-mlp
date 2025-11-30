# Sentiment Analysis with MLP

Binary sentiment classification on the Sentiment140 dataset using a Multi-Layer Perceptron built with PyTorch Lightning.

## Dataset
- **Sentiment140**: 1.6M tweets annotated for sentiment (0 = negative, 4 = positive)
- Features extracted using HashingVectorizer (4096 features)

## Model Architecture
- Input: 4096 features
- Hidden layers: 256 → 128 → 64 neurons
- Dropout: 0.3
- Output: Binary classification (Sigmoid)

## Requirements
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/train.py
```
## Results
- Training samples: 40,000
- Validation samples: 4,000
- Test samples: 4,000

## Project Structure
```
├── notebooks/      # Jupyter notebooks
├── src/           # Source code
├── data/          # Dataset storage
├── checkpoints/   # Model checkpoints
└── results/       # Training results
```

## License
MIT
