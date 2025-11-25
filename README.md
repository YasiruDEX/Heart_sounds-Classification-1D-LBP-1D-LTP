# Heart Sound Classification

Deep learning-based classification of heart sounds using the PASCAL dataset.

## Overview

This project implements a CNN-based approach for classifying heart sounds into four categories:
- **Normal**: Normal heart sounds
- **Murmur**: Heart sounds with murmurs (abnormal sounds)
- **Artifact**: Heart sounds with artifacts (noise)
- **Extrahls**: Heart sounds with extra heart sounds

## Dataset

The PASCAL Heart Sound Challenge dataset is used. The dataset contains `.wav` audio files organized into folders by class:
- `Atraining_normal/` - Normal heart sounds
- `Atraining_murmur/` - Heart sounds with murmurs
- `Atraining_artifact/` - Heart sounds with artifacts
- `Atraining_extrahls/` - Heart sounds with extra sounds

## Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Heart_sounds-2/
├── PASCAL/                    # Dataset directory
│   ├── Atraining_normal/
│   ├── Atraining_murmur/
│   ├── Atraining_artifact/
│   └── Atraining_extrahls/
├── models/                    # Saved models and results
│   ├── best_model.keras
│   ├── final_model.keras
│   └── results/
│       ├── training_history.png
│       ├── confusion_matrix.png
│       └── roc_curves.png
├── config.py                  # Configuration parameters
├── data_preprocessing.py      # Data loading and preprocessing
├── model.py                   # Model architectures
├── train.py                   # Training script
├── predict.py                 # Inference script
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### Training

```bash
# Train with default settings (CNN + Attention model)
python main.py

# Train with specific model type
python main.py --model cnn_attention  # CNN with attention mechanism
python main.py --model cnn            # Basic CNN
python main.py --model resnet         # ResNet-style model

# Custom training parameters
python main.py --model cnn_attention --epochs 50 --batch-size 32

# Disable data augmentation
python main.py --no-augment
```

### Making Predictions

```bash
# Predict on a single audio file
python predict.py path/to/audio.wav

# Specify custom model path
python predict.py audio.wav --model-path models/best_model.keras
```

## Model Architecture

### CNN with Attention (Default)

The default model uses a CNN architecture with attention mechanisms:

1. **Input**: Mel spectrogram (128 x 157 x 1)
2. **Convolutional Blocks**: 4 blocks with increasing filters (32 → 64 → 128 → 256)
3. **Attention Mechanism**: Channel attention at each block
4. **Global Average Pooling**: Reduces spatial dimensions
5. **Dense Layers**: 256 → 128 → 4 (softmax)

Features:
- Batch Normalization for stable training
- Dropout for regularization
- L2 weight regularization
- Residual connections (ResNet variant)

## Data Preprocessing

1. **Audio Loading**: Resample to 4000 Hz, pad/truncate to 5 seconds
2. **Mel Spectrogram**: Convert audio to Mel spectrogram representation
3. **Normalization**: Scale values to [0, 1]
4. **Data Augmentation**:
   - Time stretching
   - Pitch shifting
   - Noise addition
   - Volume adjustment

## Results

After training, results are saved in `models/results/`:
- `training_history.png`: Accuracy, loss, precision, recall curves
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curves.png`: ROC curves for each class

## Configuration

Key parameters in `config.py`:
- `SAMPLE_RATE`: Audio sample rate (4000 Hz)
- `DURATION`: Audio duration (5 seconds)
- `N_MELS`: Number of Mel bands (128)
- `EPOCHS`: Training epochs (100)
- `BATCH_SIZE`: Batch size (16)
- `LEARNING_RATE`: Initial learning rate (0.001)

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- librosa
- NumPy
- scikit-learn
- matplotlib
- seaborn

## License

This project is for educational and research purposes.

## References

- PASCAL Heart Sound Challenge Dataset
- Heart Sound Classification using Deep Learning
