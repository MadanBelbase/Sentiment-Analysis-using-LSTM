"""
Central configuration for the sentiment analysis project.
Keeping all hyperparameters and paths here (instead of scattered
across files) makes the project easy to tune and reproduce.
"""

import os
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")

MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_lstm_model.h5")
TRAINING_CURVES_PATH = os.path.join(PLOTS_DIR, "training_curves.png")
METRICS_PATH = os.path.join(METRICS_DIR, "metrics.json")

# Hyperparameters
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
SPATIAL_DROPOUT_RATE = 0.2
RECURRENT_DROPOUT_RATE = 0.2

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 2
