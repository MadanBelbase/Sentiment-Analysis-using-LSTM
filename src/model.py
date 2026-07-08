"""
LSTM model architecture for sentiment classification.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D

from src import config


def build_model():
    """Construct and compile the LSTM sentiment classifier."""
    model = Sequential([
        Embedding(
            input_dim=config.VOCAB_SIZE,
            output_dim=config.EMBEDDING_DIM,
            input_length=config.MAX_LEN,
        ),
        SpatialDropout1D(config.SPATIAL_DROPOUT_RATE),
        LSTM(
            config.LSTM_UNITS,
            dropout=config.DROPOUT_RATE - 0.1,
            recurrent_dropout=config.RECURRENT_DROPOUT_RATE,
        ),
        Dense(config.DENSE_UNITS, activation="relu"),
        Dropout(config.DROPOUT_RATE),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model
