"""
Training routine with early stopping.
"""

from tensorflow.keras.callbacks import EarlyStopping

from src import config


def train_model(model, X_train, y_train):
    """Train the model with early stopping on validation loss."""
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1,
    )
    return history
