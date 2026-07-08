"""
Model evaluation and training-curve visualization.
"""

import json
import matplotlib.pyplot as plt

from src import config


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the held-out test set and save metrics to disk."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    with open(config.METRICS_PATH, "w") as f:
        json.dump({"test_loss": loss, "test_accuracy": accuracy}, f, indent=2)

    return loss, accuracy


def plot_history(history, out_path=None):
    """Plot and save training/validation accuracy and loss curves."""
    out_path = out_path or config.TRAINING_CURVES_PATH
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")
