"""
Main entry point: trains, evaluates, and demos the sentiment LSTM model.

Usage:
    python main.py
"""

from src import config
from src.data_loader import load_data, get_word_index
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model, plot_history
from src.predict import predict_sentiment


def main():
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = load_data()

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training model...")
    history = train_model(model, X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving artifacts...")
    model.save(config.MODEL_PATH)
    plot_history(history)

    print("\n--- Sample Predictions ---")
    word_index = get_word_index()
    samples = [
        "This movie was absolutely wonderful and touching",
        "Terrible plot, bad acting, complete waste of time",
        "It was an okay movie, nothing special but not bad either",
    ]
    for s in samples:
        label, prob = predict_sentiment(model, word_index, s)
        print(f"'{s}' -> {label} ({prob:.4f})")


if __name__ == "__main__":
    main()
