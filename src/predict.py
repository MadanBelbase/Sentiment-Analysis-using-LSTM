"""
Inference utilities for running the trained model on new, unseen text.
"""

from src.data_loader import encode_review


def predict_sentiment(model, word_index, text):
    """Predict sentiment (Positive/Negative) for a raw text string.

    Returns:
        (label, probability)
    """
    encoded = encode_review(text, word_index)
    prob = model.predict(encoded, verbose=0)[0][0]
    label = "Positive" if prob > 0.5 else "Negative"
    return label, float(prob)
