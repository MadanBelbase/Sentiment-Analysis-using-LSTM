"""
Data loading and preprocessing utilities.
"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src import config


def load_data():
    """Load the IMDB dataset and pad sequences to a fixed length.

    Returns:
        (X_train, y_train), (X_test, y_test): padded integer sequences and labels
    """
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config.VOCAB_SIZE)

    X_train = pad_sequences(
        X_train, maxlen=config.MAX_LEN, padding="post", truncating="post"
    )
    X_test = pad_sequences(
        X_test, maxlen=config.MAX_LEN, padding="post", truncating="post"
    )
    return (X_train, y_train), (X_test, y_test)


def get_word_index():
    """Return the word -> integer index mapping used by the IMDB dataset."""
    return imdb.get_word_index()


def encode_review(text, word_index, vocab_size=config.VOCAB_SIZE, max_len=config.MAX_LEN):
    """Convert a raw text string into the padded integer format the model expects."""
    words = text.lower().split()
    encoded = [word_index.get(word, 0) + 3 for word in words]
    encoded = [i if i < vocab_size else 2 for i in encoded]
    return pad_sequences([encoded], maxlen=max_len, padding="post", truncating="post")
