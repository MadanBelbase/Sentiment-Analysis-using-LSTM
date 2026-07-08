"""
Basic unit tests for the sentiment analysis pipeline.

Run with: pytest tests/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model
from src import config


def test_model_builds_without_error():
    model = build_model()
    assert model is not None


def test_model_output_shape():
    model = build_model()
    assert model.output_shape == (None, 1)


def test_model_input_length():
    model = build_model()
    input_shape = model.input_shape
    assert input_shape[1] == config.MAX_LEN


def test_config_values_are_sane():
    assert config.VOCAB_SIZE > 0
    assert config.MAX_LEN > 0
    assert 0 <= config.DROPOUT_RATE <= 1
