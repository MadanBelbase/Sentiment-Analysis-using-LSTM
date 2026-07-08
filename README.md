# Sentiment Analysis using LSTM 🎬🧠

A deep learning project that classifies movie reviews as **positive** or **negative** using a Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras, trained on the IMDB dataset of 50,000 labeled reviews.

## 🔍 Overview

This project demonstrates an end-to-end NLP pipeline:
- Text preprocessing and sequence padding
- Word embeddings learned from scratch
- An LSTM-based recurrent neural network for sequence classification
- Regularization via dropout to reduce overfitting
- Model evaluation, training curve visualization, and inference on custom text
- Modular, testable code structure (not a single notebook dump)

## 🏗️ Model Architecture

```
Input (word IDs, len=200)
        │
Embedding Layer (10,000 vocab → 128-dim vectors)
        │
SpatialDropout1D (0.2)
        │
LSTM (64 units, dropout=0.2, recurrent_dropout=0.2)
        │
Dense (64, ReLU)
        │
Dropout (0.3)
        │
Dense (1, Sigmoid) → Positive / Negative
```

## 📁 Project Structure

```
sentiment-analysis-lstm/
├── main.py                    # Entry point — trains, evaluates, demos the model
├── app.py                      # Streamlit web demo
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore
│
├── src/                         # Source code (importable package)
│   ├── __init__.py
│   ├── config.py               # Hyperparameters & file paths
│   ├── data_loader.py           # Data loading, padding, text encoding
│   ├── model.py                 # LSTM architecture definition
│   ├── train.py                 # Training loop with early stopping
│   ├── evaluate.py              # Evaluation + training curve plots
│   └── predict.py               # Inference on custom text
│
├── tests/                       # Unit tests (pytest)
│   └── test_model.py
│
├── notebooks/                    # Exploratory analysis / experiments (optional)
│
├── data/
│   ├── raw/                     # Original/unprocessed data (if using custom dataset)
│   └── processed/               # Cleaned/tokenized data
│
├── models/                       # Saved trained models (.h5)
│
└── outputs/
    ├── plots/                    # Training curve images
    └── metrics/                  # Saved evaluation metrics (JSON)
```

## 🚀 Getting Started

### 1. Clone and set up environment
```bash
git clone https://github.com/<your-username>/sentiment-analysis-lstm.git
cd sentiment-analysis-lstm
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run training
```bash
python main.py
```

This will:
1. Download the IMDB dataset (built into Keras)
2. Train the LSTM model for up to 10 epochs (with early stopping)
3. Evaluate on the held-out test set
4. Save the trained model to `models/`, training curve plots to `outputs/plots/`, and metrics to `outputs/metrics/`
5. Run predictions on a few sample sentences

### 3. Run the Streamlit demo
Once you've trained a model at least once (step 2 above saves it to `models/`), launch the interactive web demo:
```bash
streamlit run app.py
```
This opens a browser tab where you can type any review and get a live Positive/Negative prediction with a confidence score.

### 4. Run tests
```bash
pytest tests/
```

### 4. Example output
```
Test Accuracy: 0.87xx

'This movie was absolutely wonderful and touching' -> Positive (0.9123)
'Terrible plot, bad acting, complete waste of time' -> Negative (0.0512)
'It was an okay movie, nothing special but not bad either' -> Positive (0.6104)
```

## 📊 Results

| Metric          | Score   |
|-----------------|---------|
| Test Accuracy   | ~87–89% |
| Test Loss       | ~0.30   |

*(exact numbers vary run to run due to random initialization)*

## 🧠 Key Concepts Demonstrated

- **Word Embeddings** – dense vector representations of words learned during training
- **Recurrent Neural Networks (LSTM)** – handling sequential/contextual dependencies in text
- **Regularization** – Dropout & SpatialDropout1D to prevent overfitting
- **Binary Classification** – sigmoid output + binary cross-entropy loss
- **Early Stopping** – halting training when validation loss stops improving
- **Modular software design** – separated config, data, model, training, and inference logic
- **Unit testing** – pytest tests validating model construction

## 🔧 Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pytest
- Streamlit

## 📈 Possible Extensions

- Swap the built-in Embedding layer for pretrained **GloVe** or **Word2Vec** vectors
- Replace LSTM with **Bidirectional LSTM** or **GRU** for comparison
- Add an **attention layer** to visualize which words drove the prediction
- Deploy as a REST API (Flask/FastAPI) or simple web app (Streamlit)
- Extend to multi-class sentiment (e.g., 1–5 star ratings)
- Swap in a custom CSV dataset using `src/data_loader.py` as the integration point

## 📄 License

MIT License — free to use and modify.
