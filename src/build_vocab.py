import json
import pandas as pd
from collections import Counter
import os

def build_vocab_from_dataset(processed_csv, min_freq=2):
    print(f"Loading processed dataset from {processed_csv}...")
    df = pd.read_csv(processed_csv)

    counter = Counter()
    for text in df["clean_review"]:
        tokens = text.split()
        counter.update(tokens)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    print(f"Vocabulary size (including special tokens): {len(vocab)}")
    return vocab


def save_vocab(vocab, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to {save_path}")


if __name__ == "__main__":
    PROCESSED_CSV = "data/processed/imdb_clean.csv"
    VOCAB_PATH = "data/processed/vocab.json"

    vocab = build_vocab_from_dataset(PROCESSED_CSV, min_freq=2)
    save_vocab(vocab, VOCAB_PATH)
