import numpy as np
import os
import torch
import json

def load_glove_embeddings(glove_path):
    print(f"Loading GloVe from {glove_path}...")
    embeddings_index = {}
    
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    
    print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
    return embeddings_index


def build_embedding_matrix(vocab, embeddings_index, embed_dim):
    vocab_size = len(vocab)
    embedding_matrix = torch.randn(vocab_size, embed_dim)
    
    for word, idx in vocab.items():
        if word in embeddings_index:
            vector = torch.tensor(embeddings_index[word], dtype=torch.float32)
            embedding_matrix[idx] = vector
    
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    return embedding_matrix


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    print(f"Loaded vocab with size {len(vocab)}")
    return vocab


if __name__ == "__main__":
    GLOVE_PATH = "data/glove/glove.6B.100d.txt"
    VOCAB_PATH = "data/processed/vocab.json"
    EMBED_DIM = 100
    SAVE_PATH = "data/processed/embedding_matrix.pt"

    vocab = load_vocab(VOCAB_PATH)
    embeddings_index = load_glove_embeddings(GLOVE_PATH)
    embedding_matrix = build_embedding_matrix(vocab, embeddings_index, EMBED_DIM)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(embedding_matrix, SAVE_PATH)
    print(f"Saved embedding matrix to {SAVE_PATH}")