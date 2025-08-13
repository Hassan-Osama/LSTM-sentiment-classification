from src.data_preprocessing import clean_text
from src.dataset import IMDBDataset
import json
import torch
from src.model import LSTMSentimentClassifier

vocab_path = "data/processed/vocab.json"
embedding_matrix_path = "data/processed/embedding_matrix.pt"
UNK_TOKEN = "<UNK>"
MODEL_PATH = "models/lstm_glove.pth"


def encode_text(text, vocab):
        tokens = text.split()
        ids = [vocab.get(token, vocab.get(UNK_TOKEN, 1)) for token in tokens]
        if len(ids) > 200:
            ids = ids[:200]
        for i in range(len(ids),  200):
             ids.append(0)
        return torch.tensor(ids, dtype=torch.long)

def get_vocab():
     with open(vocab_path, "r") as f:
        return json.load(f)


def predict_sentiment(text):
    text = clean_text(text)
    vocab = get_vocab()
    text = encode_text(text=text,vocab=vocab)
    embedding_matrix = torch.load(embedding_matrix_path)
    model = LSTMSentimentClassifier(embedding_matrix=embedding_matrix)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        out = model(text)
    return out.float().item()*100, "Positive" if out>0.5 else "Negative"


