import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

class IMDBDataset(Dataset):
    def __init__(self, csv_path, vocab_path, max_len=200):
        self.df = pd.read_csv(csv_path)
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        
        self.max_len = max_len
        self.pad_idx = self.vocab.get(PAD_TOKEN, 0)
        self.unk_idx = self.vocab.get(UNK_TOKEN, 1)

        self.label_map = {"positive": 1, "negative": 0}

    def __len__(self):
        return len(self.df)

    def encode_text(self, text):
        tokens = text.split()
        ids = [self.vocab.get(token, self.unk_idx) for token in tokens]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_tensor = self.encode_text(row["clean_review"])
        label = self.label_map[row["sentiment"]]
        return text_tensor, torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_texts, labels


def get_dataloaders(csv_path, vocab_path, batch_size=32, max_len=200, val_split=0.1, test_split=0.1, shuffle=True):
    dataset = IMDBDataset(csv_path, vocab_path, max_len=max_len)
    n = len(dataset)
    
    val_size = int(n * val_split)
    test_size = int(n * test_split)
    train_size = n - val_size - test_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    CSV_PATH = "data/processed/imdb_clean.csv"
    VOCAB_PATH = "data/processed/vocab.json"
    BATCH_SIZE = 64
    MAX_LEN = 200

    train_loader, val_loader, test_loader = get_dataloaders(CSV_PATH, VOCAB_PATH, BATCH_SIZE, MAX_LEN)

    for batch_texts, batch_labels in train_loader:
        print(batch_texts.shape)
        print(batch_labels.shape)
        break
