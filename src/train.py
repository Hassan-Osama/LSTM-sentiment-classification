import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from dataset import get_dataloaders
from model import LSTMSentimentClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    losses = []
    all_preds = []
    all_labels = []

    for texts, labels in dataloader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = sum(losses) / len(losses)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, acc, f1


def eval_epoch(model, dataloader, criterion):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = sum(losses) / len(losses)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, acc, f1


def train(
    epochs=5,
    batch_size=64,
    lr=1e-3,
    max_len=200,
    embedding_matrix_path="data/processed/embedding_matrix.pt",
    csv_path="data/processed/imdb_clean.csv",
    vocab_path="data/processed/vocab.json",
    model_save_path="models/lstm_glove.pth",
):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("Loading embedding matrix...")
    embedding_matrix = torch.load(embedding_matrix_path)

    print("Initializing model...")
    model = LSTMSentimentClassifier(embedding_matrix).to(DEVICE)

    print("Preparing data loaders...")
    train_loader, val_loader, _ = get_dataloaders(csv_path, vocab_path, batch_size=batch_size, max_len=max_len)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0

    for epoch in range(1, epochs + 1):
        print(f"Training epoch: {epoch}")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        print(f"validating epoch: {epoch}")
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with val F1: {best_val_f1:.4f}")

        print('--------------')

if __name__ == "__main__":
    train()
