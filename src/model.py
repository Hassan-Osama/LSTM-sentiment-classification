import torch
import torch.nn as nn

class LSTMSentimentClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, num_layers=1, bidirectional=False, dropout=0.3):
        super(LSTMSentimentClassifier, self).__init__()
        
        vocab_size, embed_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.fc = nn.Linear(lstm_output_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        out = self.fc(hidden)
        out = self.sigmoid(out)
        return out.view(-1)

