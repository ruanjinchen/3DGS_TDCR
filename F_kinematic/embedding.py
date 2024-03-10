import torch
import torch.nn as nn
class SimpleEmbedding(nn.Module):
    def __init__(self, input_dim=5, embedding_dim=128):
        super().__init__()
        # deep linear layer with relu activation
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        # self.embedding.weight.data.uniform_(-1.0 / math.sqrt(embedding_dim), 1.0 / math.sqrt(embedding_dim))
        # self.embedding = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

class EmbeddingDecomposition(nn.Module):
    def __init__(self, input_dim=256, embedding_dim=128, num_part=5):
        super().__init__()
        self.num_part = num_part
        self.embedding_dim = embedding_dim
        self.up_scale = nn.Linear(input_dim, embedding_dim * num_part)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0, batch_first=True), num_layers=2)

    def forward(self, x):
        x = self.up_scale(x).reshape(self.num_part, -1)
        x = self.encoder(x)
        return x