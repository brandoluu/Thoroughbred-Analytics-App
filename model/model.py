import os
import sys
import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, dimension=64, numFeatures=12):
        super().__init__()

        # Embedding layers:
        self.embedding = nn.Embedding(65023, dimension)  # Example embedding layer
        self.embeddingSire = nn.Embedding(1030, dimension)
        self.embeddingDam = nn.Embedding(27100, dimension)
        self.embeddingBmSire = nn.Embedding(2091, dimension)

        # Dropout layer and normalization layer:
        self.dropout = nn.Dropout(0.25)
        self.norm = nn.LayerNorm(dimension * 4)

        self.batchNorm = nn.BatchNorm1d(numFeatures)




    def forward(self, x):
        return