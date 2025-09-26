import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()

        dimension=64

        # Embedding layers:
        self.embedding = nn.Embedding(65023, dimension)  # Example embedding layer
        self.embeddingSire = nn.Embedding(65926, dimension)
        self.embeddingDam = nn.Embedding(88981, dimension)
        self.embeddingBmSire = nn.Embedding(90479, dimension)

        # Dropout layer and normalization layer:
        self.dropout = nn.Dropout(0.25)
        self.norm = nn.LayerNorm(dimension * 4)

        self.batchNorm = nn.BatchNorm1d(15)
        self.project = nn.Linear(15, dimension)

        self.network = nn.Sequential(
            nn.Linear(dimension * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),  
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        
        # creating embeddings
        nameEmbedding = self.embedding(x["name_encoded"].long())
        sireEmbedding = self.embeddingSire(x["sire"].long())
        damEmbedding = self.embeddingDam(x["dam"].long())   
        bmSireEmbedding = self.embeddingBmSire(x["bmSire"].long())  

        # Concatenating embeddings
        embedding = torch.cat((nameEmbedding, sireEmbedding, damEmbedding, bmSireEmbedding), dim=1)
        embedding = self.dropout(embedding)
        embedding = self.norm(embedding)

        numeric = self.batchNorm(x["numeric"])
        numeric = self.project(numeric)

        dataset = torch.cat((embedding, numeric), dim=1)
        y = self.network(dataset)
            
        return y