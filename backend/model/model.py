import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dimension=64, num_forms=1000):  # Adjust num_forms based on your data
        super().__init__()

        dimension = 64

        # Embedding layers
        self.embedding = nn.Embedding(49909, dimension)
        self.embeddingSire = nn.Embedding(50754, dimension)
        self.embeddingDam = nn.Embedding(71786, dimension)
        self.embeddingBmSire = nn.Embedding(73190, dimension)

        # NEW: Form embeddings (add padding_idx=0 for unknown forms)
        self.embeddingForm = nn.Embedding(num_forms + 1, dimension, padding_idx=0)
        self.embeddingDamForm = nn.Embedding(num_forms + 1, dimension, padding_idx=0)

        # Dropout and normalization
        self.dropout = nn.Dropout(0.25)
        self.norm = nn.LayerNorm(dimension * 6)  # Changed from 4 to 6

        # Numeric features (now 13 instead of 15, removed form and damForm)
        self.batchNorm = nn.BatchNorm1d(13)

        self.network = nn.Sequential(
            nn.Linear(dimension * 6 + 13, 1024),  # Changed to 6 embeddings
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),  
            nn.Dropout(0.2),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),  
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),   

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
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

        dataset = torch.cat((embedding, numeric), dim=1)
        y = self.network(dataset)
            
        return y
    
class recallModel(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()

        dimension=64

        # Embedding layers (number of features of each labeled category + 1 *for unknowns* ): 
        self.embedding = nn.Embedding(49909, dimension)  # Example embedding layer
        self.embeddingSire = nn.Embedding(50754, dimension)
        self.embeddingDam = nn.Embedding(71786, dimension)
        self.embeddingBmSire = nn.Embedding(73190, dimension)

        # Dropout layer and normalization layer:
        self.dropout = nn.Dropout(0.25)
        self.norm = nn.LayerNorm(dimension * 4)

        # recallModel used for /recall (includes `form`) -> numeric length = 16
        self.batchNorm = nn.BatchNorm1d(16)

        self.network = nn.Sequential(
            nn.Linear(dimension * 4 + 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),


            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),  
            nn.Dropout(0.2),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),  
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),   
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1)
        )

    def forward(self, x):
    
        # Creating embeddings for all categorical features
        nameEmbedding = self.embedding(x["name_encoded"].long())
        sireEmbedding = self.embeddingSire(x["sire"].long())
        damEmbedding = self.embeddingDam(x["dam"].long())   
        bmSireEmbedding = self.embeddingBmSire(x["bmSire"].long())
        formEmbedding = self.embeddingForm(x["form"].long())
        damFormEmbedding = self.embeddingDamForm(x["damForm"].long())
    
        # Concatenating all embeddings (now 6 instead of 4)
        embedding = torch.cat((nameEmbedding, sireEmbedding, damEmbedding, bmSireEmbedding, 
                              formEmbedding, damFormEmbedding), dim=1)
        embedding = self.dropout(embedding)
        embedding = self.norm(embedding)
    
        numeric = self.batchNorm(x["numeric"])
    
        dataset = torch.cat((embedding, numeric), dim=1)
        y = self.network(dataset)
            
        return y