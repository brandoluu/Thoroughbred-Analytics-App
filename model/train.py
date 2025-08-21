import torch
import torch.nn as nn
import torch.optim as optim

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import HorseDataset  
from model import model
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluate import evaluate

def trainOneEpoch(model, loader, optimizer, device, grad_clip=None, use_amp=True):
    model.train()

    totalLoss = 0.0
    n = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y_true = batch["rating"].float()

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            with torch.cuda.amp.autocast():
                y_pred = model(batch).squeeze(-1)
                loss = F.mse_loss(y_pred, y_true)
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(batch).squeeze(-1)
            loss = F.mse_loss(y_pred, y_true)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        ySize = y_true.size(0)
        totalLoss += loss.item() * ySize
        n += ySize

        return totalLoss / max(n,1)
    

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Laoding the dataset
    df = pd.read_csv('horseDataProcessed.csv')  # Load your dataset

    trainDs = HorseDataset(df)  # Assuming df_train is defined
    print(" ======= Successfully loaded dataset ======= ")

    # Creating the train and test
    dfTrain, dfVal = train_test_split(df, test_size=0.2, random_state=42)
    trainDataset = HorseDataset(dfTrain)
    valDataset = HorseDataset(dfVal)    

    trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True, num_workers=4)
    valLoader = DataLoader(valDataset, batch_size=64, shuffle=False, num_workers=4)
    print(" ======= Successfully created dataloaders ======= ")

    model = model(dimension=64, numFeatures=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    print(" ======= Successfully created model and optimizer ======= ")

    # Training parameters to be added to command line arguments after
    bestValLoss = float('inf')
    numEpochs = 10

    # training loop
    for epoch in range(numEpochs):
        trainMSE = trainOneEpoch(model, trainLoader, optimizer, device)

        validation = evaluate(model, valLoader, device)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch:03d} | "
              f"Train MSE: {trainMSE:.5f} | "
              f"Val MSE: {validation['mse']:.5f} | "
              f"Val RMSE: {validation['rmse']:.5f} | "
              f"Val MAE: {validation['mae']:.5f}")

