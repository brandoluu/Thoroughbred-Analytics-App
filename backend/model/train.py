
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd
from model.model import Model
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model.evaluate import evaluate
from model.dataset import HorseDataset  
import time


def trainOneEpoch(model, loader, optimizer, device, grad_clip=True, use_amp=True):
    model.train()

    totalLoss = 0.0
    totalLossPrint = 0.0
    n = 0

    for batchIdx, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        y_true = batch["rating"].float()
        

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(batch).squeeze(-1)

        loss = F.mse_loss(y_pred, y_true)

        # regulatization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        regularized_loss = loss + 1e-3 * l1_norm

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        ySize = y_true.size(0)
        totalLoss += regularized_loss.item() * ySize

        totalLossPrint += loss.item() * ySize
        n += ySize

    return totalLoss / max(n,1), totalLossPrint / max(n,1)


"""
Function to train model on a given dataset.


Input: dataset: Clean dataset as a .csv file
       num_epochs: the number of epochs to train the model
       learning rate: set the learning rate of model training
       batch_size: size of the batches being trained for each epoch
       path_name: the name of the finished model with weights
Output: None (saves the best model as 'path_name')
"""
def trainModel(dataset, num_epochs, path_name, learning_rate, batch_size):

    best_model_loss = float('inf')
    best_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" ======= Using device: {device} ======= ")

    # Laoding the dataset and drop the name column
    df = pd.read_csv(dataset) 
    
    df = df.drop(["name"], axis=1)

    df = HorseDataset(df) 
    print(" ======= Successfully loaded dataset ======= ")

    # Creating the train and test datasets
    dfTrain, dfVal = train_test_split(df, test_size=0.2, random_state=42)


    trainLoader = DataLoader(dfTrain, batch_size=batch_size, shuffle=True, num_workers=0)
    valLoader = DataLoader(dfVal, batch_size=batch_size, shuffle=False, num_workers=0)
    print(" ======= Successfully created dataloaders ======= ")

    modelInstance = Model(dimension=64, numFeatures=12).to(device)
    optimizer = optim.Adam(modelInstance.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print(" ======= Successfully created model and optimizer ======= ")

    # Training parameters to be added to command line arguments after

    # training loop
    early_stopping_patience = 10
    early_stopping_counter = 0
    print(" ======= Starting training =======\n")
    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        _, trainMSEPrint = trainOneEpoch(modelInstance, trainLoader, optimizer, device)

        validation = evaluate(modelInstance, valLoader, device)

        if scheduler is not None:
            scheduler.step(validation['mse'])

        print(f"Epoch {epoch:03d} | "
              f"Train MSE: {trainMSEPrint:.2f} | "
              f"Val MSE: {validation['mse']:.2f} | "
              f"Val RMSE: {validation['rmse']:.2f} | "
              f"Val MAE: {validation['mae']:.2f}")
        
        if validation['mse'] < best_model_loss:
            best_model_loss = validation['mse']
            best_model = copy.deepcopy(modelInstance)
            print(f"Best model saved with Val MSE: {best_model_loss:.5f}\n")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    torch.save(best_model.state_dict(), f"{path_name}.pth")

    end_time = time.time()
    total_time = end_time - start_time
    print(f" ======= Training complete in {total_time/60:.2f} minutes ======= ")

