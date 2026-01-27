
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from model.util import plot_learning_curve
import time


def trainOneEpoch(model, loader, optimizer, device, grad_clip=1.0, use_amp=False):
    model.train()
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    totalLoss = 0.0
    n = 0

    for batchIdx, batch in enumerate(loader):
        y_true = batch["rating"].float().to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        X = {k: v for k, v in batch.items() if k != "rating"}

        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training (optional but faster)
        if use_amp:
            with torch.cuda.amp.autocast():
                y_pred = model(X).squeeze(-1)
                loss = F.mse_loss(y_pred, y_true)
            
            scaler.scale(loss).backward()
            
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(X).squeeze(-1)
            loss = F.mse_loss(y_pred, y_true)
            
            loss.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        ySize = y_true.size(0)
        totalLoss += loss.item() * ySize
        n += ySize

    return model, totalLoss / max(n, 1)


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
    modelInstance = Model()
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

    modelInstance = Model(dimension=64).to(device)
    #optimizer = optim.AdamW(modelInstance.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer = optim.SGD(modelInstance.parameters(), lr=learning_rate, momentum=0.7, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    print(" ======= Successfully created model and optimizer ======= ")

    # Training parameters to be added to command line arguments after

    # training loop
    early_stopping_patience = 20
    early_stopping_counter = 0
    
    # Track metrics for learning curve
    epoch_history = {
        'epochs': [],
        'train_mse': [],
        'val_mse': [],
        'val_mae': [],
        'learning_rates': []
    }
    
    print(" ======= Starting training =======\n")
    start_time = time.time()
    for epoch in range(1, num_epochs+1):
    # trainOneEpoch returns (model, avg_loss) based on your function
        modelInstance, trainMSEPrint = trainOneEpoch(
            modelInstance, 
            trainLoader, 
            optimizer, 
            device,
            grad_clip=1.0,
            use_amp=False
        )

        validation = evaluate(modelInstance, valLoader, device)

        if scheduler is not None:
            scheduler.step(validation['mae'])

        # Track metrics for learning curve
        current_lr = optimizer.param_groups[0]['lr']
        epoch_history['epochs'].append(epoch)
        epoch_history['train_mse'].append(trainMSEPrint)
        epoch_history['val_mse'].append(validation['mse'])
        epoch_history['val_mae'].append(validation['mae'])
        epoch_history['learning_rates'].append(current_lr)

        print(f"Epoch {epoch:03d} | "
              f"Train MSE: {trainMSEPrint:.2f} | "
              f"Val MSE: {validation['mse']:.2f} | "
              f"Val MAE: {validation['mae']:.2f} | "
              f"LR: {current_lr:.6f}")

        # Save best model using state_dict (more efficient than deepcopy)
        if validation['mae'] < best_model_loss:
            best_model_loss = validation['mae']
            torch.save(modelInstance.state_dict(), f"{path_name}_best.pth")
            print(f"Best model saved with Val MAE: {best_model_loss:.5f}\n")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    modelInstance.load_state_dict(torch.load(f"{path_name}_best.pth"))

    end_time = time.time()
    total_time = end_time - start_time
    print(f" ======= Training complete in {total_time/60:.2f} minutes ======= ")
    
    # Plot the learning curve
    plot_learning_curve(epoch_history)

