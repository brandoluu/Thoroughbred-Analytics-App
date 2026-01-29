import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from model.model import Model
from model.dataset import HorseDataset
from model.util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from torch.utils.data import DataLoader, Subset


"""
Function to make predictions from sample dataset and evaluate. 

Input: model (path to model)
       dataset (path to dataset)s
       plot_results (bool -> plot the results)
       num_samples (int -> number of samples)

output: list of predictions based on number of samples proivded
"""
def predict_samples(model_path, dataset_path, plot_results, num_samples):
    predictions = []

    # model loading
    model = Model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_states = torch.load(model_path, map_location=device)
    model.load_state_dict(model_states) 
    model.to(device)
    model.eval()

    # data preperation and subset creation
    dataset = pd.read_csv(dataset_path)

    # Map row indices to horse names (ensures correct mapping when selecting random rows)
    idToName = dataset['name'].to_dict()

    # Drop the name column for model input
    dataset = dataset.drop(['name'], axis=1)
    model_input = HorseDataset(dataset)

    random_indices = random.sample(range(len(dataset)), num_samples)
    subset_dataset = Subset(model_input, random_indices)

    dataloader = DataLoader(subset_dataset, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=collate_fn)



    # predictions
    for batch_idx, batch in enumerate(dataloader):
        # Get the original index from the random_indices list
        original_index = random_indices[batch_idx]
    
        # Move batch to device
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            prediction = model(batch)

        # Use the original index for mapping
        horse_name = idToName[original_index]

        #print(f"Predicted rating for {horse_name} is {prediction.item()}")


        # For grahing:
        actual_rating = dataset.iloc[original_index]['rating']
        predictions.append({
            'horse_name': horse_name,
            'predicted_rating': prediction.item(),
            'actual_rating': actual_rating,
            'error': abs(prediction.item() - actual_rating)
        })

    if plot_results:
        plot_connected_dot_plot(predictions)

    calculateAccuracy(predictions)
    return predictions

"""

"""
def predict_new(input, model_path):

    # Model instantiation
    model = Model()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_states = torch.load(model_path, map_location=device)
    model.load_state_dict(model_states) 
    model.to(device)
    model.eval()

    df = pd.read_csv(input)

    dataloader = DataLoader(df, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=collate_fn)


