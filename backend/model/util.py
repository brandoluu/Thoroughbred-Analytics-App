import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from model.model import Model
from model.dataset import HorseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from torch.utils.data import DataLoader, Subset

"""
Function to preprocess a csv file, returning a cleaned DataFrame.

Input: path_to_csv: str - Path to the CSV file to be processed.

Output: df: pd.DataFrame - The cleaned DataFrame.
        idToName: dict - Mapping from unique IDs back to horse names.
"""
def preprocess_csv(path_to_csv):
    df = pd.read_csv(path_to_csv, low_memory=False)

    df = df[df['name'] != 'Unnamed']
    print(f"Shape of dataset: {df.shape}")

    # Dropping columns we know we don't need:
    df = df.drop(columns=['ems', 'grade', 'grade4', 'code', 'lot', 'price', 'status', 'vendor', 'purchaser', 'prev. price'], axis=1)

    # converting fees to a numeric value
    df['fee'] = pd.to_numeric(df['fee'], errors='coerce')

    # removing horses with ratings of 0
    df = df[df['rating'] > 0]


    # ---- Turning the birth year to the age of the horse ----
    df['yob'] = 2025 - df['yob']
    df = df.rename(columns={'yob': 'age'})

    # ---- Encoding the ordinal features (form) ----
    ordinalEncoder = OrdinalEncoder()
    encodedForm = ordinalEncoder.fit_transform(np.array(df['form']).reshape(-1,1))
    encodedFormDam = ordinalEncoder.fit_transform(np.array(df['form2']).reshape(-1,1))

    df = df.drop(['form', 'form2'], axis=1)
    df['form'] = encodedForm
    df['damForm'] = encodedFormDam


    # ---- Encoding the names of the horses with label encoding ----
    labels = df['sex'].unique()
    uniqueNames = pd.concat([df['name'], df['sire'], df['dam'], df['bmSire']]).unique()
    nameToId = {name: idx for idx, name in enumerate(uniqueNames)}
    idToName = {idx: name for idx, name in enumerate(uniqueNames)}

    df['name_encoded'] = df['name'].map(nameToId)
    df['sire'] = df['sire'].map(nameToId)
    df['dam'] = df['dam'].map(nameToId)
    df['bmSire'] = df['bmSire'].map(nameToId)



    # ---- One hot encoding for gender ----
    hotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    encodedSex = hotEncoder.fit_transform(df[['sex']])

    sexCols = hotEncoder.get_feature_names_out(['sex'])
    sexDf = pd.DataFrame(encodedSex, columns=sexCols, index=df.index)

    df = pd.concat([df.drop(columns=['sex']), sexDf], axis=1)

    print(f"number of unique names: {df['name'].nunique()}")
    print(f"number of unique sires: {df['sire'].max()}")
    print(f"number of unique dams: {df['dam'].max()}")
    print(f"number of unique bmSires: {df['bmSire'].max()}")


    # ---- Filling in missing values in Fee category ----
    df['fee'] = df['fee'].fillna(df['fee'].median())


    return df, idToName

"""
takes a clean datasaet and returns a map horse names to indices. 
"""
def encode_names(clean_dataset):
    df = pd.read_csv(clean_dataset)

    uniqueNames = pd.concat([df['name'], df['sire'], df['dam'], df['bmSire']]).unique()
    nameToId = {idx: name for idx, name in enumerate(uniqueNames)}
    
    return nameToId


"""
Custom helper collate function to properly batch dictionary data
"""
def collate_fn(batch):
    batched = {}
    for key in batch[0].keys():
        batched[key] = torch.stack([item[key] for item in batch])
    return batched

"""
Function to plot connected dot plot of predicted horse ratings vs actual horse ratings.

Input: results: dictionary of results from the predictions made by the model. 

Output: None (displays a plot)
"""
def plot_connected_dot_plot(results):

    df_results = pd.DataFrame(results)

    fig, ax = plt. subplots(figsize=(14, 8))
    x_pos = np.arange(len(df_results))

    # plotting the actual ratings (blue circles)
    actual_plots = ax.scatter(x_pos, df_results['actual_rating'], 
                        color='steelblue', s=120, label='Actual Rating', 
                        marker='o', alpha=0.8, zorder=3)

    # plotting the predicted ratings (red squares)
    predicted_plots = ax.scatter(x_pos, df_results['predicted_rating'], 
                           color='orange', s=120, label='Predicted Rating', 
                           marker='s', alpha=0.8, zorder=3)

    # connected the two plots with a verticle line
    for i in range(len(df_results)):
        ax.plot([i, i], 
            [df_results.iloc[i]['actual_rating'], df_results.iloc[i]['predicted_rating']], 
            color='gray', alpha=0.6, linewidth=2, zorder=1)
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_results['horse_name'], rotation=45, ha='right')
    ax.set_xlabel('Horse Names', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)    
    ax.set_title('Horse Ratings: Model Predictions vs Actual Values', fontsize=14,  pad=20)

    # Add legend
    ax.legend(loc='upper right', fontsize=11)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, zorder=0)

    # Add value labels next to dots
    for i, row in df_results.iterrows():
        # Label for actual rating (blue dot)
        ax.annotate(f'{row["actual_rating"]:.2f}', 
                    (i, row['actual_rating']), 
                    xytext=(-15, 10), textcoords='offset points',
                    fontsize=9, color='steelblue', weight='bold',
                    ha='center')

        # Label for predicted rating (orange square)
        ax.annotate(f'{row["predicted_rating"]:.2f}', 
                    (i, row['predicted_rating']), 
                    xytext=(15, 10), textcoords='offset points',
                    fontsize=9, color='orange', weight='bold',
                    ha='center')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean Absolute Error: {df_results['error'].mean():.3f}")
    print(f"Max Error: {df_results['error'].max():.3f}")
    print(f"Min Error: {df_results['error'].min():.3f}")



"""
Helper function to calculate the number of correct predictions vs actual with 10 points of tolerance
"""
def calculateAccuracy(predictions):

    predictions = pd.DataFrame(predictions)

    totalSampleSize = len(predictions)
    num_valid = 0
    # compare the ratings with a tolerance of 10 points
    for i in range(totalSampleSize):
        if predictions.iloc[i]['actual_rating'] - 10 < predictions.iloc[i]['predicted_rating'] and predictions.iloc[i]['predicted_rating'] < predictions.iloc[i]['actual_rating'] + 10:
            num_valid += 1

    accuracy = (num_valid / totalSampleSize) * 100

    print(f"Accuracy (with 10 points of tolerance): {accuracy:.2f}%")