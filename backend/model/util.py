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
import joblib


formHeirarchy = {
    'G1w': 9,  
    'G2w': 8,  
    'G3w': 7, 
    'G1p': 6,  
    'G2p': 5,  
    'G3p': 4,  
    'BTw': 3,  
    'W': 2,    
    'P': 1,    
    'UR': 0,   
    'UP': 0,   
    'UNKNOWN': 0,
    np.nan: 0,
    '': 0
}


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

    # removing horses with ratings of 0 -> means they haven't raced yet?
    df = df[df['rating'] > 0]

    # ---- Turning the birth year to the age of the horse ----
    df['yob'] = 2026 - df['yob']
    df = df.rename(columns={'yob': 'age'})

    # # Get all unique form values across both columns
    # uniqueForms = pd.concat([df['form'], df['form2']]).unique()
    # uniqueForms = [f for f in uniqueForms if pd.notna(f)]  # Remove NaN
    
    # # Create mapping with 0 reserved for unknown/missing
    # formToId = {form: idx + 1 for idx, form in enumerate(uniqueForms)}
    # formToId[np.nan] = 0  # Handle missing values
    # formToId['UNKNOWN'] = 0  # Handle unknown values
    
    # Save the mapping for later use
    # joblib.dump(formToId, "data/formEncoder.pkl")
    
    # Encode both form columns
    df['form'] = df['form'].fillna('UNKNOWN')
    df['form2'] = df['form2'].fillna('UNKNOWN')

    # Map to hierarchy
    df['form'] = df['form'].map(formHeirarchy).fillna(0).astype(int)
    df['damForm'] = df['form2'].map(formHeirarchy).fillna(0).astype(int)

    # Drop original form2 column
    df = df.drop(['form2'], axis=1)
    
    # ---- Encoding the names of the horses with label encoding ----
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

    return df, idToName  # Return formToId for inference

"""
Helper function to create a clean dataframe from raw input data in our 
application
"""
def clean_df_input(df, dataset="data/baseData.csv"):

    # Load the ORIGINAL dataset to get all the unique values
    original_df = pd.read_csv(dataset)
    
    # Create separate mappings for each field
    nameToId = {name: idx for idx, name in enumerate(original_df['name'].unique())}
    sireToId = {name: idx for idx, name in enumerate(original_df['sire'].unique())}
    damToId = {name: idx for idx, name in enumerate(original_df['dam'].unique())}
    bmSireToId = {name: idx for idx, name in enumerate(original_df['bmSire'].unique())}
    
    ordinalEncoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=0)
    ordinalEncoder = joblib.load("data/ordinalEncoder.pk1")

    # ---- Turning the birth year to the age of the horse ----
    df['yob'] = 2026 - df['yob']
    df = df.rename(columns={'yob': 'age'})

    #print(df.head())

    # ---- Encoding the ordinal features (form) ----
    
    encodedForm = ordinalEncoder.transform(np.array(df['form']).reshape(-1,1))
    encodedFormDam = ordinalEncoder.transform(np.array(df['damForm']).reshape(-1,1))

    df['form'] = encodedForm
    df['damForm'] = encodedFormDam

    # Use separate default IDs for each field (last valid index)
    default_name_id = len(nameToId) - 1
    default_sire_id = len(sireToId) - 1
    default_dam_id = len(damToId) - 1
    default_bmSire_id = len(bmSireToId) - 1
    
    # Map with appropriate defaults
    df['name_encoded'] = df['name'].map(nameToId).fillna(default_name_id).astype(int)
    df['sire'] = df['sire'].map(sireToId).fillna(default_sire_id).astype(int)
    df['dam'] = df['dam'].map(damToId).fillna(default_dam_id).astype(int)
    df['bmSire'] = df['bmSire'].map(bmSireToId).fillna(default_bmSire_id).astype(int)

    # Verify all values are within bounds
    # print(f"\nValue ranges:")
    # print(f"name_encoded: {df['name_encoded'].min()} to {df['name_encoded'].max()} (max allowed: 49908)")
    # print(f"sire: {df['sire'].min()} to {df['sire'].max()} (max allowed: 50753)")
    # print(f"dam: {df['dam'].min()} to {df['dam'].max()} (max allowed: 71785)")
    # print(f"bmSire: {df['bmSire'].min()} to {df['bmSire'].max()} (max allowed: 73189)")

    # ---- One hot encoding for gender ----
    hotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    horse_genders = np.array([['F'], ['C'], ['R'], ['G']])
    
    hotEncoder.fit(horse_genders)
    encodedSex = hotEncoder.transform(df[['sex']])

    sexCols = hotEncoder.get_feature_names_out(['sex'])
    sexDf = pd.DataFrame(encodedSex, columns=sexCols, index=df.index)

    df = pd.concat([df.drop(columns=['sex']), sexDf], axis=1)

    # ---- Filling in missing values in Fee category ----
    df['fee'] = df['fee'].fillna(df['fee'].median())
    print(f"\n {df.head(1)}")

    print(f"\nDatatypes:\n {df.dtypes}")
    return df


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
        if predictions.iloc[i]['actual_rating'] - 15 < predictions.iloc[i]['predicted_rating'] and predictions.iloc[i]['predicted_rating'] < predictions.iloc[i]['actual_rating'] + 15:
            num_valid += 1

    accuracy = (num_valid / totalSampleSize) * 100

    print(f"Accuracy (with 10 points of tolerance): {accuracy:.2f}%")


"""
Function to plot the learning curve showing training and validation metrics over epochs.

Input: epoch_history: dict containing lists of metrics tracked per epoch
       - 'epochs': list of epoch numbers
       - 'train_mse': list of training MSE values
       - 'val_mse': list of validation MSE values
       - 'val_mae': list of validation MAE values
       - 'learning_rates': list of learning rates per epoch

Output: None (displays a plot)
"""
def plot_learning_curve(epoch_history):
    if not epoch_history or 'epochs' not in epoch_history or len(epoch_history['epochs']) == 0:
        print("No epoch history to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = epoch_history['epochs']
    
    # Plot 1: MSE (Training vs Validation)
    axes[0].plot(epochs, epoch_history['train_mse'], label='Train MSE', marker='o', linewidth=2, markersize=6)
    axes[0].plot(epochs, epoch_history['val_mse'], label='Val MSE', marker='s', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Training and Validation MSE', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Validation MAE
    axes[1].plot(epochs, epoch_history['val_mae'], label='Val MAE', marker='o', color='green', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Validation MAE', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    best_val_mae_idx = np.argmin(epoch_history['val_mae'])
    best_val_mse_idx = np.argmin(epoch_history['val_mse'])
    
    print(f"\n=== Learning Curve Summary ===")
    print(f"Best Val MAE: {epoch_history['val_mae'][best_val_mae_idx]:.4f} at epoch {epochs[best_val_mae_idx]}")
    print(f"Best Val MSE: {epoch_history['val_mse'][best_val_mse_idx]:.4f} at epoch {epochs[best_val_mse_idx]}")
    print(f"Final Train MSE: {epoch_history['train_mse'][-1]:.4f}")
    print(f"Final Val MSE: {epoch_history['val_mse'][-1]:.4f}")
    print(f"Final Val MAE: {epoch_history['val_mae'][-1]:.4f}")