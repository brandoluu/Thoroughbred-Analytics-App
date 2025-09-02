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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification





if __name__ == "__main__":
    df = pd.read_csv('../horseDataProcessed.csv')  # Load your dataset

    X = df.drop(columns=['rating']).values  # Features
    y = df['rating'].values

    print("Features shape:", X.shape)