
import numpy as np
from dataset import HorseDataset  
import pandas as pd

# Importing Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

class horse_predictor:
    def __init__(self, numerical_features, categorical_features):
        



if __name__ == "__main__":
    df = pd.read_csv('../horseDataProcessed.csv')  # Load your dataset

    X = df.drop(columns=['rating']).values  # Features
    y = df['rating'].values

    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)

    print("--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("--- Training Random Forest Classifier ---")
    randomForest = RandomForestClassifier(random_state=42, verbose=1)
    randomForest.fit(X_train, y_train)


    print("--- Evaluating Model With Accuracy ---")
    y_pred = randomForest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")