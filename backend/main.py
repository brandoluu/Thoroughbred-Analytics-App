from model.model import *
from model.util import *
from model.train import *
from model import model
import torch
import pandas as pd
import argparse
import sys


def create_parser():
    parser = argparse.ArgumentParser(description="Horse Predictor Rating")

    subparsers = parser.add_subparsers(
        dest='command',
        title='commands',
        description='Available commands',
        help='Choose a command to run',
        required=True
    )

    # Training argument
    train_parser = subparsers.add_parser('train',
                                         help='Train a new model',
                                         description='Train a new model with a specific dataset.')
    train_parser.add_argument('--data', type=str, required=True,
                             help='Path to training data CSV')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=64,
                             help='Training batch size')
    train_parser.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--output', type=str, default='model.pth',
                             help='Output model path')
    
    predict_parser = subparsers.add_parser('predict',
                                           help='Make a random prediction',
                                           description='With a trained model, make a random prediction with samples within'
                                           'the dataset to visualize model predictions vs actual ratings.')
    predict_parser.add_argument('--model', type=str, default="model/best_model.pth",
                                help='Path to saved model')
    predict_parser.add_argument('--num-samples', type=int, default=20,
                                help='Number of random samples to predict and plot')
    predict_parser.add_argument('--graph', type=bool, default=False,
                               help='show a plot of the predicted values vs the actual values.')
    
    
    dataset_parser = subparsers.add_parser('create-dataset',
                                           help='create a new dataset from csv file',
                                           description='create a clean dataset for model training from an ' \
                                           'input .csv file.')
    dataset_parser.add_argument("input path",
                                help="path to the dataset that will be processed")
    dataset_parser.add_argument("output dataset",
                                help="name of the output dataset. ")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    print(args)



if __name__ == "__main__":
    main()