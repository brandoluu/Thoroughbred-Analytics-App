# Horse Rating Prediction

Custom trained AI models trained on over 60,000 samples powering a streamlined predictor for horses. 

## Dependencies
First, python2 or python3 must be installed on your computer. You can download it [here](https://www.python.org/downloads/). After cloning this repository onto your computer with `git clone`, open a terminal in the directory where the project is located. 

After, enter the command `pip install -r requirements.txt` or `pip3 install -r requiremnts.txt`. <mark>NOTE: if you want to train the model yourself, pytorch with CUDA must be installed seperately [here](https://pytorch.org/get-started/locally/) with the correct cuda version</mark> 

In addition, [Node JS](https://nodejs.org/en/download) must be installed in order to run the predictor client locally. 

## Running the Project

There are 2 components of the backend, `dev.py` for training, command-line predicitons, and creating datasets while `main.py` contains the API calls for the frontend. 

### Developer Side:
**`dev.py`**: in the command line make sure you are in `horsePrediction/backend` and run the following command: `python dev.py {train, predict, create-dataset}` with the following optional flags:

`train`: Train a model

| `data`| path to the dataset used to train the model. (Required)|
|:-------:|:-------------------------------------------------:|
| `--epochs`| number of epochs to train model. (Default: 50)|
| `--batch-size`| size of batches to be processed during forward pass (Default: 64)|
| `--lr` | learning rate (Default: 0.0004)|
| `--output`| name of the final model. Note: input must include the path as well (Default: `model/trainedModels/model`)| 

`predict`: Generate sample predictions from trained model:

| `--model` | path to the save model (Default: `model/trainedModels/base.pth`) |
|:---:|:---:|
| `--num-samples` | number of samples to generate for testing (Default: 20) |
| `--dataset` | path to the dataset used for the model (Default: `data/horseData/baseData.csv`) |
| `--graph` | boolean to decide whetever to generate a chart of predictions vs actual values (Default: `False`) |

`create-dataset`: Create a clean dataset from raw `.csv` files

| input_path | path to the dataset that will be processed (Required) |
|:---:|:---:|
| output_path | path to the clean dataset that will be the result of data processing (Required) |


## Client Side:

To start the client side, run the following command: `npm start` in the `frontend` directory. In addition, in the `../backend` directory, run the command `python main.py`