from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from model.model import Model
from model.train import trainModel
from model.util import *
from fastapi.staticfiles import StaticFiles
import pandas as pd
from typing import Optional
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model and the trained dataset for embeddings
model = Model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_states = torch.load("model/trainedModels/experiment3L1Norm.pth", map_location=device)
model.load_state_dict(model_states) 
model.to(device)
model.eval

# class for input validation and conversion to a tensor
class HorseData(BaseModel):
    name: str
    form: str
    rawErg: float
    erg: float
    yob: int
    sex: str
    sire: str
    fee: float
    crop: int
    dam: str
    damForm: str
    ems3: int
    bmSire: str

# model for output response
class predictionResponse(BaseModel):
    predicted_rating: float


def preprocess_input(horse_data: HorseData) -> torch.Tensor:
    """
    Preprocess a single horse input to match training preprocessing
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([horse_data.model_dump()])

    df = clean_df_input(df)
    #print(df.head)
    df = df.drop(columns=['name'], axis=1)
    #print(f"\n {df.dtypes}")
    

    inputTensor = HorseDataset(df)[0] # need to take the first batch since we are using the dataset class
    
    # Convert to tensor
    logger.info(f"\n Input tensor: {inputTensor}")


    inputTensor = {k: v.unsqueeze(0).to(device) for k, v in inputTensor.items()}
    return inputTensor

@app.get("/")
def read_root():
    return {"message": "Welcome to the Horse Rating Prediction API!"}


@app.post("/predict")
def predict_rating(horse_data: HorseData) -> predictionResponse:
    try:
        input_tensor = preprocess_input(horse_data)
        print(f"\n{input_tensor}")

        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)

        predicted_rating = prediction.item()

        return predictionResponse(predicted_rating=predicted_rating)
    


    except Exception as e:
        print("\n" + "="*50)
        print("FULL ERROR TRACEBACK:")
        print("="*50)
        traceback.print_exc()
        print("="*50 + "\n")
        raise  # Re-raise the error so FastAPI shows it too


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True,
    }