from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from model.model import Model
from model.train import trainModel
from model.util import *
import pandas as pd
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model = Model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_states = torch.load("model/trainedModels/base.pth", map_location=device)
model.load_state_dict(model_states) 
model.to(device)
model.eval

# Pydantic model for input validation and tensor conversion
class HorseData(BaseModel):
    name: str
    form: str
    rawErg: float
    erg: float
    ems: float
    grade: float
    yob: int
    sex: str
    sire: str
    fee: float
    crop: int
    dam: str
    form2: str
    ems3: int
    grade4: str
    bmSire: str
    price: float
    status: str
    code: str
    lot: int
    vendor: str
    purchaser: str
    prev_price: float

class predictionResponse(BaseModel):
    predicted_rating: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Horse Rating Prediction API!"}

@app.post
def predict_rating(horse_data: HorseData) -> predictionResponse:


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)