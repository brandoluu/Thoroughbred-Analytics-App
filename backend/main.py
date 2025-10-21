from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from model.model import Model
from model.train import trainModel
from model.util import *
import pandas as pd
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
model_states = torch.load("model/trainedModels/base2.pth", map_location=device)
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


def preprocess_input(horse_data: HorseData) -> torch.Tensor:
    """
    Preprocess a single horse input to match training preprocessing
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([horse_data.model_dump()])

    df = clean_df_input(df)
    df = df.drop(columns=['name'], axis=1)
    print(f"\n {df.dtypes}")
    
    inputTensor = HorseDataset(df)[0]
    
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)