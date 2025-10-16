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


def preprocess_input(horse_data: HorseData) -> torch.Tensor:
    """
    Preprocess a single horse input to match training preprocessing
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([horse_data.model_dump()])

    logger.info(f"Raw input data: {df}")
    
    name_to_id = encode_names("data/horseDataBase.csv")

    # Convert fee to numeric
    df['fee'] = pd.to_numeric(df['fee'], errors='coerce')
    
    # Convert birth year to age
    df['age'] = 2025 - df['yob']
    df = df.drop('yob', axis=1)
    
    # TODO: fix form encoding (Dummy Values)
    encoded_form = 0
    encoded_form_dam = 0
    

    df = df.rename(columns={'form2': 'damForm'})
    
    df['form'] = encoded_form
    df['damForm'] = encoded_form_dam

    # change the dtpye of the column
    df['form'] = df['form'].astype(float)
    df['damForm'] = df['damForm'].astype(float)

    # Encode names using saved name_to_id mapping: need to check if the embeddings are in the 
    # trained model embeddings, if not set to embedding for unknown
    df['name_encoded'] = df['name'].map(name_to_id).fillna(0)
    df['sire'] = df['sire'].map(name_to_id).fillna(0)
    df['dam'] = df['dam'].map(name_to_id).fillna(0)
    df['bmSire'] = df['bmSire'].map(name_to_id).fillna(0)
    
    # Drop original name column
    df = df.drop('name', axis=1)
    
    # Ensure all features are numeric and in correct order
    # Adjust this list to match your exact feature order from training
    feature_columns = ['name_encoded', 'form', 'rawErg', 'erg', 'ems', 'grade', 'age', 'sex', 
                       'sire', 'fee', 'crop', 'dam', 'damForm', 'ems3', 'grade4', 'bmSire', 
                       'price', 'status', 'code', 'lot', 'vendor', 'purchaser', 'prev_price']
    
    # Handle sex encoding if needed
    if 'sex' in df.columns:
        sex_mapping = {'M': 0, 'F': 1, 'G': 2, 'C': 3}  # Adjust based on your data
        df['sex'] = df['sex'].map(sex_mapping).fillna(0)
    
    # Select features in correct order
    try:
        features = df[feature_columns].values[0]
    except KeyError:
        # Use all numeric columns if specific order fails
        features = df.select_dtypes(include=[np.number]).values[0]
    
    # Convert to tensor
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    logger.info(tensor)
    return tensor.to(device)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Horse Rating Prediction API!"}

@app.post("/predict")
def predict_rating(horse_data: HorseData) -> predictionResponse:
    try:
        input_tensor = preprocess_input(horse_data)

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