import torch
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from model import DenseModel

# FastAPI app instance
app = FastAPI()

# Define User input
class UserInput(BaseModel):
    age: float
    bmi: float
    children: int
    smoker: int
    sex: int
    region: str

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "Insurance_Cost_Forcasting_Model.pth"
DATA_PATH = PROJECT_DIR / "datasets" / "insurance.csv"

feature_data = pd.read_csv(DATA_PATH)

scaler = StandardScaler()
scaler.fit(feature_data[["age", "bmi"]])

y_train_mean = 13346.089736364485
y_train_std = 12019.510778442855

model = DenseModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

def preprocess_user_input(data: UserInput) -> torch.Tensor:
    try:
        scaled_age, scaled_bmi = scaler.transform([[data.age, data.bmi]])[0]
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        raise HTTPException(status_code=400, detail="Unable to scale age and bmi") from exc

    region = data.region.strip().lower()
    region_features = [
        int(region == "northeast"),
        int(region == "northwest"),
        int(region == "southeast"),
        int(region == "southwest"),
    ]

    if sum(region_features) != 1:
        raise HTTPException(
            status_code=400,
            detail="region must be one of: northeast, northwest, southeast, southwest",
        )

    features = [
        float(scaled_age),
        float(data.sex),
        float(scaled_bmi),
        float(data.children),
        float(data.smoker),
        *map(float, region_features),
    ]

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

# Routes

# default endpoint
@app.get("/")
def home():
    return {
        "message": "Backend is running"
    }

# prediction endpoint
@app.post("/predict")
def predict(data: UserInput):
    features = preprocess_user_input(data)

    with torch.inference_mode():
        normalized_prediction = model(features).squeeze()
        prediction = normalized_prediction * y_train_std + y_train_mean

    return {"prediction": float(prediction.item())}