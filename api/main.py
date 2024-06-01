import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
import jwt
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import comet_ml

load_dotenv()

app = FastAPI()
SECRET_KEY = os.getenv("SECRET_KEY")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

model = joblib.load("model.pkl")

# Obter a assinatura do modelo do CometML
experiment = comet_ml.API(api_key=os.getenv("COMET_API_KEY"))
model_info = experiment.get_model_registry_model(
    "your-workspace-name", "xgboost-regression-model", version="1.0"
)
signature = model_info["modelMetadata"]["signature"]


# Definir PredictionRequest baseado na assinatura
class PredictionRequest(BaseModel):
    data: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, Any]]


def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid token")


@app.post("/predict")
def predict(request: PredictionRequest, token: str = Depends(verify_token)):
    df = pd.DataFrame([request.data])
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}


@app.post("/batch_predict")
def batch_predict(request: BatchPredictionRequest, token: str = Depends(verify_token)):
    df = pd.DataFrame(request.data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
