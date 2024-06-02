from fastapi import FastAPI, HTTPException, Request
import joblib
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import joblib
from comet_ml import API
import requests

load_dotenv()

app = FastAPI()


class ModelSignature(BaseModel):
    columns: list = Field(..., example=["feature1", "feature2"])


class PredictionRequest(BaseModel):
    data: dict


class BatchPredictionRequest(BaseModel):
    data: list


api = API(api_key=os.getenv("COMET_API_KEY"))


def fetch_model():
    try:
        model_url = api.get_model(
            workspace=os.getenv("COMET_WORKSPACE"),
            model_name=os.getenv("COMET_MODEL_NAME"),
        )
        latest_version = model_url.find_versions()[0]
        print("Model version:", latest_version)
        model_url.download(
            version=latest_version,
            expand=True,
            output_folder="./",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model : {str(e)}")


fetch_model()
model_path = "model.pkl"
model = joblib.load(model_path)

booster = model.get_booster()
feature_names = booster.feature_names
print(feature_names)
# signature = ModelSignature(columns=model_signature["columns"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        data = pd.DataFrame(request.data, columns=feature_names, index=[0])
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    try:
        data = pd.DataFrame(request.data, columns=feature_names)
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate(request: Request):
    try:
        data = await request.json()
        df = pd.DataFrame(data["data"], columns=feature_names)
        return {"message": "Data is valid", "data": df.head().to_dict()}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
