from fastapi import FastAPI, HTTPException, Request
import joblib
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
from comet_ml import API

load_dotenv()

app = FastAPI()


class ModelSignature(BaseModel):
    columns: list = Field(..., example=["feature1", "feature2"])


class PredictionRequest(BaseModel):
    data: dict = Field(..., example={"feature1": 1.0, "feature2": 2.0})


class BatchPredictionRequest(BaseModel):
    data: list = Field(
        ...,
        example=[
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0},
        ],
    )


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
signature = ModelSignature(columns=feature_names)

# Create examples dynamically based on feature names
example_single = {feature: 0.0 for feature in feature_names}
example_batch = [{feature: 0.0 for feature in feature_names} for _ in range(2)]


class PredictionRequest(BaseModel):
    data: dict = Field(..., example=example_single)


class BatchPredictionRequest(BaseModel):
    data: list = Field(..., example=example_batch)


@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Ensure the request data keys match the feature names
        request_data = request.data
        if set(request_data.keys()) != set(feature_names):
            raise HTTPException(
                status_code=400, detail="Invalid features in request data"
            )

        data = pd.DataFrame([request_data], columns=feature_names)
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
        # Ensure each item in request data keys match the feature names
        request_data = request.data
        for item in request_data:
            if set(item.keys()) != set(feature_names):
                raise HTTPException(
                    status_code=400, detail="Invalid features in request data"
                )

        data = pd.DataFrame(request_data, columns=feature_names)
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

    uvicorn.run(
        app, host="0.0.0.0", port=os.getenv("PORT") if os.getenv("PORT") else 8000
    )
