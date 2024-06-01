import joblib
from comet_ml import API
import os

api = API(api_key="your-comet-api-key")
model = api.get_model_registry_model("your-project-name", "xgboost-regression-model", version="1.0.0")

model.download("model.pkl")
