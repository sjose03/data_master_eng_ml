import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000"
TOKEN = os.getenv("API_TOKEN")

st.title("MLOps Pipeline - Batch Inference")


def validate_csv(file):
    df = pd.read_csv(file)
    required_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    if all(column in df.columns for column in required_columns):
        return df
    else:
        st.error("CSV file is not in the expected format")
        return None


def make_predictions(data):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.post(
        f"{API_URL}/batch_predict", json={"data": data}, headers=headers
    )
    if response.status_code == 200:
        return response.json()["predictions"]
    else:
        st.error("Failed to get predictions")
        return None


upload_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if upload_file is not None:
    df = validate_csv(upload_file)
    if df is not None:
        st.write(df)
        if st.button("Predict"):
            predictions = make_predictions(df.to_dict(orient="records"))
            if predictions:
                st.write(predictions)

st.sidebar.title("Online Prediction")
st.sidebar.write("Input data for a single prediction")

input_data = {
    "MedInc": st.sidebar.number_input("Median Income", min_value=0.0),
    "HouseAge": st.sidebar.number_input("House Age", min_value=0),
    "AveRooms": st.sidebar.number_input("Average Rooms", min_value=0.0),
    "AveBedrms": st.sidebar.number_input("Average Bedrooms", min_value=0.0),
    "Population": st.sidebar.number_input("Population", min_value=0),
    "AveOccup": st.sidebar.number_input("Average Occupancy", min_value=0.0),
    "Latitude": st.sidebar.number_input("Latitude", min_value=0.0),
    "Longitude": st.sidebar.number_input("Longitude", min_value=0.0),
}

if st.sidebar.button("Predict Online"):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.post(
        f"{API_URL}/predict", json={"data": input_data}, headers=headers
    )
    if response.status_code == 200:
        st.sidebar.write(response.json()["predictions"])
    else:
        st.sidebar.error("Failed to get prediction")
