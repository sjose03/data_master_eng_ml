import pytest
from fastapi.testclient import TestClient
from app.main import app, SECRET_KEY
import jwt

client = TestClient(app)


def create_token():
    payload = {"sub": "test"}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token


def test_predict():
    token = create_token()
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984126984126984,
        "AveBedrms": 1.0238095238095237,
        "Population": 322.0,
        "AveOccup": 2.5555555555555554,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    response = client.post("/predict", json=data, headers=headers)
    assert response.status_code == 200
    assert "predictions" in response.json()


def test_batch_predict():
    token = create_token()
    headers = {"Authorization": f"Bearer {token}"}
    data = [
        {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984126984126984,
            "AveBedrms": 1.0238095238095237,
            "Population": 322.0,
            "AveOccup": 2.5555555555555554,
            "Latitude": 37.88,
            "Longitude": -122.23,
        },
        {
            "MedInc": 8.3014,
            "HouseAge": 21.0,
            "AveRooms": 6.238137082601054,
            "AveBedrms": 0.9718804920913884,
            "Population": 2401.0,
            "AveOccup": 2.109841827768014,
            "Latitude": 37.86,
            "Longitude": -122.22,
        },
    ]
    response = client.post("/batch_predict", json=data, headers=headers)
    assert response.status_code == 200
    assert "predictions" in response.json()
