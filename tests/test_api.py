import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


class TestAPI(unittest.TestCase):

    @patch("api.main.app")
    def setUp(self, MockAPI):
        self.client = TestClient(app)

        # Mock the CometML API
        self.mock_api = MockAPI.return_value
        self.mock_api.get_registry_model_details.return_value = {
            "latestVersion": "1.0.0"
        }
        self.mock_api.get_registry_model_version_details.return_value = {
            "assets": [
                {"type": "model", "url": "http://example.com/model.pkl"},
                {
                    "type": "json",
                    "fileName": "signature.json",
                    "url": "http://example.com/signature.json",
                },
            ]
        }

    def test_home(self):
        response = self.client.get("/")
        expected_status_code = 200
        self.assertEqual(
            response.status_code,
            expected_status_code,
            f"Expected status code {expected_status_code}, got {response.status_code}",
        )

    def test_predict(self):
        request_data = {
            "data": {
                "MedInc": 8.3014,
                "HouseAge": 21,
                "AveRooms": 7099,
                "AveBedrms": 1106,
                "Population": 2401,
                "AveOccup": 1138,
                "Latitude": 37.86,
                "Longitude": -122.22,
                "rooms_per_household": 2.95752,
                "bedrooms_per_room": 0.155797,
                "population_per_household": 0.473194,
            }
        }
        response = self.client.post("/predict", json=request_data)
        expected_status_code = 200
        self.assertEqual(
            response.status_code,
            expected_status_code,
            f"Expected status code {expected_status_code}, got {response.status_code}. Request data: {request_data}, Response: {response.json()}",
        )

    def test_batch_predict(self):
        request_data = {
            "data": [
                {
                    "AveBedrms": 0,
                    "AveOccup": 0,
                    "AveRooms": 0,
                    "HouseAge": 0,
                    "Latitude": 0,
                    "Longitude": 0,
                    "MedInc": 0,
                    "Population": 0,
                    "bedrooms_per_room": 0,
                    "population_per_household": 0,
                    "rooms_per_household": 0,
                },
                {
                    "AveBedrms": 0,
                    "AveOccup": 0,
                    "AveRooms": 0,
                    "HouseAge": 0,
                    "Latitude": 0,
                    "Longitude": 0,
                    "MedInc": 0,
                    "Population": 0,
                    "bedrooms_per_room": 0,
                    "population_per_household": 0,
                    "rooms_per_household": 0,
                },
            ]
        }
        response = self.client.post("/batch_predict", json=request_data)
        expected_status_code = 200
        self.assertEqual(
            response.status_code,
            expected_status_code,
            f"Expected status code {expected_status_code}, got {response.status_code}. Request data: {request_data}, Response: {response.json()}",
        )

    def test_validate(self):
        request_data = {
            "data": [[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]]
        }
        response = self.client.post("/validate", json=request_data)
        expected_status_code = 200
        self.assertEqual(
            response.status_code,
            expected_status_code,
            f"Expected status code {expected_status_code}, got {response.status_code}. Request data: {request_data}, Response: {response.json()}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
