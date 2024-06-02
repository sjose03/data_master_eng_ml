import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


class TestAPI(unittest.TestCase):

    @patch("api.main.API")
    @patch("api.main.requests.get")
    def setUp(self, MockAPI, mock_requests_get):
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

        # Mock the requests.get call
        mock_requests_get.side_effect = [
            MagicMock(content=b"mock model content"),
            MagicMock(json=lambda: {"columns": ["feature1", "feature2"]}),
        ]

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_predict(self):
        response = self.client.post(
            "/predict",
            json={
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
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_batch_predict(self):
        response = self.client.post(
            "/batch_predict",
            json={
                "data": [
                    [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
                    [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
                ]
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_validate(self):
        response = self.client.post(
            "/validate",
            json={"data": [[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]]},
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
