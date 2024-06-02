import unittest
from unittest.mock import patch, MagicMock
import joblib
from fastapi.testclient import TestClient
import sys
import os

os.environ["DISABLE_COMET_LOGGING"] = "true"
# Adicione o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.main import app
from dotenv import load_dotenv


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.client = TestClient(app)
        cls.model = joblib.load("model.pkl")

    @patch("comet_ml.Experiment")
    @patch("data.featurization.get_data_from_mongodb")
    def test_predict(self, mock_get_data, MockExperiment):
        # Mock the data returned by get_data_from_mongodb
        mock_data = {
            "MedInc": [8.3252, 8.3014],
            "HouseAge": [41, 21],
            "AveRooms": [880, 7099],
            "AveBedrms": [129, 1106],
            "Population": [322, 2401],
            "AveOccup": [126, 1138],
            "Longitude": [-122.23, -122.22],
            "Latitude": [37.88, 37.86],
            "rooms_per_household": [2.72857, 2.95752],
            "bedrooms_per_room": [0.146591, 0.155797],
            "population_per_household": [0.391304, 0.473194],
        }
        mock_target = [452600, 358500]
        mock_get_data.return_value = (mock_data, mock_target)

        # Mock the CometML experiment
        mock_experiment = MockExperiment.return_value
        mock_experiment.log_metric = MagicMock()
        mock_experiment.log_model = MagicMock(return_value="test_model_version")
        mock_experiment.end = MagicMock()

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
        self.assertIn("predictions", response.json())

    @patch("comet_ml.Experiment")
    @patch("data.featurization.get_data_from_mongodb")
    def test_batch_predict(self, mock_get_data, MockExperiment):
        # Mock the data returned by get_data_from_mongodb
        mock_data = {
            "MedInc": [8.3252, 8.3014],
            "HouseAge": [41, 21],
            "AveRooms": [880, 7099],
            "AveBedrms": [129, 1106],
            "Population": [322, 2401],
            "AveOccup": [126, 1138],
            "Longitude": [-122.23, -122.22],
            "Latitude": [37.88, 37.86],
            "rooms_per_household": [2.72857, 2.95752],
            "bedrooms_per_room": [0.146591, 0.155797],
            "population_per_household": [0.391304, 0.473194],
        }
        mock_target = [452600, 358500]
        mock_get_data.return_value = (mock_data, mock_target)

        # Mock the CometML experiment
        mock_experiment = MockExperiment.return_value
        mock_experiment.log_metric = MagicMock()
        mock_experiment.log_model = MagicMock(return_value="test_model_version")
        mock_experiment.end = MagicMock()

        data = [
            {
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
            },
            {
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
            },
        ]
        response = self.client.post("/batch_predict", json={"data": data})
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())
        # Restore environment variable
        del os.environ["DISABLE_COMET_LOGGING"]


if __name__ == "__main__":
    unittest.main()
