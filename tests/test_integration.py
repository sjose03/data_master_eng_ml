import unittest
from unittest.mock import patch
import joblib
from fastapi.testclient import TestClient
from app.main import app
from dotenv import load_dotenv


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.client = TestClient(app)
        cls.model = joblib.load("model.pkl")

    @patch("comet_ml.Experiment")
    @patch("app.main.get_data_from_mongodb")
    def test_predict(self, mock_get_data, MockExperiment):
        # Mock the data returned by get_data_from_mongodb
        mock_data = {
            "longitude": [-122.23, -122.22],
            "latitude": [37.88, 37.86],
            "housing_median_age": [41, 21],
            "total_rooms": [880, 7099],
            "total_bedrooms": [129, 1106],
            "population": [322, 2401],
            "households": [126, 1138],
            "median_income": [8.3252, 8.3014],
            "median_house_value": [452600, 358500],
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
                    "longitude": -122.23,
                    "latitude": 37.88,
                    "housing_median_age": 41,
                    "total_rooms": 880,
                    "total_bedrooms": 129,
                    "population": 322,
                    "households": 126,
                    "median_income": 8.3252,
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())

    @patch("comet_ml.Experiment")
    @patch("app.main.get_data_from_mongodb")
    def test_batch_predict(self, mock_get_data, MockExperiment):
        # Mock the data returned by get_data_from_mongodb
        mock_data = {
            "longitude": [-122.23, -122.22],
            "latitude": [37.88, 37.86],
            "housing_median_age": [41, 21],
            "total_rooms": [880, 7099],
            "total_bedrooms": [129, 1106],
            "population": [322, 2401],
            "households": [126, 1138],
            "median_income": [8.3252, 8.3014],
            "median_house_value": [452600, 358500],
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
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41,
                "total_rooms": 880,
                "total_bedrooms": 129,
                "population": 322,
                "households": 126,
                "median_income": 8.3252,
            },
            {
                "longitude": -122.22,
                "latitude": 37.86,
                "housing_median_age": 21,
                "total_rooms": 7099,
                "total_bedrooms": 1106,
                "population": 2401,
                "households": 1138,
                "median_income": 8.3014,
            },
        ]
        response = self.client.post("/batch_predict", json={"data": data})
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())


if __name__ == "__main__":
    unittest.main()
