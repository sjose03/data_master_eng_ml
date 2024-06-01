import unittest
from unittest.mock import patch, MagicMock
import os
import joblib
from models.train import train_model
from data.featurization import get_data_from_mongodb, create_features
from dotenv import load_dotenv


class TestTrainModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()

    @patch("comet_ml.Experiment")
    @patch("data.featurization.get_data_from_mongodb")
    def test_train_model(self, mock_get_data, MockExperiment):
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

        # Temporarily set DISABLE_COMET_LOGGING to true for the test
        os.environ["DISABLE_COMET_LOGGING"] = "true"

        features = create_features(mock_data)
        model = train_model(features, mock_target)
        self.assertIsNotNone(model)
        joblib.dump(model, "model.pkl")
        loaded_model = joblib.load("model.pkl")
        self.assertIsNotNone(loaded_model)

        # Restore environment variable
        del os.environ["DISABLE_COMET_LOGGING"]


if __name__ == "__main__":
    unittest.main()
