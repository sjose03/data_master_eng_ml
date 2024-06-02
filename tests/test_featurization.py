import unittest
import pandas as pd
import sys
import os

# Adicione o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.featurization import create_features
from dotenv import load_dotenv


class TestFeaturization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()

    def test_create_features(self):
        data = pd.DataFrame(
            {
                "longitude": [-122.23, -122.22],
                "latitude": [37.88, 37.86],
                "housing_median_age": [41, 21],
                "AveRooms": [880, 7099],
                "AveBedrms": [129, 1106],
                "Population": [322, 2401],
                "AveOccup": [126, 1138],
                "median_income": [8.3252, 8.3014],
                "median_house_value": [452600, 358500],
            }
        )
        features = create_features(data)
        self.assertIn("rooms_per_household", features.columns)
        self.assertIn("bedrooms_per_room", features.columns)
        self.assertIn("population_per_household", features.columns)


if __name__ == "__main__":
    unittest.main()
