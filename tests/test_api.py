import unittest
from fastapi.testclient import TestClient
import sys
import os

# Adicione o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.main import app


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_token_auth(self):
        response = self.client.post(
            "/token", data={"username": "test", "password": "test"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())
        self.assertIn("token_type", response.json())

    def test_protected_route(self):
        response = self.client.post(
            "/token", data={"username": "test", "password": "test"}
        )
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        response = self.client.post(
            "/predict",
            headers=headers,
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


if __name__ == "__main__":
    unittest.main()
