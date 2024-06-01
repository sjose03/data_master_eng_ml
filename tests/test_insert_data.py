import unittest
from unittest.mock import patch
from data.insert_data import insert_data
from pymongo.collection import Collection
from dotenv import load_dotenv


class TestInsertData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()

    @patch("data.insert_data.MongoClient")
    def test_insert_data(self, MockMongoClient):
        # Mock the MongoClient and its behavior
        mock_client = MockMongoClient.return_value
        mock_db = mock_client[os.getenv("DATABASE_NAME")]
        mock_collection = mock_db[os.getenv("COLLECTION_NAME")]

        # Mock the insert_many method
        mock_collection.insert_many.return_value.inserted_ids = [1, 2, 3]

        insert_data(os.getenv("DATABASE_NAME"), os.getenv("COLLECTION_NAME"))

        # Assert that insert_many was called once
        mock_collection.insert_many.assert_called_once()
        # Assert that the collection now contains documents
        self.assertEqual(len(mock_collection.insert_many.return_value.inserted_ids), 3)


if __name__ == "__main__":
    unittest.main()
