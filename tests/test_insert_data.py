import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Adicione o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.insert_data import fetch_and_prepare_data, insert_data_to_mongodb
from dotenv import load_dotenv


class TestInsertData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()

    @patch("data.insert_data.pymongo.MongoClient")
    def test_insert_data(self, MockMongoClient):
        # Mock the MongoClient and its behavior
        mock_client = MagicMock()
        MockMongoClient.return_value = mock_client
        mock_db = mock_client[os.getenv("DATABASE_NAME")]
        mock_collection = mock_db[os.getenv("COLLECTION_NAME")]

        # Dados de teste
        data = fetch_and_prepare_data()
        mock_uri = os.getenv("MONGODB_URI")

        # Mock the insert_many method
        mock_collection.insert_many.return_value.inserted_ids = [1, 2, 3]

        insert_data_to_mongodb(
            data, mock_uri, mock_db, mock_collection, mongo_client=mock_client
        )

        # Verificar se insert_many foi chamado uma vez
        self.assertEqual(mock_collection.insert_many.call_count, 1)

        # Verificar se insert_many foi chamado com os dados corretos
        args, kwargs = mock_collection.insert_many.call_args
        inserted_data = args[0]
        self.assertEqual(inserted_data, data.to_dict("records"))


if __name__ == "__main__":
    unittest.main()
