import unittest
import pandas as pd
from pymongo import MongoClient
from featurization import create_features
from insert_data import fetch_and_prepare_data, insert_data_to_mongodb

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.mongodb_uri = "your-mongodb-atlas-uri"
        self.database_name = "test_database"
        self.collection_name = "test_collection"

        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[self.database_name]
        self.db[self.collection_name].delete_many({})

    def tearDown(self):
        self.db[self.collection_name].delete_many({})
        self.client.close()

    def test_insert_data_to_mongodb(self):
        data = fetch_and_prepare_data()
        insert_data_to_mongodb(data, self.mongodb_uri, self.database_name, self.collection_name)
        
        collection = self.db[self.collection_name]
        count = collection.count_documents({})
        self.assertEqual(count, len(data))

    def test_create_features(self):
        data = {
            'AveRooms': [6.0, 8.0],
            'AveOccup': [3.0, 2.0],
            'AveBedrms': [1.0, 2.0],
            'Population': [1500, 3000],
            'HouseAge': [20, 30]
        }
        df = pd.DataFrame(data)
        
        features = create_features(df)
        
        self.assertIn('RoomsPerHouse', features.columns)
        self.assertIn('BedroomsPerHouse', features.columns)
        self.assertIn('PopulationPerHouse', features.columns)

if __name__ == '__main__':
    unittest.main()
