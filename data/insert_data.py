import pymongo
import pandas as pd
from sklearn.datasets import fetch_california_housing
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

print(MONGODB_URI, DATABASE_NAME, COLLECTION_NAME)


def fetch_and_prepare_data():
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data["target"] = housing.target
    return data


def insert_data_to_mongodb(data, uri, db_name, collection_name):
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    data_dict = data.to_dict("records")
    collection.insert_many(data_dict)
    client.close()


if __name__ == "__main__":
    load_dotenv()
    data = fetch_and_prepare_data()
    insert_data_to_mongodb(data, MONGODB_URI, DATABASE_NAME, COLLECTION_NAME)
    print("Dados inseridos no MongoDB com sucesso!")
