import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def create_features(data):
    # Remover colunas que não são necessárias para o modelo
    if "_id" in data.columns:
        data = data.drop(columns=["_id"])

    # poly_df = pd.DataFrame(poly_features, columns=unique_feature_names)

    # Combinar com as features originais
    # final_features = pd.concat([data.reset_index(drop=True), poly_df], axis=1)

    return data


def get_data_from_mongodb(uri, database_name, collection_name):
    from pymongo import MongoClient

    client = MongoClient(uri)
    db = client[database_name]
    collection = db[collection_name]
    data = pd.DataFrame(list(collection.find()))
    target = data.pop("target")  # Ajuste conforme sua coluna alvo
    return data, target


if __name__ == "__main__":
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")

    data, target = get_data_from_mongodb(MONGODB_URI, DATABASE_NAME, COLLECTION_NAME)
    features = create_features(data)
    print("Data shape:", features.shape)
    print("Target shape:", target.shape)
