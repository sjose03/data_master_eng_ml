import pymongo
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def get_data_from_mongodb(uri, db_name, collection_name):
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]

    data = pd.DataFrame(list(collection.find()))

    target = data["target"]
    features = data.drop(columns=["target"])

    return features, target


def create_features(data):
    # Remover colunas que não são necessárias para o modelo
    if "_id" in data.columns:
        data = data.drop(columns=["_id"])

    data["RoomsPerHouse"] = data["AveRooms"] / data["AveOccup"]
    data["BedroomsPerHouse"] = data["AveBedrms"] / data["AveRooms"]
    data["PopulationPerHouse"] = data["Population"] / data["HouseAge"]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(data)
    poly_feature_names = poly.get_feature_names_out(data.columns)
    data = pd.DataFrame(poly_features, columns=poly_feature_names)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data)
    return features_scaled


if __name__ == "__main__":
    load_dotenv()
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")

    data, target = get_data_from_mongodb(MONGODB_URI, DATABASE_NAME, COLLECTION_NAME)
    features = create_features(data)
    print("Data shape:", features.shape)
    print("Target shape:", target.shape)
