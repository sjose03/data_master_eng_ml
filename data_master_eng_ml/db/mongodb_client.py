from pymongo.errors import ConnectionFailure
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import typer
from loguru import logger


class MongoDBClient:
    _instance = None

    def __new__(cls, uri="mongodb://localhost:27017", db_name="default_db"):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(uri, db_name)
        return cls._instance

    def _initialize(self, uri, db_name):
        self._uri = uri
        self._db_name = db_name
        self._client = MongoClient(self._uri, server_api=ServerApi("1"))
        try:
            self._client.admin.command(
                "ping"
            )  # Verifica se o MongoDB está acessível
            logger.debug("Conexão bem-sucedida ao MongoDB!")
        except ConnectionFailure:
            logger.error(f"Erro ao conectar ao MongoDB: at {self._uri}")
            raise Exception(f"Failed to connect to MongoDB at {self._uri}")

        self._db = self._client[self._db_name]

    def get_database(self):
        return self._db

    def close_connection(self):
        if self._client:
            self._client.close()
            print("MongoDB connection closed.")
