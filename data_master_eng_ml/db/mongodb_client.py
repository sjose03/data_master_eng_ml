from pymongo.errors import ConnectionFailure
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from data_master_eng_ml.config import MONGODB_URI, MONGODB_DEFAULT_DATABASE
import typer
from loguru import logger


class MongoDBClient:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(
        self,
    ):
        self._uri = MONGODB_URI
        self._db_name = MONGODB_DEFAULT_DATABASE
        self._client = MongoClient(self._uri)
        try:
            self._client.admin.command(
                "ping"
            )  # Verifica se o MongoDB está acessível
            logger.success(f"Connected to MongoDB at {self._uri}")
        except ConnectionFailure:
            logger.error(f"Failed to connect to MongoDB at {self._uri}")
            raise Exception(f"Failed to connect to MongoDB at {self._uri}")
        self._db = self._client[self._db_name]

    def get_database(self):
        return self._db

    def close_connection(self):
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed.")
