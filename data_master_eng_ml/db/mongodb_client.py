from pymongo.errors import ConnectionFailure
from pymongo.mongo_client import MongoClient

from data_master_eng_ml.config import MONGODB_URI, MONGODB_DEFAULT_DATABASE
from loguru import logger


class MongoDBClient:
    _instance = None  # Atributo de classe para armazenar a única instância

    def __new__(cls, *args, **kwargs):
        """Implementa o padrão Singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, client=None):
        """Inicializa o cliente MongoDB."""
        if client != None:
            logger.info("cliente fornecido")
        if not hasattr(
            self, "_initialized"
        ):  # Garante que o init seja executado uma única vez
            self._uri = MONGODB_URI  # URI interna padrão
            self._db_name = MONGODB_DEFAULT_DATABASE  # Nome do banco padrão
            self._client = client or MongoClient(self._uri)
            self._db = self._client[self._db_name]

            # Tenta se conectar ao MongoDB
            try:
                self._client.admin.command("ping")
                logger.info(
                    f"Connected to MongoDB at {self._uri}, database: {self._db_name}"
                )
            except ConnectionFailure as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise Exception(f"Failed to connect to MongoDB: {e}")

            self._initialized = True  # Marca como inicializado

    def get_database(self):
        """Retorna o banco de dados configurado."""
        return self._db

    def close_connection(self):
        """Fecha a conexão com o MongoDB."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed.")
