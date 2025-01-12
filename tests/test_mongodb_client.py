import unittest
from unittest.mock import patch, MagicMock
from pymongo.errors import ConnectionFailure
from data_master_eng_ml.db import MongoDBClient
from data_master_eng_ml.config import MONGODB_DEFAULT_DATABASE


class TestMongoDBClient(unittest.TestCase):
    def setUp(self):
        """Configuração inicial antes de cada teste."""
        # Cria o mock para MongoClient
        self.patcher = patch("data_master_eng_ml.db.MongoDBClient")
        self.mock_mongo_client = self.patcher.start()

        # Mock do cliente e do banco de dados
        self.mock_client_instance = MagicMock()
        self.mock_mongo_client.return_value = self.mock_client_instance
        self.mock_db = self.mock_client_instance[
            self.mock_client_instance.db_name
        ]
        self.mock_client_instance.__getitem__.return_value = self.mock_db

        self.db_name = MONGODB_DEFAULT_DATABASE
        self.client = MongoDBClient()
        self.db = self.client.get_database()

    def tearDown(self):
        """Finaliza o mock após cada teste."""
        self.patcher.stop()

    def test_successful_connection(self):
        """Testa conexão bem-sucedida ao MongoDB."""
        # Simula o comando 'ping' no banco
        self.mock_client_instance.admin.command.return_value = {"ok": 1}

        # Verifica se o banco retornado é o correto
        db = self.client.get_database()
        self.assertEqual(db.name, self.db_name)

        # Valida que o comando de 'ping' foi chamado
        self.mock_client_instance.admin.command.assert_called_once_with("ping")

    def test_create_and_validate_collection(self):
        """Testa a criação de uma coleção e sua validação."""
        collection_name = "test_collection"

        # Cria uma coleção
        self.db.create_collection(collection_name)

        # Verifica se a coleção foi criada
        collections = self.db.list_collection_names()
        self.assertIn(collection_name, collections)

    def test_insert_and_read_document(self):
        """Testa a inserção e leitura de um documento em uma coleção."""
        collection_name = "test_collection"
        collection = self.db[collection_name]

        # Insere um documento
        document = {"name": "John Doe", "age": 30}
        insert_result = collection.insert_one(document)
        self.assertIsNotNone(insert_result.inserted_id)

        # Lê o documento inserido
        found_document = collection.find_one({"name": "John Doe"})
        self.assertIsNotNone(found_document)
        self.assertEqual(found_document["name"], "John Doe")
        self.assertEqual(found_document["age"], 30)

    def test_failed_connection(self):
        """Testa falha de conexão ao MongoDB."""
        with self.assertRaises(ConnectionFailure):
            # Tenta criar um cliente com URI inválido
            MongoDBClient(uri="mongodb://invalid:27017", db_name="test_db")

    def tearDown(self):
        """Limpa o banco de dados após cada teste."""
        self.db.client.drop_database(self.db_name)


if __name__ == "__main__":
    unittest.main()
