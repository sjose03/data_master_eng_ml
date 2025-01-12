import unittest
from unittest.mock import patch, MagicMock
from pymongo.errors import ConnectionFailure
from data_master_eng_ml.db.mongodb_client import MongoDBClient
from data_master_eng_ml.config import MONGODB_DEFAULT_DATABASE


class TestMongoDBClientSingleton(unittest.TestCase):
    def setUp(self):
        """Configura um mock para MongoDBClient."""
        # Mock do MongoClient
        self.mock_client = MagicMock()

        # Mock do banco de dados
        self.mock_db = MagicMock()
        self.mock_client.__getitem__.return_value = self.mock_db

        MongoDBClient._instance = None

    def test_singleton_behavior(self):
        """Testa se MongoDBClient segue o padrão Singleton."""
        client1 = MongoDBClient(client=self.mock_client)
        client2 = MongoDBClient(client=self.mock_client)

        # Verifica se ambas as instâncias são iguais
        self.assertIs(client1, client2)

    def test_successful_connection(self):
        """Testa conexão bem-sucedida ao MongoDB."""
        # Simula o comando 'ping'
        self.mock_client.admin.command.return_value = {"ok": 1}

        # Inicializa o cliente
        client = MongoDBClient(client=self.mock_client)
        db = client.get_database()

        # Verifica se o banco retornado é o correto
        self.assertEqual(db, self.mock_db)

        # Verifica se o comando 'ping' foi chamado
        self.mock_client.admin.command.assert_called_once_with("ping")

    def test_create_and_validate_collection(self):
        """Testa criação e validação de uma coleção."""
        collection_name = "test_collection"
        client = MongoDBClient(client=self.mock_client)
        # Simula a criação de uma coleção
        self.mock_db.create_collection.return_value = None

        # Chama o método de criação
        client.get_database().create_collection(collection_name)

        # Verifica se o método foi chamado corretamente
        self.mock_db.create_collection.assert_called_once_with(collection_name)

    def test_insert_and_read_document(self):
        """Testa inserção e leitura de um documento."""
        collection_name = "test_collection"
        mock_collection = self.mock_db[collection_name]

        # Simula a inserção de um documento
        document = {"name": "John Doe", "age": 30}
        mock_collection.insert_one.return_value.inserted_id = "mock_id"

        # Insere o documento
        insert_result = mock_collection.insert_one(document)
        self.assertEqual(insert_result.inserted_id, "mock_id")

        # Simula a leitura do documento
        mock_collection.find_one.return_value = document
        found_document = mock_collection.find_one({"name": "John Doe"})
        self.assertEqual(found_document, document)

    def test_failed_connection(self):
        """Testa falha de conexão ao MongoDB."""
        # Simula uma falha no comando 'ping'
        self.mock_client.admin.command.side_effect = ConnectionFailure(
            "Connection failed"
        )

        with self.assertRaises(Exception) as context:
            MongoDBClient(client=self.mock_client)

        self.assertIn("Failed to connect to MongoDB", str(context.exception))


if __name__ == "__main__":
    unittest.main()
