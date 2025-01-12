import unittest
from unittest.mock import patch, MagicMock
import time
from requests.exceptions import RequestException
from data_master_eng_ml.utils.auth_igdb import (
    IGDBAuthenticatedClient,
    Token,
    TokenRequestException,
)


class TestIGDBAuth(unittest.TestCase):

    def setUp(self):
        """Configura um cliente com valores padrão antes de cada teste."""
        self.client = IGDBAuthenticatedClient(
            "test_id", "test_secret", "http://mock_url"
        )
        # Configura um token falso como se já tivesse sido obtido
        self.client.token = Token(
            access_token="mock_access_token",
            expires_in=3600,
            token_type="bearer",
            expiration_time=9999999999,  # Token válido por muito tempo
        )

    @patch("requests.post")
    def test_get_token_success(self, mock_post):
        # Mock da resposta da API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "mock_access_token",
            "expires_in": 3600,
            "token_type": "bearer",
        }
        mock_post.return_value = mock_response

        self.client.get_token()

        self.assertIsNotNone(self.client.token)
        self.assertEqual(self.client.token.access_token, "mock_access_token")

    @patch("requests.post")
    def test_get_token_error_response(self, mock_post):
        """Valida erro ao obter token devido a status de erro."""
        # Mock de uma resposta com erro (status 400)
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Invalid client credentials"
        mock_post.return_value = mock_response

        with self.assertRaises(TokenRequestException) as context:
            self.client.get_token()

        self.assertIn("Erro ao obter o token", str(context.exception))
        self.assertIn("400", str(context.exception))

    @patch("requests.post")
    def test_get_token_request_exception(self, mock_post):
        """Valida erro de exceção de requisição ao obter token."""
        # Simula uma exceção de requisição (exemplo: timeout)
        mock_post.side_effect = RequestException("Timeout occurred")

        with self.assertRaises(TokenRequestException) as context:
            self.client.get_token()

        self.assertIn("Erro na requisição", str(context.exception))

    @patch("requests.post")
    def test_make_authenticated_request_unauthorized(self, mock_post):
        """Valida erro ao realizar requisição autenticada com token inválido."""
        # Mock de uma resposta inicial com erro 401 (Unauthorized)
        mock_response_401 = MagicMock()
        mock_response_401.status_code = 401
        mock_response_401.text = "Unauthorized"

        # Mock de uma resposta com sucesso após renovação de token
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"data": "success"}

        # Simula respostas em sequência
        mock_post.side_effect = [mock_response_401, mock_response_success]

        # Mock do token válido para teste
        with patch.object(self.client, "get_token") as mock_get_token:
            mock_get_token.return_value = None  # Simula renovação do token
            response = self.client.make_authenticated_request(
                "http://mock_url", {"key": "value"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"data": "success"})

    @patch("requests.post")
    def test_make_authenticated_request_error_response(self, mock_post):
        """Valida erro ao realizar requisição autenticada devido a status de erro."""
        # Mock de uma resposta com erro 500
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with self.assertRaises(TokenRequestException) as context:
            self.client.make_authenticated_request(
                "http://mock_url", {"key": "value"}
            )

        self.assertIn("Erro na requisição autenticada", str(context.exception))
        self.assertIn("500", str(context.exception))


if __name__ == "__main__":
    unittest.main()
