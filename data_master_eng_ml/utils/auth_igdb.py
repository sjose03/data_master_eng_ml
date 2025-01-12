import time
import requests
from typing import Dict, Optional
from data_master_eng_ml.config import URL_TOKEN, TWITCH_ID, TWITCH_SECRET
from loguru import logger


from dataclasses import dataclass


@dataclass
class Token:
    access_token: str
    expires_in: int
    token_type: str
    expiration_time: float

    @classmethod
    def from_response(cls, response: Dict) -> "Token":
        """Cria um Token a partir de uma resposta JSON."""
        return cls(
            access_token=response["access_token"],
            expires_in=response["expires_in"],
            token_type=response["token_type"],
            expiration_time=time.time() + response["expires_in"],
        )

    def is_expired(self) -> bool:
        """Verifica se o token expirou."""
        return time.time() >= self.expiration_time


class TokenException(Exception):
    """Exceção base para erros relacionados ao token."""


class TokenRequestException(TokenException):
    """Erro ao obter o token."""


class IGDBAuthenticatedClient:
    def __init__(
        self,
        client_id: str = TWITCH_ID,
        client_secret: str = TWITCH_SECRET,
        token_url: str = URL_TOKEN,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.token: Optional[Token] = None

    def get_token(self) -> None:
        """Obtém um novo token e o armazena."""
        try:
            response = requests.post(
                url=f"{self.token_url}?client_id={self.client_id}&client_secret={self.client_secret}&grant_type=client_credentials"
            )
            if response.status_code == 200:
                self.token = Token.from_response(response.json())
                logger.info("Novo token obtido com sucesso.")
            else:
                logger.error(
                    f"Erro ao obter o token: {response.status_code} - {response.text}"
                )
                raise TokenRequestException(
                    f"Erro ao obter o token: {response.status_code} - {response.text}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao realizar a requisição do token: {e}")
            raise TokenRequestException(f"Erro na requisição: {e}")

    def get_valid_token(self) -> str:
        """Retorna um token válido, obtendo um novo se necessário."""
        if not self.token or self.token.is_expired():
            logger.info("Token expirado ou inexistente. Obtendo um novo...")
            self.get_token()
        return self.token.access_token

    def make_authenticated_request(
        self, url: str, data: Dict
    ) -> requests.Response:
        """Faz uma requisição autenticada."""
        try:
            headers = self._get_headers()
            response = requests.post(url, headers=headers, data=data)

            if response.status_code == 401:  # Unauthorized
                logger.warning("Token inválido, obtendo um novo token...")
                self.get_token()
                headers = self._get_headers()
                response = requests.post(url, headers=headers, data=data)
                if response.status_code == 200:
                    response.raise_for_status()
                    return response
                else:
                    logger.error(
                        f"Erro na requisição autenticada: {response.status_code} - {response.text}"
                    )
                raise TokenRequestException(
                    f"Erro na requisição autenticada: {response.status_code} - {response.text}"
                )
            elif response.status_code == 200:
                response.raise_for_status()
                return response
            else:
                logger.error(
                    f"Erro na requisição autenticada: {response.status_code} - {response.text}"
                )
                raise TokenRequestException(
                    f"Erro na requisição autenticada: {response.status_code} - {response.text}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro na requisição autenticada: {e}")
            raise TokenRequestException(f"Erro na requisição autenticada: {e}")

    def _get_headers(self) -> Dict[str, str]:
        """Gera cabeçalhos de autenticação."""
        return {
            "Authorization": f"Bearer {self.get_valid_token()}",
            "Client-Id": self.client_id,
        }
