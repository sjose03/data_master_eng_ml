import os
import time
import requests
from dotenv import load_dotenv
from typing import Dict, Optional

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Variáveis de ambiente
TWITCH_ID = os.getenv("TWITCH_ID")
TWITCH_SECRET = os.getenv("TWITCH_SECRET")
URL_TOKEN = "https://id.twitch.tv/oauth2/token"

# Variáveis globais para armazenar o token e seu tempo de expiração
token_data: Optional[Dict] = None
token_expiration_time: float = 0


def get_token() -> None:
    """Obtém um novo token do Twitch e atualiza o tempo de expiração."""
    global token_data, token_expiration_time
    try:
        token_response = requests.post(
            url=f"{URL_TOKEN}?client_id={TWITCH_ID}&client_secret={TWITCH_SECRET}&grant_type=client_credentials"
        )

        if token_response.status_code == 200:
            token_data = token_response.json()
            token_expiration_time = time.time() + token_data["expires_in"]
            print("Novo token obtido com sucesso:", token_data)
        else:
            print(
                f"Erro ao obter o token: {token_response.status_code} - {token_response.text}"
            )
            token_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Ocorreu um erro na requisição: {e}")


def is_token_expired() -> bool:
    """Verifica se o token expirou."""
    return time.time() >= token_expiration_time


def get_valid_token() -> str:
    """Retorna um token válido, obtendo um novo se o atual tiver expirado."""
    if token_data is None or is_token_expired():
        print("Token expirado ou inexistente. Obtendo um novo...")
        get_token()
    return token_data["access_token"]


def make_authenticated_request(url: str, data: Dict) -> requests.Response:
    """Faz uma requisição autenticada usando o token válido."""
    token = get_valid_token()
    headers = {"Authorization": f"Bearer {token}", "Client-Id": TWITCH_ID}
    response = requests.post(url, headers=headers, data=data)
    if (
        response.status_code == 401
    ):  # Unauthorized, o token pode ter expirado ou ser inválido
        print("Token inválido, obtendo um novo token...")
        get_token()
        headers["Authorization"] = f"Bearer {get_valid_token()}"
        response = requests.post(url, headers=headers, data=data)
    return response
