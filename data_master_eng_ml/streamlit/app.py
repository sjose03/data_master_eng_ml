import streamlit as st
import requests
import os
import time
from dotenv import load_dotenv
from typing import Dict, Optional
from datetime import datetime

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Variáveis de ambiente para autenticação na API do Twitch
CLIENT_ID = os.getenv("TWITCH_ID")
TWITCH_SECRET = os.getenv("TWITCH_SECRET")
URL_TOKEN = "https://id.twitch.tv/oauth2/token"

# Variáveis globais para armazenar o token e o tempo de expiração
token_data: Optional[Dict] = None
token_expiration_time: float = 0


def get_token() -> None:
    """Obtém um novo token da API do Twitch e atualiza o tempo de expiração."""
    global token_data, token_expiration_time
    try:
        # Requisição POST para obter o token de acesso
        token_response = requests.post(
            url=f"{URL_TOKEN}?client_id={CLIENT_ID}&client_secret={TWITCH_SECRET}&grant_type=client_credentials"
        )

        if token_response.status_code == 200:
            # Armazena o token e calcula o tempo de expiração
            token_data = token_response.json()
            token_expiration_time = time.time() + token_data["expires_in"]
            print("Novo token obtido com sucesso:", token_data)
        else:
            print(f"Erro ao obter o token: {token_response.status_code} - {token_response.text}")
            token_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Ocorreu um erro na requisição: {e}")


def is_token_expired() -> bool:
    """Verifica se o token de acesso expirou."""
    return time.time() >= token_expiration_time


def get_valid_token() -> str:
    """Retorna um token válido, obtendo um novo se o atual tiver expirado."""
    if token_data is None or is_token_expired():
        print("Token expirado ou inexistente. Obtendo um novo...")
        get_token()
    return token_data["access_token"]


# Inicializa o token de acesso e a URL base da API do IGDB
ACCESS_TOKEN = get_valid_token()
BASE_URL = "https://api.igdb.com/v4"


def get_games_by_genre(genre_id):
    """
    Obtém uma lista de jogos de um gênero específico lançados a partir de 2023.

    Args:
        genre_id (int): ID do gênero para busca.

    Returns:
        list: Lista de jogos filtrados.
    """
    headers = {"Client-ID": CLIENT_ID, "Authorization": f"Bearer {ACCESS_TOKEN}"}
    timestamp_2023 = int(datetime(2023, 1, 1).timestamp())  # Timestamp para 1º de janeiro de 2023
    # Consulta para obter jogos lançados a partir de 2023
    body = f"fields name,cover.url,rating,first_release_date; where genres = {genre_id} & first_release_date >= {timestamp_2023}; limit 10;"

    response = requests.post(f"{BASE_URL}/games", headers=headers, data=body)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erro na API: {response.status_code}")
        return []


def get_genres():
    """
    Obtém todos os gêneros de jogos disponíveis.

    Returns:
        list: Lista de gêneros de jogos.
    """
    headers = {"Client-ID": CLIENT_ID, "Authorization": f"Bearer {ACCESS_TOKEN}"}
    body = "fields name; limit 50;"

    response = requests.post(f"{BASE_URL}/genres", headers=headers, data=body)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erro na API: {response.status_code}")
        return []


def send_game_id(game_id):
    """
    Envia o ID do jogo selecionado para outro serviço.

    Args:
        game_id (int): ID do jogo a ser enviado.
    """
    other_service_url = "https://outro-servico.com/api"  # URL do outro serviço
    payload = {"game_id": game_id}
    response = requests.post(other_service_url, json=payload)

    if response.status_code == 200:
        st.success("Game ID enviado com sucesso!")
    else:
        st.error(f"Erro ao enviar o Game ID: {response.status_code}")


# Interface do Streamlit
st.title("Front Data Master")

# Texto explicativo sobre como os dados são buscados
st.write(
    """
    **Front Data Master** é um aplicativo que busca informações sobre jogos de videogame da API do IGDB (Internet Game Database). 
    Você pode selecionar um gênero de jogo e visualizar uma lista de jogos lançados a partir de 2023. 
    Os dados são obtidos em tempo real utilizando a API do IGDB, que requer autenticação através da API do Twitch.
"""
)

# Obter lista de gêneros
genres = get_genres()
genre_options = {genre["name"]: genre["id"] for genre in genres}

# Seleção de gênero
selected_genre_name = st.selectbox("Escolha um gênero", list(genre_options.keys()))
selected_genre_id = genre_options[selected_genre_name]

# Consultar API por jogos do gênero selecionado
games = get_games_by_genre(selected_genre_id)

# Exibir tabela de jogos com botões de seleção
if games:
    st.write("### Jogos Disponíveis")

    # Criar um dicionário para mapear o nome do jogo ao ID
    game_options = {game["name"]: game["id"] for game in games}

    # Exibir jogos com imagens e permitir seleção
    selected_game_name = st.selectbox("Escolha um jogo", list(game_options.keys()))
    selected_game_id = game_options[selected_game_name]

    # Mostrar detalhes do jogo selecionado
    for game in games:
        if game["id"] == selected_game_id:
            release_date = (
                datetime.utcfromtimestamp(game["first_release_date"]).strftime("%d-%m-%Y")
                if "first_release_date" in game
                else "Data de lançamento desconhecida"
            )
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h2 style="font-size: 24px;">Nome: {game['name']}</h2>
                    <h3 style="font-size: 20px;">Rating: {round(game.get('rating', 0), 2)}</h3>
                    <h4>Data de Lançamento: {release_date}</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "cover" in game and game["cover"]:
                game_cover_url = f"https:{game['cover']['url'].replace('t_thumb', 't_cover_big')}"
                st.markdown(
                    f'<div style="text-align: center;"><img src="{game_cover_url}" width="300"></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.write("Imagem não disponível")

            st.write("---")

    # Botão para enviar o ID do jogo selecionado
    if st.button("Enviar ID do Jogo"):
        print(selected_game_id)
        # send_game_id(selected_game_id)

else:
    st.write("Nenhum jogo encontrado para este gênero.")
