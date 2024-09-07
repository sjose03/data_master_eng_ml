import pandas as pd

from data_master_eng_ml.utils.twitch_api import build_query, fetch_data_with_pagination
from data_master_eng_ml.config import URL_TWITCH_BASE
from data_master_eng_ml.utils.mappings import (
    region_mapping_inverted,
    plataform_mapping,
    player_perspectives_mapping,
    genres_mapping,
    game_modes_mapping,
    age_rating_mapping,
    age_order,
)
from data_master_eng_ml.utils.helpers import (
    process_companies_data,
    ensure_columns,
    process_game_data,
)
from typing import Union, List
import pandas as pd


def fetch_game_release_dates(
    single: bool, game_id: Union[int, List[int]] = 0, year: int = 2021
) -> pd.DataFrame:
    """
    Busca as datas de lançamento de jogos da API do IGDB.

    Esta função consulta a API do IGDB para obter datas de lançamento de jogos, com base em
    parâmetros específicos. Se o parâmetro `single` for verdadeiro, a função busca apenas a
    data de lançamento de um jogo específico. Caso contrário, ela retorna todas as datas de
    lançamento de jogos para um determinado ano e região.

    Args:
        single (bool): Indica se a busca é para um único jogo (`True`) ou múltiplos jogos (`False`).
        game_id (int, optional): ID do jogo a ser pesquisado. Usado apenas se `single` for `True`.
            O valor padrão é 0.
        year (int, optional): Ano das datas de lançamento a serem buscadas. Usado apenas se `single`
            for `False`. O valor padrão é 2021.

    Returns:
        pd.DataFrame: Um DataFrame contendo as datas de lançamento dos jogos, com informações
        adicionais como categoria, plataforma, status, região, etc.
    """
    url = f"{URL_TWITCH_BASE}/release_dates"
    fields = [
        "category",
        "created_at",
        "date",
        "game",
        "human",
        "platform",
        "region",
        "status",
        "updated_at",
        "y",
    ]

    if single:
        filters = {"game": f"= {game_id}"}
    else:
        filters = {"status": "= 6", "y": f"= {year}", "region": f"= 8"}

    data_frame_games_find = fetch_data_with_pagination(url, build_query, fields, filters)
    data_frame_games_find["region_name"] = data_frame_games_find["region"].map(
        region_mapping_inverted
    )

    return data_frame_games_find


def fetch_involved_companies(game_id: int) -> pd.DataFrame:
    """
    Busca as empresas envolvidas no desenvolvimento e publicação de um jogo na API do IGDB.

    Esta função faz uma consulta à API do IGDB para recuperar informações sobre as empresas que
    estiveram envolvidas no desenvolvimento ou na publicação de um jogo específico, identificado
    pelo seu ID.

    Args:
        game_id (int): O ID do jogo para o qual as informações de empresas envolvidas devem ser buscadas.

    Returns:
        pd.DataFrame: Um DataFrame contendo os dados das empresas envolvidas, incluindo detalhes
        como nomes, funções (desenvolvedora, publicadora, etc.), e outras informações relevantes.
    """
    url = f"{URL_TWITCH_BASE}/involved_companies"
    fields = ["*"]
    filters = {"game": f"= ({game_id})"}

    data_frame_companies_find = fetch_data_with_pagination(url, build_query, fields, filters)

    return data_frame_companies_find


def fetch_companies_info(company_id_list: List[int]) -> pd.DataFrame:
    """
    Busca informações detalhadas sobre as empresas envolvidas no desenvolvimento e publicação de jogos na API do IGDB.

    Esta função consulta a API do IGDB para recuperar informações específicas sobre uma lista de
    empresas, identificadas pelos seus IDs. Em seguida, processa os dados retornados para
    preencher valores nulos, adicionar colunas derivadas, e formatar o DataFrame para análise.

    Args:
        company_id_list (List[int]): Uma lista de IDs das empresas cujas informações devem ser buscadas.

    Returns:
        pd.DataFrame: Um DataFrame contendo informações detalhadas sobre as empresas, como
        jogos desenvolvidos e publicados, país de origem, datas de início de operação, e informações parentais.
    """
    url = f"{URL_TWITCH_BASE}/companies"
    fields = [
        "developed",
        "slug",
        "published",
        "country",
        "start_date",
        "start_date_category",
        "parent",
    ]
    filters = {"id": f"= ({','.join(map(str, company_id_list))})"}

    # Busca dados de empresas com paginação
    data_frame_companies = fetch_data_with_pagination(url, build_query, fields, filters)

    # Processa dados de empresas
    data_frame_companies = process_companies_data(data_frame_companies)

    return data_frame_companies


def fetch_multiplayer_modes(game_id: int) -> pd.DataFrame:
    """
    Busca os modos multiplayer disponíveis para um jogo específico na API do IGDB.

    Esta função consulta a API do IGDB para recuperar informações detalhadas sobre os modos multiplayer
    disponíveis para um jogo identificado pelo seu ID. Os dados incluem informações sobre modos cooperativos,
    número máximo de jogadores, e se o jogo suporta tela dividida.

    Args:
        game_id (int): O ID do jogo para o qual as informações de modos multiplayer devem ser buscadas.

    Returns:
        pd.DataFrame: Um DataFrame contendo as informações dos modos multiplayer, com colunas como
        'campaigncoop', 'lancoop', 'offlinecoop', 'offlinecoopmax', 'offlinemax', 'onlinecoop',
        'onlinecoopmax', 'onlinemax', e 'splitscreen'.
    """
    url = f"{URL_TWITCH_BASE}/multiplayer_modes"
    fields = [
        "campaigncoop",
        "game",
        "lancoop",
        "offlinecoop",
        "offlinecoopmax",
        "offlinemax",
        "onlinecoop",
        "onlinecoopmax",
        "onlinemax",
        "splitscreen",
    ]
    filters = {"game": f"= ({game_id})"}

    # Busca dados de modos multiplayer com paginação
    data_frame_multiplayer_modes = fetch_data_with_pagination(url, build_query, fields, filters)

    # Garante que todas as colunas esperadas estejam presentes no DataFrame
    data_frame_multiplayer_modes = ensure_columns(data_frame_multiplayer_modes, fields)

    return data_frame_multiplayer_modes


def fetch_game_info(game_id: int) -> pd.DataFrame:
    """
    Busca informações detalhadas de um jogo específico na API do IGDB.

    Esta função faz uma consulta à API do IGDB para recuperar informações detalhadas sobre um jogo,
    identificado pelo seu ID. As informações incluem modos de jogo, gêneros, classificações etárias,
    empresas envolvidas, perspectivas de jogador, plataformas, classificações, e remasterizações.

    Args:
        game_id (int): O ID do jogo para o qual as informações devem ser buscadas.

    Returns:
        pd.DataFrame: Um DataFrame contendo as informações detalhadas do jogo, com colunas como 'name',
        'game_modes', 'genres', 'age_ratings', 'involved_companies', 'player_perspectives',
        'platforms', 'rating', e 'remasters'.
    """
    url = f"{URL_TWITCH_BASE}/games"
    fields = [
        "name",
        "game_modes",
        "genres",
        "age_ratings",
        "involved_companies",
        "player_perspectives",
        "platforms",
        "rating",
        "remasters",
    ]
    filters = {"id": f"= ({game_id})"}

    # Busca dados do jogo com paginação
    data_frame_games = fetch_data_with_pagination(url, build_query, fields, filters)

    # Garante que todas as colunas esperadas estejam presentes no DataFrame
    data_frame_games = ensure_columns(data_frame_games, fields)

    # Processa dados do jogo para formatação adicional
    data_frame_games = process_game_data(data_frame_games)

    return data_frame_games


def batch_fetch_age_classifications(age_ratings_list: List[int]) -> pd.DataFrame:
    """
    Busca classificações etárias em lotes para reduzir o número de chamadas à API.

    Esta função busca classificações etárias em lotes usando a API do IGDB para evitar múltiplas
    chamadas desnecessárias. O resultado é um DataFrame que mapeia cada ID de classificação etária
    para seu grupo correspondente.

    Args:
        age_ratings_list (List[int]): Lista de IDs de classificações etárias associadas aos jogos.

    Returns:
        pd.DataFrame: Um DataFrame contendo os IDs de classificação etária e seus respectivos grupos,
        ou um DataFrame vazio se a lista de IDs for vazia.
    """
    # Filtra para apenas os IDs únicos
    unique_age_ratings = list(set(age_ratings_list))

    if not unique_age_ratings:
        return pd.DataFrame(columns=["id", "age_rating_group"])

    # Monta a consulta em lote
    url = f"{URL_TWITCH_BASE}/age_ratings"
    fields = ["id", "rating"]
    filters = {"id": f"= ({','.join(map(str, unique_age_ratings))})"}

    # Busca dados de classificações etárias com paginação
    data_frame = fetch_data_with_pagination(url, build_query, fields, filters)

    # Aplica o mapeamento de classificação etária
    data_frame["age_rating_group"] = data_frame["rating"].map(age_rating_mapping)
    data_frame["age_rating_group"] = data_frame["age_rating_group"].astype(age_order)

    # Converte o campo 'id' para string para garantir consistência
    data_frame["id"] = data_frame["id"].astype(str)

    return data_frame
