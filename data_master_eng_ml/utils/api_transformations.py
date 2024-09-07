import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pycountry_convert as pc
import pycountry
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
from data_master_eng_ml.utils.helpers import process_companies_data
from typing import Union, List
import pandas as pd


def fetch_game_release_dates(single: bool, game_id: int = 0, year: int = 2021) -> pd.DataFrame:
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
