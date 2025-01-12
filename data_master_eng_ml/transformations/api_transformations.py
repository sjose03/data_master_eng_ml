import pandas as pd

from data_master_eng_ml.utils.twitch_api import build_query, fetch_data_with_pagination
from data_master_eng_ml.config import URL_TWITCH_BASE
from data_master_eng_ml.utils.mappings import (
    age_rating_mapping,
    age_order,
)

from typing import List


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
