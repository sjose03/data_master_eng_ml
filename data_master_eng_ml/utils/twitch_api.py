import requests
import pandas as pd
from typing import List, Dict, Optional

from utils.auth_twitch import make_authenticated_request


def split_filters(filters: Dict[str, str], max_options: int) -> List[Dict[str, str]]:
    """
    Divide os filtros em várias partes, se necessário.

    Args:
        filters (Dict[str, str]): Dicionário de filtros a serem aplicados na consulta.
        max_options (int): Número máximo de opções por filtro.

    Returns:
        List[Dict[str, str]]: Lista de dicionários de filtros divididos.
    """
    split_filters_list = []
    for key, value in filters.items():
        options = value.split(",")
        for i in range(0, len(options), max_options):
            sub_filter = {key: ",".join(options[i : i + max_options])}
            split_filters_list.append(sub_filter)
    return split_filters_list


def build_query(
    fields: List[str], filters: Optional[Dict[str, str]], limit: int, offset: int
) -> str:
    """
    Constrói a query com base nos campos selecionados e filtros fornecidos.

    Args:
        fields (List[str]): Lista de campos a serem selecionados na consulta.
        filters (Optional[Dict[str, str]]): Dicionário de filtros a serem aplicados na consulta.
        limit (int): Número máximo de registros a serem retornados.
        offset (int): Offset para a consulta, usado para paginação.

    Returns:
        str: Query SQL-like para a API.
    """
    fields_str = ",".join(fields)
    where_clause = (
        f"where {' & '.join([f'{key} {value}' for key, value in filters.items()])};"
        if filters
        else ""
    )
    return f"fields {fields_str}; {where_clause} limit {limit}; offset {offset};"


def fetch_data_with_pagination(
    url: str,
    query_builder,
    fields: List[str],
    filters: Optional[Dict[str, str]] = None,
    max_filter_options: int = 1,
) -> pd.DataFrame:
    """
    Busca dados com paginação e retorna um DataFrame consolidado.

    Args:
        url (str): URL do endpoint da API.
        query_builder (function): Função responsável por construir a query.
        fields (List[str]): Lista de campos a serem selecionados.
        filters (Optional[Dict[str, str]]): Dicionário de filtros a serem aplicados na consulta.
        max_filter_options (int): Número máximo de opções por filtro antes de dividir a consulta.

    Returns:
        pd.DataFrame: DataFrame contendo todos os registros obtidos pela consulta com paginação.
    """
    all_data = []

    # Dividindo filtros se necessário
    if filters:
        filter_combinations = split_filters(filters, max_filter_options)
    else:
        filter_combinations = [filters]

    for sub_filters in filter_combinations:
        offset, limit = 0, 500
        while True:
            query = query_builder(fields, sub_filters, limit, offset)
            response = make_authenticated_request(url, query)
            if response.status_code != 200:
                print(f"Erro ao obter dados: {response.status_code} - {response.text}")
                break
            data = response.json()
            all_data.extend(data)
            total_count = int(response.headers.get("x-count", 0))
            if offset + limit >= total_count:
                break
            offset += limit

    return pd.DataFrame(all_data)
