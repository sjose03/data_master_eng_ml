import pycountry_convert as pc
import pycountry


def ensure_columns(data_frame, columns):
    """
    Garante que todas as colunas estejam presentes no DataFrame.
    """
    for column in columns:
        if column not in data_frame.columns:
            data_frame[column] = None
    return data_frame


def country_to_continent(country_code):
    """
    Converte o código do país para o nome do continente.
    """
    try:
        country = pycountry.countries.get(numeric=str(country_code))
        continent_code = pc.country_alpha2_to_continent_code(country.alpha_2)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except:
        return "Unknown"


def array_count(value):
    """
    Conta o número de elementos em uma lista, retornando -1 se for 'Unknown'.
    """
    return -1 if value == "Unknown" else len(value)


import pandas as pd


def process_companies_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Processa dados de empresas, preenchendo valores nulos, adicionando novas colunas e removendo colunas desnecessárias.

    Esta função realiza o pré-processamento de um DataFrame contendo informações sobre empresas,
    preenchendo valores nulos, criando novas colunas derivadas e removendo colunas que não são
    mais necessárias para análises futuras.

    Args:
        data_frame (pd.DataFrame): DataFrame contendo os dados das empresas a serem processados.
        Deve incluir as colunas 'developed', 'published', 'country', 'parent', e 'start_date_category'.

    Returns:
        pd.DataFrame: Um DataFrame processado, contendo novas colunas para análises adicionais, como
        'games_developed', 'has_parents', 'games_published', e 'continent_name'.
    """
    # Preenchendo valores nulos
    data_frame["developed"] = data_frame["developed"].fillna("Unknown")
    data_frame["published"] = data_frame["published"].fillna("Unknown")
    data_frame["country"] = data_frame["country"].fillna(1).astype(int)

    # Adicionando novas colunas derivadas
    data_frame["games_developed"] = data_frame["developed"].apply(array_count)
    data_frame["has_parents"] = data_frame["parent"].notna().astype(int)
    data_frame["games_published"] = data_frame["published"].apply(array_count)
    data_frame["continent_name"] = data_frame["country"].apply(country_to_continent)

    # Removendo colunas desnecessárias
    data_frame = data_frame.drop(
        columns=["developed", "published", "country", "parent", "start_date_category"]
    )

    return data_frame
