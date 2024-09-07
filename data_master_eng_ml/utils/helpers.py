import pycountry_convert as pc
import pycountry
from typing import List
import pandas as pd
import numpy as np

from data_master_eng_ml.utils.mappings import (
    plataform_mapping,
    player_perspectives_mapping,
    genres_mapping,
)


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


def map_perspectives(perspectives: List[int]) -> List[str]:
    """
    Mapeia perspectivas de jogadores para seus nomes correspondentes.

    Esta função converte uma lista de identificadores de perspectivas de jogadores
    em seus nomes correspondentes, usando um mapeamento predefinido. Se uma perspectiva
    não for encontrada no mapeamento, ela é marcada como 'unknown_player_perspectives'.

    Args:
        perspectives (List[int]): Lista de IDs de perspectivas de jogadores.

    Returns:
        List[str]: Lista de nomes de perspectivas de jogadores, sem duplicatas.
    """
    return list(
        set(
            [
                player_perspectives_mapping.get(i, "unknown_player_perspectives")
                for i in perspectives
            ]
        )
    )


def map_platforms(platforms: List[int]) -> List[str]:
    """
    Mapeia plataformas de jogos para seus nomes correspondentes.

    Esta função converte uma lista de identificadores de plataformas em seus nomes
    correspondentes, usando um mapeamento predefinido. Se uma plataforma não for
    encontrada no mapeamento, ela é marcada como 'unknown_platforms_name'.

    Args:
        platforms (List[int]): Lista de IDs de plataformas de jogos.

    Returns:
        List[str]: Lista de nomes de plataformas de jogos, sem duplicatas.
    """
    return list(set([plataform_mapping.get(i, "unknown_platforms_name") for i in platforms]))


def map_genres(genres: List[int]) -> str:
    """
    Mapeia gêneros de jogos para seus nomes correspondentes.

    Esta função converte uma lista de identificadores de gêneros de jogos em seus nomes
    correspondentes, usando um mapeamento predefinido. Se um gênero não for encontrado no
    mapeamento, ele é marcado como 'unknown_genres_name'. O resultado é o primeiro item
    da lista resultante.

    Args:
        genres (List[int]): Lista de IDs de gêneros de jogos.

    Returns:
        str: O nome do gênero de jogo correspondente ao primeiro item da lista mapeada.
    """
    return list(set([genres_mapping.get(i, "unknown_genres_name") for i in genres]))[0]


def map_game_modes(game_modes: List[int]) -> List[str]:
    """
    Mapeia modos de jogos para seus nomes correspondentes.

    Esta função converte uma lista de identificadores de modos de jogos em seus nomes
    correspondentes, usando um mapeamento predefinido. Se um modo de jogo não for encontrado
    no mapeamento, ele é marcado como 'unknown_game_mode'.

    Args:
        game_modes (List[int]): Lista de IDs de modos de jogos.

    Returns:
        List[str]: Lista de nomes de modos de jogos, sem duplicatas.
    """
    return list(set([game_modes_mapping.get(i, "unknown_game_mode") for i in game_modes]))


def process_game_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Processa dados de jogos, preenchendo valores nulos, mapeando colunas e adicionando novas informações.

    Esta função realiza o pré-processamento de um DataFrame contendo informações de jogos. Ela
    preenche valores nulos, aplica mapeamentos para transformar identificadores em nomes, adiciona
    novas colunas derivadas e remove colunas desnecessárias para análise futura.

    Args:
        data_frame (pd.DataFrame): DataFrame contendo os dados dos jogos a serem processados.
        Deve incluir as colunas 'player_perspectives', 'game_modes', 'genres', 'platforms', 'remasters',
        e 'rating'.

    Returns:
        pd.DataFrame: Um DataFrame processado com colunas transformadas e novas colunas para análise,
        como 'player_perspective_name', 'platforms_name', 'genres_first', 'game_modes_name',
        'has_remaster', e 'target'.
    """
    # Preenchendo valores nulos
    data_frame["player_perspectives"] = data_frame["player_perspectives"].fillna("Unknown")
    data_frame["game_modes"] = data_frame["game_modes"].fillna("unknown_game_mode")
    data_frame["genres"] = data_frame["genres"].fillna("unknown_genres_name")

    # Aplicando mapeamentos e criando novas colunas
    data_frame["player_perspective_name"] = data_frame["player_perspectives"].apply(
        map_perspectives
    )
    data_frame["platforms_name"] = data_frame["platforms"].apply(map_platforms)
    data_frame["genres_first"] = data_frame["genres"].apply(map_genres)
    data_frame["game_modes_name"] = data_frame["game_modes"].apply(map_game_modes)

    # Adicionando colunas derivadas
    data_frame["has_remaster"] = data_frame["remasters"].notna()
    data_frame["target"] = np.where(data_frame["rating"].isna(), 0, 1)

    # Removendo colunas desnecessárias
    data_frame = data_frame.drop(
        columns=[
            "game_modes",
            "player_perspectives",
            "remasters",
            "genres",
            "platforms",
            "involved_companies",
            "rating",
        ]
    )

    return data_frame


def map_age_classifications(df: pd.DataFrame, age_ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia as classificações etárias para o DataFrame principal.

    Esta função expande a coluna 'age_ratings' do DataFrame principal para ter uma linha por classificação,
    faz o merge com um DataFrame de classificações etárias, e determina a classificação etária máxima para
    cada jogo. Se não houver classificação disponível, define 'No Rating'.

    Args:
        df (pd.DataFrame): DataFrame principal que contém informações sobre os jogos, incluindo a coluna 'age_ratings'.
        age_ratings_df (pd.DataFrame): DataFrame contendo os IDs de classificações etárias e seus respectivos grupos.

    Returns:
        pd.DataFrame: O DataFrame principal atualizado com a coluna 'age_rating_group' mapeada, contendo a
        classificação etária máxima para cada jogo ou 'No Rating' se nenhuma estiver disponível.
    """
    # Explode a coluna 'age_ratings' para ter uma linha por classificação
    df_exploded = df.explode("age_ratings")

    # Converte 'age_ratings' para string para compatibilidade
    df_exploded["age_ratings"] = df_exploded["age_ratings"].astype(str)

    # Faz o merge com as classificações etárias
    df_exploded = df_exploded.merge(
        age_ratings_df[["id", "age_rating_group"]],
        left_on="age_ratings",
        right_on="id",
        how="left",
    )

    # Reagrupar para pegar a classificação máxima ou 'No Rating' se não houver
    df_grouped = (
        df_exploded.groupby("id_x")
        .agg(
            {
                "age_rating_group": lambda x: (
                    x.dropna().max() if len(x.dropna()) > 0 else "No Rating"
                )
            }
        )
        .reset_index()
    )

    # Renomeia a coluna de volta para 'id'
    df_grouped = df_grouped.rename(columns={"id_x": "id"})

    # Faz o merge de volta com o DataFrame original
    df = df.drop(columns="age_ratings").merge(df_grouped, on="id", how="left")

    return df


def generate_dummies(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Gera variáveis dummy para colunas específicas.

    Esta função cria variáveis dummy para as colunas especificadas de um DataFrame.
    Para cada coluna, a função expande os valores em múltiplas linhas, cria variáveis
    dummy para cada categoria distinta, e depois agrupa essas variáveis por ID, somando
    os valores de cada grupo.

    Args:
        df (pd.DataFrame): O DataFrame principal que contém as colunas para as quais as variáveis dummy serão geradas.
        columns (List[str]): Lista de nomes de colunas para as quais as variáveis dummy serão criadas.

    Returns:
        pd.DataFrame: O DataFrame atualizado com as variáveis dummy geradas para as colunas especificadas.
    """
    for column in columns:
        # Explode a coluna para ter uma linha por categoria
        df_exploded = df[["id", column]].explode(column)

        # Cria variáveis dummy e substitui caracteres "-" por "_"
        dummies = pd.get_dummies(df_exploded[column].str.replace("-", "_"))

        # Agrupa por 'id' somando as dummies
        dummies = dummies.groupby(df_exploded["id"]).sum().reset_index()

        # Remove a coluna original e adiciona as dummies
        df = df.drop(columns=[column]).merge(dummies, on="id")

    return df
