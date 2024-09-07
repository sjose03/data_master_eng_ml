import os
from dotenv import load_dotenv
import pandas as pd
import requests
import numpy as np

# Carregando as variáveis de ambiente do arquivo .env
load_dotenv()


from data_master_eng_ml.utils.twitch_api import build_query, fetch_data_with_pagination


URL_TWITCH_BASE = "https://api.igdb.com/v4"


from data_master_eng_ml.utils.mappings import (
    region_mapping_inverted,
    plataform_mapping,
    player_perspectives_mapping,
    genres_mapping,
    game_modes_mapping,
    age_rating_mapping,
    age_order,
)


# game_id = 311813
def game_featurization(game_id):

    # Exemplo de uso
    url = f"{URL_TWITCH_BASE}/release_dates"

    # Define os campos a serem selecionados
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

    # Define os filtros a serem aplicados
    filters = {
        "game": f"= {game_id}",
    }

    # Chamada da função com paginação
    data_frame_games_find = fetch_data_with_pagination(url, build_query, fields, filters)
    data_frame_games_find["region_name"] = data_frame_games_find["region"].map(
        region_mapping_inverted
    )

    # Exemplo de uso
    url = f"{URL_TWITCH_BASE}/involved_companies"

    # Define os campos a serem selecionados
    fields = ["*"]
    # Define os filtros a serem aplicados
    filters = {
        "game": f"= ({game_id})",
    }

    # Chamada da função com paginação
    data_frame_companies_find = fetch_data_with_pagination(url, build_query, fields, filters)
    data_frame_companies_find
    company_id = list(set(data_frame_companies_find["company"].to_list()))
    company_id_list = ",".join(map(str, company_id))

    import pycountry_convert as pc
    import pycountry

    def country_to_continent(country_code):
        try:
            # Converte o código do país para o nome do continente
            country = pycountry.countries.get(numeric=str(country_code))
            continent_code = pc.country_alpha2_to_continent_code(country.alpha_2)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
            return continent_name
        except:
            return "Unknown"  # Retorna 'Unknown' se o código do país não for reconhecido

    def array_count(value):
        if value == "Unknown":
            return -1
        else:
            return len(value)

    # Exemplo de uso
    url = f"{URL_TWITCH_BASE}/companies"

    # Define os campos a serem selecionados
    fields = [
        "developed",
        "slug",
        "published",
        "country",
        "start_date",
        "start_date_category",
        "parent",
    ]
    # Define os filtros a serem aplicados
    filters = {
        "id": f"= ({company_id_list})",
    }
    # Chamada da função com paginação
    data_frame_companies = fetch_data_with_pagination(url, build_query, fields)
    data_frame_companies["developed"] = data_frame_companies["developed"].fillna("Unknown")
    data_frame_companies["published"] = data_frame_companies["published"].fillna("Unknown")
    data_frame_companies["country"] = data_frame_companies["country"].fillna(1)
    data_frame_companies["country"] = data_frame_companies["country"].astype(int)
    data_frame_companies["games_developed"] = data_frame_companies["developed"].apply(array_count)
    data_frame_companies["has_parents"] = data_frame_companies["parent"].notna().astype(int)
    data_frame_companies["games_published"] = data_frame_companies["published"].apply(array_count)
    data_frame_companies["continent_name"] = data_frame_companies["country"].apply(
        country_to_continent
    )

    data_frame_companies = data_frame_companies.drop(
        columns=["developed", "published", "country", "parent", "start_date_category"]
    )

    # Inner Join (default)
    companies_join = pd.merge(
        data_frame_companies_find,
        data_frame_companies,
        left_on="company",
        right_on="id",
        how="inner",
    )

    df_companies_final = companies_join[
        ["game", "games_developed", "has_parents", "games_published", "continent_name"]
    ]

    # Exemplo de uso
    url = f"{URL_TWITCH_BASE}/multiplayer_modes"

    # Define os campos a serem selecionados
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
    # Define os filtros a serem aplicados
    filters = {
        "game": f"= ({game_id})",
    }

    # Chamada da função com paginação
    data_frame_multiplayer_modes = fetch_data_with_pagination(url, build_query, fields, filters)
    for coluna in fields:
        if coluna not in data_frame_multiplayer_modes.columns:
            data_frame_multiplayer_modes[coluna] = None
    # Converte todas as colunas booleanas em 0/1
    boolean_columns = data_frame_multiplayer_modes.select_dtypes(include="bool").columns
    data_frame_multiplayer_modes[boolean_columns] = data_frame_multiplayer_modes[
        boolean_columns
    ].astype(int)

    data_frame_multiplayer_modes

    # Exemplo de uso
    url = f"{URL_TWITCH_BASE}/games"

    # Define os campos a serem selecionados
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

    # Define os filtros a serem aplicados
    filters = {
        "id": f"= {game_id}",
    }

    # Chamada da função com paginação
    data_frame_games = fetch_data_with_pagination(url, build_query, fields, filters)

    for coluna in fields:
        if coluna not in data_frame_games.columns:
            data_frame_games[coluna] = None

    # Trata as colunas
    data_frame_games["player_perspectives"] = data_frame_games["player_perspectives"].fillna(
        "Unknown"
    )
    data_frame_games["game_modes"] = data_frame_games["game_modes"].fillna("unknown_game_mode")
    data_frame_games["genres"] = data_frame_games["genres"].fillna("unknown_genres_name")
    data_frame_games["player_perspective_name"] = (
        data_frame_games["player_perspectives"]
        .map(
            lambda x: [
                player_perspectives_mapping.get(i, "unknown_player_perspectives") for i in x
            ]
        )
        .apply(lambda x: list(set(x)))
    )
    data_frame_games["platforms_name"] = (
        data_frame_games["platforms"]
        .map(lambda x: [plataform_mapping.get(i, "unknown_platforms_name") for i in x])
        .apply(lambda x: list(set(x)))
    )
    data_frame_games["genres_first"] = (
        data_frame_games["genres"]
        .map(lambda x: [genres_mapping.get(i, "unknown_genres_name") for i in x])
        .apply(lambda x: list(set(x)))
        .apply(lambda x: x[0])
    )
    data_frame_games["game_modes_name"] = (
        data_frame_games["game_modes"]
        .map(lambda x: [game_modes_mapping.get(i, "unknown_game_mode") for i in x])
        .apply(lambda x: list(set(x)))
    )
    data_frame_games["has_remaster"] = data_frame_games["remasters"].notna()

    data_frame_games["target"] = np.where(data_frame_games["rating"].isna(), 0, 1)

    data_frame_games = data_frame_games.drop(
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

    data_frame_games

    def age_classif(list):
        url = f"{URL_TWITCH_BASE}/age_ratings"

        try:
            lista_query = ",".join(map(str, list))
            # Define os campos a serem selecionados
            fields = ["rating"]

            filters = {
                "id": f"= ({lista_query})",
            }
            # Chamada da função com paginação
            data_frame = fetch_data_with_pagination(url, build_query, fields, filters)
            # # Exemplo de uso no DataFrame
            data_frame["age_rating_group"] = data_frame["rating"].map(age_rating_mapping)
            # Convertendo a coluna 'age_rating_group' para essa ordem
            data_frame["age_rating_group"] = data_frame["age_rating_group"].astype(age_order)
            # Pegar o maior valor de 'age_rating_group'
            max_age_rating_group = data_frame["age_rating_group"].max()
            return max_age_rating_group
        except:
            return "No Rating"

    data_frame_games["age_classif"] = (
        data_frame_games["age_ratings"]
        .apply(age_classif)
        .drop(
            columns=[
                "age_ratings",
            ]
        )
    )

    data_frame_games = data_frame_games.drop(columns=["age_ratings"])

    data_frame_games_find[
        [
            "game",
            "region_name",
        ]
    ]

    data_frame_games_find.drop_duplicates(subset="game")

    df_companies_games = pd.merge(
        df_companies_final,
        data_frame_games_find[
            [
                "game",
                "region_name",
            ]
        ].drop_duplicates(subset="game"),
        on="game",
        how="inner",
    )

    df_companies_games_multiplayer = pd.merge(
        df_companies_games.drop_duplicates(subset="game"),
        data_frame_multiplayer_modes[
            ["game", "onlinecoop", "onlinecoopmax", "onlinemax", "splitscreen"]
        ],
        on="game",
        how="left",
    )

    data_frame_multiplayer_modes[
        ["game", "onlinecoop", "onlinecoopmax", "onlinemax", "splitscreen"]
    ]

    df_join = (
        pd.merge(
            data_frame_games,
            df_companies_games_multiplayer,
            left_on="id",
            right_on="game",
            how="left",
        )
        .drop(columns=["game"])
        .drop_duplicates(subset="id")
    )

    # Explode as listas de 'attributes' para transformar cada elemento em uma linha separada
    df_exploded = df_join[["id", "platforms_name"]].explode("platforms_name")

    # Converte a coluna 'attributes' em colunas de dummies, com prefixo para evitar conflitos
    dummies = pd.get_dummies(df_exploded["platforms_name"].str.replace("-", "_"))

    # Agrupa por 'id' e faz a soma para agrupar as flags 0/1
    dummies = dummies.groupby(df_exploded["id"]).sum().reset_index()

    # Junte os dummies ao DataFrame original
    df_join = df_join.drop(columns=["platforms_name"]).merge(dummies, on="id")

    # Explode as listas de 'attributes' para transformar cada elemento em uma linha separada
    df_exploded = df_join[["id", "game_modes_name"]].explode("game_modes_name")

    # Converte a coluna 'attributes' em colunas de dummies, com prefixo para evitar conflitos
    dummies = pd.get_dummies(df_exploded["game_modes_name"].str.replace("-", "_"))

    # Agrupa por 'id' e faz a soma para agrupar as flags 0/1
    dummies = dummies.groupby(df_exploded["id"]).sum().reset_index()

    # Junte os dummies ao DataFrame original
    df_join = df_join.drop(columns=["game_modes_name"]).merge(dummies, on="id")

    # Explode as listas de 'attributes' para transformar cada elemento em uma linha separada
    df_exploded = df_join[["id", "player_perspective_name"]].explode("player_perspective_name")

    # Converte a coluna 'attributes' em colunas de dummies, com prefixo para evitar conflitos
    dummies = pd.get_dummies(df_exploded["player_perspective_name"].str.replace("-", "_"))

    # Agrupa por 'id' e faz a soma para agrupar as flags 0/1
    dummies = dummies.groupby(df_exploded["id"]).sum().reset_index()

    # Junte os dummies ao DataFrame original
    df_join = df_join.drop(columns=["player_perspective_name"]).merge(dummies, on="id")

    # Converte todas as colunas booleanas em 0/1
    boolean_columns = df_join.select_dtypes(include="bool").columns
    df_join[boolean_columns] = df_join[boolean_columns].astype(int)

    import numpy as np

    df_join["has_global_launch"] = np.where(df_join["region_name"] == "worldwide", 1, 0)

    df_join = df_join.drop(columns=["region_name"])

    df_join
