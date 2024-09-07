from pathlib import Path

import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd

from data_master_eng_ml.config import YEAR, SINGLE
from data_master_eng_ml.utils.api_transformations import (
    batch_fetch_age_classifications,
    fetch_companies_info,
    fetch_game_info,
    fetch_game_release_dates,
    fetch_involved_companies,
    fetch_multiplayer_modes,
)
from data_master_eng_ml.utils.helpers import generate_dummies, map_age_classifications

app = typer.Typer()


def game_featurization(single: bool, game_id: int = 0, year: int = 2021):
    """
    Função principal que realiza a featurização do jogo.
    """
    # Obtém dados de diferentes fontes da API
    data_frame_games_find = fetch_game_release_dates(single=single, year=year)
    data_frame_companies_find = fetch_involved_companies(game_id)
    company_id_list = ",".join(map(str, set(data_frame_companies_find["company"].to_list())))
    data_frame_companies = fetch_companies_info(company_id_list)

    # Processa dados das empresas
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

    # Obtém dados de modos multiplayer
    data_frame_multiplayer_modes = fetch_multiplayer_modes(game_id)

    # Obtém informações gerais do jogo
    data_frame_games = fetch_game_info(game_id)

    # Faz o batch fetch das classificações etárias
    unique_age_ratings = data_frame_games["age_ratings"].explode().dropna().unique().tolist()
    age_ratings_df = batch_fetch_age_classifications(unique_age_ratings)

    # Mapeia as classificações etárias para o DataFrame de jogos
    data_frame_games = map_age_classifications(data_frame_games, age_ratings_df)

    # Merge com dados das empresas e modos multiplayer
    df_companies_games = pd.merge(
        df_companies_final,
        data_frame_games_find[["game", "region_name"]].drop_duplicates(subset="game"),
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

    # Combina tudo em um único DataFrame final
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

    # Gera variáveis dummy
    df_join = generate_dummies(
        df_join, ["platforms_name", "game_modes_name", "player_perspective_name"]
    )

    # Adiciona flag para lançamento global
    df_join["has_global_launch"] = np.where(df_join["region_name"] == "worldwide", 1, 0)
    df_join = df_join.drop(columns=["region_name"])

    return df_join


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    single: bool = SINGLE,
    year: int = YEAR,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    game_featurization(single=single, year=year)
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
