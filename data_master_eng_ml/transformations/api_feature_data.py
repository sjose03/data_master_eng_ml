import numpy as np
import pandas as pd
from loguru import logger

from data_master_eng_ml.transformations.api_transformations import batch_fetch_age_classifications

from data_master_eng_ml.utils.mappings import (
    region_mapping_inverted,
)
from data_master_eng_ml.utils.helpers import (
    generate_dummies,
    process_companies_data,
    map_age_classifications,
    process_game_data,
)
import pandas as pd


def fetch_game_release_dates_feature(data_frame_games_find: pd.DataFrame) -> pd.DataFrame:
    """
    Processes game release dates data by adding region names based on region codes.

    This function takes a DataFrame containing game release dates fetched from the IGDB API
    and maps the region codes to their corresponding region names.

    Args:
        data_frame_games_find (pd.DataFrame): A DataFrame containing raw game release dates data
        including a 'region' column.

    Returns:
        pd.DataFrame: A DataFrame with an additional column 'region_name' that maps region codes
        to their respective region names.
    """
    logger.debug("Mapping region codes to region names...")

    # Map region codes to region names
    data_frame_games_find["region_name"] = data_frame_games_find["region"].map(
        region_mapping_inverted
    )

    logger.info("Region names mapped successfully.")
    logger.debug("Returning the updated DataFrame.")

    return data_frame_games_find


def fetch_companies_info_features(data_frame_companies: pd.DataFrame) -> pd.DataFrame:
    """
    Processes detailed information about companies involved in game development and publishing.

    This function takes a DataFrame containing detailed information about companies fetched from the IGDB API
    and processes the data to fill missing values, add derived columns, and format the DataFrame for analysis.

    Args:
        data_frame_companies (pd.DataFrame): A DataFrame containing raw company information, including details
        like games developed and published, country of origin, operation start dates, and parent company information.

    Returns:
        pd.DataFrame: A DataFrame with processed information, ready for further analysis.
    """
    logger.debug("Starting to process company information data...")

    # Process company data
    data_frame_companies = process_companies_data(data_frame_companies)

    logger.info("Company information data processed successfully.")
    logger.debug("Returning the processed DataFrame.")

    return data_frame_companies


def fetch_game_info_features(data_frame_games: pd.DataFrame) -> pd.DataFrame:
    """
    Processes detailed information about a specific game.

    This function takes a DataFrame containing detailed information about a game retrieved from the IGDB API
    and processes the data to format it for further analysis. The information includes game modes, genres,
    age ratings, involved companies, player perspectives, platforms, ratings, and remasters.

    Args:
        data_frame_games (pd.DataFrame): A DataFrame containing raw game information.

    Returns:
        pd.DataFrame: A DataFrame containing processed game information, with columns such as 'name',
        'game_modes', 'genres', 'age_ratings', 'involved_companies', 'player_perspectives',
        'platforms', 'rating', and 'remasters'.
    """
    logger.debug("Starting to process game information...")

    # Process game data for additional formatting
    data_frame_games = process_game_data(data_frame_games)

    unique_age_ratings = data_frame_games["age_ratings"].explode().dropna().unique().tolist()

    age_ratings_df = batch_fetch_age_classifications(unique_age_ratings)

    data_frame_games = map_age_classifications(data_frame_games, age_ratings_df)

    logger.info("Game information processed successfully.")
    logger.debug("Returning the processed DataFrame.")

    return data_frame_games


def complete_companies_features(
    data_frame_companies_find: pd.DataFrame, data_frame_companies_feature: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins two DataFrames containing company information.

    This function merges two DataFrames: one containing basic information about companies involved
    in game development and publishing (`data_frame_companies_find`) and another with additional
    features for these companies (`data_frame_companies_feature`). The merge is performed using
    an inner join on the company ID.

    Args:
        data_frame_companies_find (pd.DataFrame): A DataFrame containing basic information about companies.
        data_frame_companies_feature (pd.DataFrame): A DataFrame containing additional features for companies.

    Returns:
        pd.DataFrame: A DataFrame resulting from the inner join of the input DataFrames.
    """
    logger.debug("Starting to join company features...")

    # Merge DataFrames on company ID
    companies_join = pd.merge(
        data_frame_companies_find,
        data_frame_companies_feature,
        left_on="company",
        right_on="id",
        how="inner",
    )
    df_companies_final = companies_join[
        ["game", "games_developed", "has_parents", "games_published", "continent_name"]
    ]
    logger.info("Company features joined successfully.")
    logger.debug("Returning the merged DataFrame.")

    return df_companies_final


def complete_game_infos(
    df_companies_final: pd.DataFrame,
    data_frame_games_find: pd.DataFrame,
    data_frame_multiplayer_modes: pd.DataFrame,
    data_frame_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Completes and merges game information by combining data from multiple sources.

    This function merges game-related data from different DataFrames, including companies involved,
    game release information, multiplayer modes, and detailed game information. It generates dummy
    variables for categorical data and adds a flag for worldwide releases.

    Args:
        df_companies_final (pd.DataFrame): A DataFrame containing detailed information about companies involved in game development.
        data_frame_games_find (pd.DataFrame): A DataFrame containing game release information.
        data_frame_multiplayer_modes (pd.DataFrame): A DataFrame containing information about multiplayer modes for games.
        data_frame_games (pd.DataFrame): A DataFrame containing detailed information about games.

    Returns:
        pd.DataFrame: A final DataFrame combining all the merged and processed information.
    """
    logger.info("Starting to merge game information...")

    # Merge with company data and multiplayer modes
    df_companies_games = pd.merge(
        df_companies_final,
        data_frame_games_find[["game", "region_name"]].drop_duplicates(subset="game"),
        on="game",
        how="inner",
    )

    logger.debug("Merged company and game release data.")

    df_companies_games_multiplayer = pd.merge(
        df_companies_games.drop_duplicates(subset="game"),
        data_frame_multiplayer_modes[
            ["game", "onlinecoop", "onlinecoopmax", "onlinemax", "splitscreen"]
        ],
        on="game",
        how="left",
    )

    logger.debug("Merged with multiplayer mode data.")

    # Combine everything into a final DataFrame
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

    logger.debug("Combined all data into the final DataFrame.")

    # Generate dummy variables
    df_join = generate_dummies(
        df_join, ["platforms_name", "game_modes_name", "player_perspective_name"]
    )

    logger.debug("Generated dummy variables for categorical data.")

    # Add a flag for worldwide release
    df_join["has_global_launch"] = np.where(df_join["region_name"] == "worldwide", 1, 0)
    df_join = df_join.drop(columns=["region_name"])

    logger.info("Final game information DataFrame created successfully.")
    logger.debug("Returning the final processed DataFrame.")

    return df_join
