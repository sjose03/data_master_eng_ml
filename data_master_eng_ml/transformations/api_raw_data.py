import pandas as pd

from data_master_eng_ml.utils.twitch_api import build_query, fetch_data_with_pagination
from data_master_eng_ml.config import URL_TWITCH_BASE

from data_master_eng_ml.utils.helpers import (
    ensure_columns,
)
from typing import Union, List, Dict

from loguru import logger


def fetch_raw_game_release_dates_single(game_id: int) -> pd.DataFrame:
    """
    Fetches release dates for a single game from the IGDB API.

    This function queries the IGDB API to obtain release dates for a specific game using its ID.

    Args:
        game_id (int): The unique identifier of the game for which to fetch release dates.

    Returns:
        pd.DataFrame: A DataFrame containing the release dates of the specified game,
        with additional information such as category, platform, status, region, and more.
    """
    logger.info(f"Fetching release dates for game ID: {game_id}")

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
    filters: Dict[str, str] = {"game": f"= {game_id}"}

    logger.debug("Constructing the query and initiating data fetch...")
    data_frame_games_find = fetch_data_with_pagination(url, build_query, fields, filters)

    logger.success("Data fetched successfully.")

    return data_frame_games_find


def fetch_raw_game_release_dates_batch(year: int = 2021) -> pd.DataFrame:
    """
    Fetches game release dates from the IGDB API for a given year and region.

    This function queries the IGDB API to obtain game release dates based on specific parameters.
    It returns all release dates for games for a specified year and region.

    Args:
        year (int): The year for which to fetch game release dates. Defaults to 2021.

    Returns:
        pd.DataFrame: A DataFrame containing the release dates of games, with additional
        information such as category, platform, status, region, and more.
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
    filters: Dict[str, str] = {"status": "= 6", "y": f"= {year}", "region": "= 8"}

    logger.debug("Constructing the query and initiating data fetch...")
    data_frame_games_find = fetch_data_with_pagination(url, build_query, fields, filters)

    logger.info("Data fetched successfully.")

    logger.debug("Returning the resulting DataFrame.")
    return data_frame_games_find


def fetch_raw_involved_companies(game_id: Union[int, List[int]]) -> pd.DataFrame:
    """
    Fetches the companies involved in the development and publishing of a game from the IGDB API.

    This function queries the IGDB API to retrieve information about the companies involved
    in the development or publishing of a specific game, identified by its ID.

    Args:
        game_id (int): The ID of the game for which the involved company information should be fetched.

    Returns:
        pd.DataFrame: A DataFrame containing data about the involved companies, including details
        such as names, roles (developer, publisher, etc.), and other relevant information.
    """
    logger.info(f"Fetching involved companies for game ID: {len(game_id)}")

    url = f"{URL_TWITCH_BASE}/involved_companies"
    fields = ["*"]
    filters: Dict[str, str] = {"game": f"= ({game_id})"}

    logger.debug("Constructing the query and initiating data fetch...")
    data_frame_companies_find = fetch_data_with_pagination(url, build_query, fields, filters)

    logger.info("Data fetched successfully.")

    logger.debug("Returning the resulting DataFrame.")
    return data_frame_companies_find


def fetch_raw_companies_info(company_id_list: Union[int, List[int]]) -> pd.DataFrame:
    """
    Fetches detailed information about companies involved in the development and publishing of games from the IGDB API.

    This function queries the IGDB API to retrieve specific information about a list of companies,
    identified by their IDs. The retrieved data is then processed to fill missing values, add
    derived columns, and format the DataFrame for analysis.

    Args:
        company_id_list (List[int]): A list of company IDs whose information should be fetched.

    Returns:
        pd.DataFrame: A DataFrame containing detailed information about the companies, such as
        games developed and published, country of origin, operation start dates, and parent company information.
    """
    logger.info(f"Fetching information for companies with IDs: {len(company_id_list)}")

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
    filters: Dict[str, str] = {"id": f"= ({','.join(map(str, company_id_list))})"}

    logger.debug("Constructing the query and initiating data fetch...")
    data_frame_companies = fetch_data_with_pagination(url, build_query, fields, filters)

    logger.success("Data fetched successfully.")

    # Ensure that all expected columns are present in the DataFrame
    logger.debug("Ensuring all expected columns are present in the DataFrame.")
    data_frame_companies_raw = ensure_columns(data_frame_companies, fields)

    logger.debug("Returning the processed DataFrame.")
    return data_frame_companies_raw


def fetch_raw_multiplayer_modes(game_id: Union[int, List[int]]) -> pd.DataFrame:
    """
    Fetches available multiplayer modes for a specific game from the IGDB API.

    This function queries the IGDB API to retrieve detailed information about the multiplayer modes
    available for a game identified by its ID. The data includes information about cooperative modes,
    the maximum number of players, and whether the game supports split-screen.

    Args:
        game_id (int): The ID of the game for which multiplayer mode information should be fetched.

    Returns:
        pd.DataFrame: A DataFrame containing information about the multiplayer modes, with columns
        such as 'campaigncoop', 'lancoop', 'offlinecoop', 'offlinecoopmax', 'offlinemax', 'onlinecoop',
        'onlinecoopmax', 'onlinemax', and 'splitscreen'.
    """
    logger.info(f"Fetching multiplayer modes for game ID: {len(game_id)}")

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
    filters: Dict[str, str] = {"game": f"= ({game_id})"}

    logger.debug("Constructing the query and initiating data fetch...")
    data_frame_multiplayer_modes = fetch_data_with_pagination(url, build_query, fields, filters)

    logger.info("Data fetched successfully.")

    # Ensure that all expected columns are present in the DataFrame
    logger.debug("Ensuring all expected columns are present in the DataFrame.")
    data_frame_multiplayer_modes = ensure_columns(data_frame_multiplayer_modes, fields)

    logger.debug("Returning the processed DataFrame.")
    return data_frame_multiplayer_modes


def fetch_raw_game_info(game_id: Union[int, List[int]]) -> pd.DataFrame:
    """
    Fetches detailed information about a specific game from the IGDB API.

    This function queries the IGDB API to retrieve detailed information about a game
    identified by its ID. The information includes game modes, genres, age ratings,
    involved companies, player perspectives, platforms, ratings, and remasters.

    Args:
        game_id (int): The ID of the game for which the information should be fetched.

    Returns:
        pd.DataFrame: A DataFrame containing detailed information about the game, with columns
        such as 'name', 'game_modes', 'genres', 'age_ratings', 'involved_companies',
        'player_perspectives', 'platforms', 'rating', and 'remasters'.
    """
    logger.info(f"Fetching detailed information for game ID: {len(game_id)}")

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
    filters: Dict[str, str] = {"id": f"= ({game_id})"}

    logger.debug("Constructing the query and initiating data fetch...")
    data_frame_games = fetch_data_with_pagination(url, build_query, fields, filters)

    logger.info("Data fetched successfully.")

    # Ensure that all expected columns are present in the DataFrame
    logger.debug("Ensuring all expected columns are present in the DataFrame.")
    data_frame_games = ensure_columns(data_frame_games, fields)

    logger.debug("Returning the processed DataFrame.")
    return data_frame_games
