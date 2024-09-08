from pathlib import Path

import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd

from data_master_eng_ml.config import (
    YEAR,
    MONGODB_DATABASE_RAW,
    INVOLVED_COMPANIES_RAW_COLLECTION,
    INVOLVED_COMPANIES_LIST_RAW_COLLECTION,
    GAME_INFO_RAW_COLLECTION,
    MULTIPLAYER_MODES_RAW_COLLECTION,
    GAME_RELEASE_DATES_RAW_COLLECTION,
)
from data_master_eng_ml.transformations.api_raw_data import (
    fetch_raw_game_release_dates_batch,
    fetch_raw_game_release_dates_single,
    fetch_raw_involved_companies,
    fetch_raw_companies_info,
    fetch_raw_multiplayer_modes,
    fetch_raw_game_info,
)
from data_master_eng_ml.utils.helpers import save_dataframe_to_mongodb

app = typer.Typer()


from typing import List
import pandas as pd
from loguru import logger


def ingest_raw_data(year: int = 2021) -> None:
    """
    Ingests raw game data from the IGDB API for a specified year.

    This function retrieves various types of game data from the IGDB API, such as game release dates,
    involved companies, multiplayer modes, and detailed game information for all games released in a
    specified year. The data is then saved to a MongoDB database.

    Args:
        year (int): The year for which to fetch game data. Defaults to 2021.

    Returns:
        None
    """
    logger.info(f"Ingesting raw data for the year: {year}")

    # Fetch game release dates
    logger.debug("Fetching game release dates batch...")
    data_frame_games_find_raw_batch = fetch_raw_game_release_dates_batch(year=year)

    # Get unique game IDs
    games_id_batch: List[int] = list(set(data_frame_games_find_raw_batch["game"].to_list()))
    games_id_list_batch: str = ",".join(map(str, games_id_batch))

    # Fetch involved companies data
    logger.debug("Fetching involved companies batch...")
    data_frame_companies_find_raw_batch = fetch_raw_involved_companies(games_id_list_batch)

    # Get unique company IDs
    company_id_list_batch: List[int] = list(
        set(data_frame_companies_find_raw_batch["company"].to_list())
    )

    # Fetch detailed companies info
    logger.debug("Fetching detailed companies information...")
    data_frame_companies_raw_batch = fetch_raw_companies_info(company_id_list_batch)

    # Fetch multiplayer modes
    logger.debug("Fetching multiplayer modes batch...")
    data_frame_multiplayer_modes_raw_batch = fetch_raw_multiplayer_modes(games_id_list_batch)

    # Fetch detailed game info
    logger.debug("Fetching detailed game information batch...")
    data_frame_games_raw_batch = fetch_raw_game_info(games_id_list_batch)

    # Save data to MongoDB
    logger.info("Saving game release dates to MongoDB...")
    data_frame_games_find_raw_batch["dat_ref"] = year
    save_dataframe_to_mongodb(
        data_frame_games_find_raw_batch, MONGODB_DATABASE_RAW, GAME_RELEASE_DATES_RAW_COLLECTION
    )

    logger.info("Saving companies list data to MongoDB...")
    data_frame_companies_find_raw_batch["dat_ref"] = year
    save_dataframe_to_mongodb(
        data_frame_companies_find_raw_batch,
        MONGODB_DATABASE_RAW,
        INVOLVED_COMPANIES_LIST_RAW_COLLECTION,
    )
    logger.info("Saving companies data to MongoDB...")
    data_frame_companies_raw_batch["dat_ref"] = year

    save_dataframe_to_mongodb(
        data_frame_companies_raw_batch, MONGODB_DATABASE_RAW, INVOLVED_COMPANIES_RAW_COLLECTION
    )

    logger.info("Saving multiplayer modes to MongoDB...")
    data_frame_multiplayer_modes_raw_batch["dat_ref"] = year

    save_dataframe_to_mongodb(
        data_frame_multiplayer_modes_raw_batch,
        MONGODB_DATABASE_RAW,
        MULTIPLAYER_MODES_RAW_COLLECTION,
    )

    logger.info("Saving game information to MongoDB...")
    data_frame_games_raw_batch["dat_ref"] = year
    save_dataframe_to_mongodb(
        data_frame_games_raw_batch, MONGODB_DATABASE_RAW, GAME_INFO_RAW_COLLECTION
    )

    logger.info("Data ingestion completed successfully.")


@app.command()
def main(year: int = YEAR) -> None:
    """
    Main entry point for the data ingestion process.

    This function calls the ingestion function to fetch and store raw game data
    for a specified year from the IGDB API and logs the completion of the process.

    Args:
        year (int): The year for which to ingest game data. Defaults to the value of YEAR constant.

    Returns:
        None
    """
    logger.info(f"Starting data ingestion for the year: {year}")

    # Call the data ingestion function
    ingest_raw_data(year=year)

    # Log successful completion
    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
