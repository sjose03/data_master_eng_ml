from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
from data_master_eng_ml.config import (
    YEAR,
    GAME_INFO_RAW_COLLECTION,
    MULTIPLAYER_MODES_RAW_COLLECTION,
    INVOLVED_COMPANIES_LIST_RAW_COLLECTION,
    INVOLVED_COMPANIES_RAW_COLLECTION,
    GAME_RELEASE_DATES_RAW_COLLECTION,
    MONGODB_DATABASE_RAW,
    MONGODB_DATABASE_SILVER,
    GAMES_SILVER_COLLECTION,
)
from data_master_eng_ml.transformations.api_feature_data import (
    fetch_game_release_dates_feature,
    fetch_companies_info_features,
    fetch_game_info_features,
    complete_companies_features,
    complete_game_infos,
)
from data_master_eng_ml.utils.helpers import read_data_from_mongodb, save_dataframe_to_mongodb

app = typer.Typer()


def featurization(year: int = YEAR) -> None:
    """
    Performs data featurization for a specified year by fetching raw data from MongoDB, processing it,
    and combining it into a single DataFrame.

    This function reads data from multiple MongoDB collections, applies various processing and
    feature extraction steps, and then merges the processed data into a final DataFrame ready for
    analysis or machine learning.

    Args:
        year (int): The reference year for which the data is to be featurized.

    Returns:
        pd.DataFrame: A DataFrame containing the combined and processed features from different data sources.
    """
    logger.info(f"Starting data featurization for the year: {year}")

    # Define the query to filter data by the specified year
    query = {"dat_ref": year}

    # Read raw data from MongoDB
    logger.debug("Reading multiplayer modes data from MongoDB...")
    data_frame_multiplayer_modes = read_data_from_mongodb(
        MONGODB_DATABASE_RAW, MULTIPLAYER_MODES_RAW_COLLECTION, query=query
    )

    logger.debug("Reading involved companies list data from MongoDB...")
    data_frame_companies_find = read_data_from_mongodb(
        MONGODB_DATABASE_RAW, INVOLVED_COMPANIES_LIST_RAW_COLLECTION, query=query
    )

    logger.debug("Reading game release dates data from MongoDB...")
    data_frame_games_find = read_data_from_mongodb(
        MONGODB_DATABASE_RAW, GAME_RELEASE_DATES_RAW_COLLECTION, query=query
    )

    # Feature extraction and processing
    logger.debug("Extracting features from game release dates data...")
    data_frame_games_find_feature = fetch_game_release_dates_feature(data_frame_games_find)

    logger.debug("Reading involved companies data from MongoDB...")
    data_frame_companies = read_data_from_mongodb(
        MONGODB_DATABASE_RAW, INVOLVED_COMPANIES_RAW_COLLECTION, query=query
    )

    logger.debug("Extracting features from companies data...")
    data_frame_companies_feature = fetch_companies_info_features(data_frame_companies)

    logger.debug("Reading game information data from MongoDB...")
    data_frame_games = read_data_from_mongodb(
        MONGODB_DATABASE_RAW, GAME_INFO_RAW_COLLECTION, query=query
    )

    logger.debug("Extracting features from game information data...")
    data_frame_games_feature = fetch_game_info_features(data_frame_games)

    # Complete company features
    logger.debug("Completing companies features...")
    df_companies_final = complete_companies_features(
        data_frame_companies_find,
        data_frame_companies_feature,
    )

    # Merge all features into a final DataFrame
    logger.debug("Merging all game information into a final DataFrame...")
    df_join = complete_game_infos(
        df_companies_final=df_companies_final,
        data_frame_games_find=data_frame_games_find_feature,
        data_frame_multiplayer_modes=data_frame_multiplayer_modes,
        data_frame_games=data_frame_games_feature,
    )

    logger.info("Data featurization completed successfully.")
    # Save the final DataFrame to MongoDB
    logger.debug("Saving the final DataFrame to MongoDB...")
    save_dataframe_to_mongodb(
        df_join, MONGODB_DATABASE_SILVER, collection_name=GAMES_SILVER_COLLECTION
    )
    logger.info("Final DataFrame saved to MongoDB successfully.")


@app.command()
def main(year: int = YEAR):
    """
    Main entry point for the data processing pipeline.

    This function triggers the featurization process for a specified year, which involves reading raw data,
    processing it, and saving the final features into MongoDB.

    Args:
        year (int): The year for which the data is to be processed and features are to be generated.
    """
    logger.info("Generating features from dataset...")

    # Call the featurization function
    df_final = featurization(year)

    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
