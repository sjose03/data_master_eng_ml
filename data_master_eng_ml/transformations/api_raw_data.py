"""
Module: api_raw_data.py
Description: This module is responsible for fetching raw data from the Twitch API using IGDB endpoints and saving the data into a DuckDB database.
It leverages pagination during data fetching and logs processing steps using Loguru.
"""

from typing import Dict, Type

from data_master_eng_ml.schemas.api_schemas import (
    GamesSchema,
    PlatformSchema,
    GenreSchema,
    PlayerPerspectiveSchema,
    AgeContentDescriptionSchema,
    AgeRatingSchema,
    CompaniesSchema,
    GamesModesSchema,
    LanguageSupportSchema,
    ThemeSchema,
)

from data_master_eng_ml.utils.helpers import (
    save_to_duckdb,
    get_unix_timestamp_for_year_start,
)
from data_master_eng_ml.utils.api_igdb import (
    build_query,
    fetch_data_with_pagination,
)
from data_master_eng_ml.config import URL_TWITCH_BASE, DUCKDB_PATH

from loguru import logger

# Mapping of raw API table names to their corresponding schema classes
raw_api_tables_schemas: Dict[str, Type] = {
    "raw_games": GamesSchema,
    "raw_platforms": PlatformSchema,
    "raw_player_perspectives": PlayerPerspectiveSchema,
    "raw_genres": GenreSchema,
    "raw_themes": ThemeSchema,
    "raw_companies": CompaniesSchema,
    "raw_age_ratings": AgeRatingSchema,
    "raw_languages": LanguageSupportSchema,
    "raw_age_rating_content_descriptions": AgeContentDescriptionSchema,
    "raw_game_modes": GamesModesSchema,
}


def fetch_raw_data(year: int = 2022) -> None:
    """
    Fetch raw data from the Twitch API and save it to DuckDB.

    This function iterates over a mapping of table names to schema definitions. For each schema, it constructs
    an API query, fetches the data using pagination support, and saves the resulting DataFrame into a
    DuckDB table. A filter is applied for the GamesSchema to retrieve data with a 'first_release_date'
    greater than or equal to the start of the given year.

    Args:
        year (int): The year for which to fetch data. Defaults to 2022.
    """
    # Convert the provided year into a Unix timestamp representing the start of the year
    year_timestamp = get_unix_timestamp_for_year_start(year)
    logger.info(f"Starting data fetch process for year: {year}")

    # Iterate over each table and its corresponding schema
    for table, schema in raw_api_tables_schemas.items():
        # Extract the endpoint name from the schema's JSON description
        schema_json = schema.model_json_schema()
        description = schema_json.get("description", "")
        endpoint_name = description.split(" ")[-2] if description else "unknown"

        # Retrieve the list of fields from the schema properties
        fields = list(schema_json.get("properties", {}).keys())

        # Define filters for the query: apply filter for GamesSchema based on release date
        filters = (
            {"first_release_date": f">= {year_timestamp}"}
            if schema == GamesSchema
            else {}
        )

        # Construct the API URL for the current endpoint
        url = f"{URL_TWITCH_BASE}/{endpoint_name}"
        logger.debug(f"Fetching data for endpoint: {endpoint_name}")
        logger.debug(f"URL: {url}")
        logger.debug(f"Fields: {fields}")
        logger.debug(f"Filters: {filters}")
        logger.debug(f"Schema: {schema.__name__}")

        # Fetch data from the API with pagination support
        dataframe = fetch_data_with_pagination(
            url, build_query, fields, filters, schema=schema
        )

        # Save the fetched DataFrame into the DuckDB database under the specified table name
        save_to_duckdb(dataframe, table, DUCKDB_PATH)
        logger.debug(f"Data from endpoint '{endpoint_name}' saved to table '{table}'.")
        logger.info(
            f"Completed processing for table: {table} with schema: {schema.__name__}"
        )
