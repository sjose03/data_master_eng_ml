from unittest import mock
import pytest
from unittest.mock import patch, MagicMock
from data_master_eng_ml.transformations.api_raw_data import (
    fetch_raw_data,
    raw_api_tables_schemas,
)
from data_master_eng_ml.utils.helpers import get_unix_timestamp_for_year_start
from data_master_eng_ml.utils.api_igdb import fetch_data_with_pagination
from data_master_eng_ml.utils.helpers import save_to_duckdb
from data_master_eng_ml.config import DUCKDB_PATH


@pytest.fixture
def mock_logger():
    with patch("data_master_eng_ml.transformations.api_raw_data.logger") as mock_logger:
        yield mock_logger


@pytest.fixture
def mock_get_unix_timestamp_for_year_start():
    with patch(
        "data_master_eng_ml.utils.helpers.get_unix_timestamp_for_year_start"
    ) as mock_timestamp:
        mock_timestamp.return_value = 1640995200  # Mocked timestamp for 2022-01-01
        yield mock_timestamp


@pytest.fixture
def mock_fetch_data_with_pagination():
    with patch(
        "data_master_eng_ml.utils.api_igdb.fetch_data_with_pagination"
    ) as mock_fetch:
        mock_fetch.return_value = MagicMock()  # Mocked DataFrame
        yield mock_fetch


@pytest.fixture
def mock_save_to_duckdb():
    with patch("data_master_eng_ml.utils.helpers.save_to_duckdb") as mock_save:
        yield mock_save


def test_fetch_raw_data(
    mock_logger,
    mock_get_unix_timestamp_for_year_start,
    mock_fetch_data_with_pagination,
    mock_save_to_duckdb,
):
    # Call the function
    fetch_raw_data(year=2022)

    # Assert the logger was called
    mock_logger.info.assert_any_call("Starting data fetch process for year: 2022")

    # Assert the timestamp function was called with the correct year
    mock_get_unix_timestamp_for_year_start.assert_called_once_with(2022)

    # Assert fetch_data_with_pagination and save_to_duckdb were called for each schema
    for table, schema in raw_api_tables_schemas.items():
        schema_json = schema.model_json_schema()
        description = schema_json.get("description", "")
        endpoint_name = description.split(" ")[-2] if description else "unknown"
        fields = list(schema_json.get("properties", {}).keys())
        filters = (
            {"first_release_date": ">= 1640995200"}
            if schema.__name__ == "GamesSchema"
            else {}
        )
        url = f"https://api.twitch.tv/helix/{endpoint_name}"

        mock_fetch_data_with_pagination.assert_any_call(
            url,
            mock.ANY,  # build_query function
            fields,
            filters,
            schema=schema,
        )
        mock_save_to_duckdb.assert_any_call(
            mock_fetch_data_with_pagination.return_value, table, DUCKDB_PATH
        )

    # Assert logger debug and info calls for each table
    assert mock_logger.debug.call_count >= len(raw_api_tables_schemas)
    assert mock_logger.info.call_count >= len(raw_api_tables_schemas)
