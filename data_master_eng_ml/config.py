from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import os


from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")


TWITCH_ID = os.getenv("TWITCH_ID")
TWITCH_SECRET = os.getenv("TWITCH_SECRET")
MONGODB_URI = os.getenv("MONGODB_URI")
URL_TWITCH_BASE = "https://api.igdb.com/v4"
URL_TOKEN = "https://id.twitch.tv/oauth2/token"
YEAR = 2022
MONGODB_DATABASE_RAW = "datamaster_raw"
GAME_RELEASE_DATES_RAW_COLLECTION = "game_release_dates_raw"
INVOLVED_COMPANIES_RAW_COLLECTION = "involved_companies_raw"
MULTIPLAYER_MODES_RAW_COLLECTION = "multiplayer_modes_raw"
GAME_INFO_RAW_COLLECTION = "game_info_raw"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create a new client and connect to the server
client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    logger.debug("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    logger.error(e)
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
