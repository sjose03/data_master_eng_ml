from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import os


# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Definir o diretório raiz do projeto
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Diretório raiz do projeto: {PROJ_ROOT}")

# Variáveis de ambiente necessárias
TWITCH_ID = os.getenv("TWITCH_ID")
TWITCH_SECRET = os.getenv("TWITCH_SECRET")
MONGODB_URI = os.getenv(
    "MONGODB_URI", "mongodb://root:example@localhost:27017/"
)
MLFLOW_S3_ENDPOINT_URL = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
)
AWS_ACCESS_KEY_ID = os.getenv(
    "AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID=minioadmin"
)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# URLs e constantes do projeto
URL_TWITCH_BASE = "https://api.igdb.com/v4"
URL_TOKEN = "https://id.twitch.tv/oauth2/token"
YEAR = 2022  # Ano padrão usado nas operações
# Nome dos bancos de dados e coleções MongoDB
MONGODB_DEFAULT_DATABASE = "datamasterml"
MONGODB_DATABASE_RAW = "datamaster_raw"
MONGODB_DATABASE_SILVER = "datamaster_silver"
GAME_RELEASE_DATES_RAW_COLLECTION = "game_release_dates_raw"
INVOLVED_COMPANIES_RAW_COLLECTION = "involved_companies_raw"
INVOLVED_COMPANIES_LIST_RAW_COLLECTION = "involved_companies_list_raw"
MULTIPLAYER_MODES_RAW_COLLECTION = "multiplayer_modes_raw"
GAME_INFO_RAW_COLLECTION = "game_info_raw"
GAMES_SILVER_COLLECTION = "cleaned_games_info"

# Diretórios do projeto
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# Integração do loguru com tqdm (se tqdm estiver instalado)
try:
    from tqdm import tqdm

    # Remove o logger padrão e substitui pela integração com tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    logger.warning(
        "tqdm não está instalado. Logs serão exibidos sem integração com tqdm."
    )
