from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# Definir o diretório raiz do projeto
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Diretório raiz do projeto: {PROJ_ROOT}")

# Variáveis de ambiente necessárias
TWITCH_ID = os.getenv("TWITCH_ID")
TWITCH_SECRET = os.getenv("TWITCH_SECRET")
MONGODB_URI = os.getenv("MONGODB_URI")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# URLs e constantes do projeto
URL_TWITCH_BASE = "https://api.igdb.com/v4"
URL_TOKEN = "https://id.twitch.tv/oauth2/token"
YEAR = 2022  # Ano padrão usado nas operações
# Nome dos bancos de dados e coleções MongoDB
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

# Configuração do cliente MongoDB
try:
    # Inicializa o cliente MongoDB com o URI fornecido
    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
    # Testa a conexão enviando um comando "ping"
    client.admin.command("ping")
    logger.debug("Conexão bem-sucedida ao MongoDB!")
except Exception as e:
    logger.error(f"Erro ao conectar ao MongoDB: {e}")

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
