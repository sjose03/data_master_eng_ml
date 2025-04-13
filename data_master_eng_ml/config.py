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
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# URLs e constantes do projeto
URL_TWITCH_BASE = "https://api.igdb.com/v4"
URL_TOKEN = "https://id.twitch.tv/oauth2/token"
YEAR = 2022  # Ano padrão usado nas operações

# Diretórios do projeto
DUCKDB_PATH = PROJ_ROOT / "db" / "duckdb" / "datamaster.db"

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
