import os
import sys
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Verificar se o log no CometML deve ser desativado
disable_comet_logging = os.getenv("DISABLE_COMET_LOGGING", "False").lower() == "true"


if not disable_comet_logging:
    import comet_ml

    COMET_API_KEY = os.getenv("COMET_API_KEY")
    COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
    COMET_PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")
    COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME")
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )

import joblib
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Adicione o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.featurization import get_data_from_mongodb, create_features


# Função principal de treinamento do modelo
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if not disable_comet_logging:
        experiment.log_metric("mean_squared_error", mse)
        experiment.log_metric("r2_score", r2)

        # Registrar o modelo no CometML
        model_path = "model.pkl"
        joblib.dump(model, model_path)
        experiment.log_model(COMET_MODEL_NAME, model_path)
        # Logar a assinatura do modelo
        model_signature = {
            "input_features": list(features.columns),
            "output_type": "float",
        }
        experiment.log_other("model_signature", model_signature)
        experiment.register_model(
            COMET_MODEL_NAME, description="Initial XGBoost model", public=True
        )

        # Salvar a versão do modelo em um arquivo de ambiente
        with open(".env", "a") as f:
            f.write(f"\nMODEL_VERSION=latest\n")  # Ajustar conforme necessário

    return model


if __name__ == "__main__":
    data, target = get_data_from_mongodb(
        os.getenv("MONGODB_URI"),
        os.getenv("DATABASE_NAME"),
        os.getenv("COLLECTION_NAME"),
    )
    features = create_features(data)
    train_model(features, target)

    if not disable_comet_logging:
        experiment.end()
