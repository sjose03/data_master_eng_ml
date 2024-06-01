import comet_ml
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from data.featurization import get_data_from_mongodb, create_features
from dotenv import load_dotenv

load_dotenv()

COMET_API_KEY = os.getenv("COMET_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME")
COMET_PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")

experiment = comet_ml.Experiment(api_key=COMET_API_KEY, project_name=COMET_PROJECT_NAME)

data, target = get_data_from_mongodb(MONGODB_URI, DATABASE_NAME, COLLECTION_NAME)
features = create_features(data)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

experiment.log_metric("mean_squared_error", mse)
experiment.log_metric("r2_score", r2)

joblib.dump(model, "model.pkl")

# Registrar assinatura do modelo
input_example = features.head(1).to_dict(orient="records")
output_example = model.predict(features.head(1))
signature = {"input": input_example, "output": output_example.tolist()}

# Usar as variáveis de ambiente para nome do modelo e workspace
model_version = experiment.log_model(COMET_MODEL_NAME, "model.pkl", signature=signature)

# Salvar a versão do modelo em um arquivo de ambiente
with open(".env", "a") as f:
    f.write(f"\nMODEL_VERSION={model_version}\n")

experiment.end()
