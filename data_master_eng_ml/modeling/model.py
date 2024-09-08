from matplotlib import pyplot as plt
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.base import BaseEstimator
import pandas as pd
import os

from data_master_eng_ml.visualization.plot_utils import (
    generate_and_log_plots,
)


# Reutilizando a função evaluate_model para calcular as métricas
def evaluate_model(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Acurácia: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"F1 Score: {f1}")
    print("Relatório de Classificação:\n", classification_rep)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.show()


def log_data_and_plots(X_train, y_train, X_test, y_test, model, experiment_name, algorithm):
    """
    Loga os dados e os gráficos (Matriz de Confusão, AUC-ROC) no MLflow.

    Parâmetros:
    - X_train: Dados de treino.
    - y_train: Rótulos de treino.
    - X_test: Dados de teste.
    - y_test: Rótulos de teste.
    - model: Modelo treinado.
    - experiment_name: Nome do experimento no MLflow.
    - algorithm: Algoritmo utilizado ('xgboost', 'lightgbm', 'random_forest')
    """
    # Logando os dados
    data_path = f"{experiment_name}_data.csv"
    df_train = pd.DataFrame(X_train)
    df_train["target"] = y_train
    df_test = pd.DataFrame(X_test)
    df_test["target"] = y_test
    df_full = pd.concat([df_train, df_test])
    df_full.to_csv(data_path, index=False)
    mlflow.log_artifact(data_path)
    os.remove(data_path)  # Remover o arquivo após o log para evitar acúmulo de arquivos

    # Fazer previsões
    y_train_pred, y_train_pred_proba = predict_model(model, X_train, algorithm)
    y_test_pred, y_test_pred_proba = predict_model(model, X_test, algorithm)

    # Gerar e logar os gráficos
    generate_and_log_plots(
        y_train,
        y_train_pred,
        y_train_pred_proba,
        y_test,
        y_test_pred,
        y_test_pred_proba,
        experiment_name,
    )


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    algorithm="xgboost",
    params=None,
    use_smote=False,
    experiment_name="Model_Training",
):
    """
    Treina um modelo de machine learning usando o algoritmo especificado e registra o processo no MLflow.

    Parâmetros:
    - X_train: Dados de treino (features).
    - y_train: Dados de treino (target).
    - X_test: Dados de teste (features).
    - y_test: Dados de teste (target).
    - algorithm: Algoritmo a ser utilizado ('xgboost', 'random_forest', 'lightgbm').
    - params: Parâmetros do modelo.
    - use_smote: Se True, aplica SMOTE para lidar com desbalanceamento.
    - experiment_name: Nome do experimento no MLflow.

    Retorno:
    - model: Modelo treinado.
    """

    experiment_name = f"{algorithm}_model{'_with_smote' if use_smote else ''}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=experiment_name) as run:
        mlflow.autolog(log_input_examples=True)
        if use_smote:
            # Aplicar SMOTE para lidar com o desbalanceamento
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        if algorithm == "xgboost":
            # Converter X_train e X_test para DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            model, model_params = train_xgboost(dtrain, dtest, params)

            # Avaliar e logar as métricas no conjunto de treino
            y_train_pred_proba = model.predict(dtrain)
            y_train_pred = (y_train_pred_proba > 0.5).astype(int)
            train_auc = roc_auc_score(y_train, y_train_pred_proba)
            mlflow.log_metric("roc_auc_train", train_auc)

        elif algorithm == "random_forest":
            model, model_params = train_random_forest(X_train, y_train, params)
            y_train_pred_proba = model.predict_proba(X_train)[:, 1]
            y_train_pred = model.predict(X_train)
            train_auc = roc_auc_score(y_train, y_train_pred_proba)
            mlflow.log_metric("roc_auc_train", train_auc)

        elif algorithm == "lightgbm":
            model, model_params = train_lightgbm(X_train, y_train, X_test, y_test, params)
            y_train_pred_proba = model.predict_proba(X_train)[:, 1]
            y_train_pred = model.predict(X_train)
            train_auc = roc_auc_score(y_train, y_train_pred_proba)
            mlflow.log_metric("roc_auc_train", train_auc)
        else:
            raise ValueError(
                f"Algoritmo {algorithm} não suportado. Escolha entre 'xgboost', 'random_forest' ou 'lightgbm'."
            )
        # Logar a métrica de AUC no conjunto de teste
        if algorithm == "xgboost":
            y_test_pred_proba = model.predict(dtest)
            y_test_pred = (y_test_pred_proba > 0.5).astype(int)
        else:
            y_test_pred_proba = (
                model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            )
            y_test_pred = model.predict(X_test)

        if y_test_pred_proba is not None:
            test_auc = roc_auc_score(y_test, y_test_pred_proba)
            mlflow.log_metric("roc_auc_test", test_auc)
            mlflow.end_run()

        # Logar a acurácia no conjunto de teste
        test_accuracy = accuracy_score(y_test, y_test_pred)
        mlflow.log_metric("accuracy_test", test_accuracy)

        # Logar parâmetros e modelo no MLflow
        mlflow.log_params(model_params)
        # mlflow.sklearn.log_model(model, f"{algorithm}_model")

        # Logar os dados e os gráficos
        log_data_and_plots(X_train, y_train, X_test, y_test, model, experiment_name, algorithm)

        return model


def train_xgboost(dtrain, dval, params):
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "learning_rate": 0.015,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 1,
        "reg_alpha": 0.35,
        "reg_lambda": 0.45,
        "seed": 42,
    }
    params = params or default_params

    evals = [(dtrain, "train"), (dval, "eval")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        early_stopping_rounds=10,
        evals=evals,
        verbose_eval=False,
    )

    return model, params


def train_random_forest(X_train, y_train, params):
    default_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    params = params or default_params

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    return model, params


def train_lightgbm(X_train, y_train, params):
    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "max_depth": 3,
        "learning_rate": 0.015,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.35,
        "reg_lambda": 0.45,
        "seed": 42,
    }
    params = params or default_params

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    return model, params


def predict_model(model: BaseEstimator, X_test, algorithm):
    """
    Realiza previsões usando o modelo treinado.

    Parâmetros:
    - model: Modelo treinado.
    - X_test: Dados de teste (features).
    - algorithm: Algoritmo utilizado ('xgboost', 'lightgbm', 'random_forest').
    Retorno:
    - y_test_pred: Predições das classes.
    - y_test_pred_proba: Predições das probabilidades (se disponível).
    """
    if algorithm == "xgboost":
        dtest = xgb.DMatrix(X_test)
        y_test_pred_proba = model.predict(dtest)
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    else:
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = (
            model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        )

    return y_test_pred, y_test_pred_proba
