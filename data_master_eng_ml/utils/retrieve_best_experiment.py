import mlflow


def get_best_experiment(min_auc_difference=0.05, experiment_name="Model_Training"):
    """
    Recupera o ID do experimento com o maior AUC no conjunto de teste e com uma diferença
    baixa entre AUC de treino e teste.

    Parâmetros:
    - min_auc_difference: Diferença máxima permitida entre AUC de treino e teste.
    - experiment_name: Nome do experimento no MLflow.

    Retorno:
    - best_run_id: ID do melhor experimento.
    - best_run_metrics: Métricas do melhor experimento.
    """

    # Buscar experimentos pelo nome
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado.")

    experiment_id = experiment.experiment_id

    # Buscar todos os runs do experimento
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    best_run_id = None
    best_run_metrics = None
    best_test_auc = float("-inf")

    for _, run in runs.iterrows():
        train_auc = run["metrics.roc_auc_train"]
        test_auc = run["metrics.roc_auc_test"]
        auc_difference = abs(train_auc - test_auc)

        if test_auc > best_test_auc and auc_difference <= min_auc_difference:
            best_test_auc = test_auc
            best_run_id = run["run_id"]
            best_run_metrics = {
                "train_auc": train_auc,
                "test_auc": test_auc,
                "auc_difference": auc_difference,
            }

    if best_run_id is None:
        raise ValueError("Nenhum experimento encontrado que atenda aos critérios.")

    print(f"Melhor Experimento Encontrado: Run ID = {best_run_id}")
    print(f"Métricas: {best_run_metrics}")

    return best_run_id, best_run_metrics


def register_model(best_run_id, model_name="Best_Model"):
    """
    Registra a versão do modelo a partir do melhor experimento.

    Parâmetros:
    - best_run_id: ID do experimento a ser registrado.
    - model_name: Nome do modelo a ser registrado.
    """

    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, model_name)
    print(f"Modelo registrado com sucesso! Nome do Modelo: {model_name}")


# Exemplo de uso:
if __name__ == "__main__":
    best_run_id, best_run_metrics = get_best_experiment(
        min_auc_difference=0.05, experiment_name="Model_Training"
    )
    register_model(best_run_id, model_name="Best_Model")
