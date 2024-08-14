import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import mlflow
import pandas as pd
import os


def plot_confusion_matrix(y_true, y_pred, title, filename, show_plot=True):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    if show_plot:
        plt.show()
    plt.savefig(filename)
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, title, filename, show_plot=True):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc_score(y_true, y_pred_proba),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if show_plot:
        plt.show()
    plt.savefig(filename)
    plt.close()


def generate_and_log_plots(
    y_train,
    y_train_pred,
    y_train_pred_proba,
    y_test,
    y_test_pred,
    y_test_pred_proba,
    experiment_name,
):
    """
    Gera, loga e imprime os gráficos de matriz de confusão e curva ROC para treino e teste no MLflow.

    Parâmetros:
    - y_train: Rótulos reais de treino.
    - y_train_pred: Predições de treino.
    - y_train_pred_proba: Probabilidades preditas de treino.
    - y_test: Rótulos reais de teste.
    - y_test_pred: Predições de teste.
    - y_test_pred_proba: Probabilidades preditas de teste.
    - experiment_name: Nome do experimento no MLflow.
    """
    # Plot de treino
    print("\nMétricas e Gráficos de Treino:")
    train_conf_matrix_path = f"{experiment_name}_train_confusion_matrix.png"
    plot_confusion_matrix(
        y_train, y_train_pred, "Matriz de Confusão - Treino", train_conf_matrix_path
    )
    mlflow.log_artifact(train_conf_matrix_path)
    os.remove(train_conf_matrix_path)

    train_roc_curve_path = f"{experiment_name}_train_roc_curve.png"
    plot_roc_curve(
        y_train, y_train_pred_proba, "Curva ROC - Treino", train_roc_curve_path
    )
    mlflow.log_artifact(train_roc_curve_path)
    os.remove(train_roc_curve_path)

    # Plot de teste
    print("\nMétricas e Gráficos de Teste:")
    test_conf_matrix_path = f"{experiment_name}_test_confusion_matrix.png"
    plot_confusion_matrix(
        y_test, y_test_pred, "Matriz de Confusão - Teste", test_conf_matrix_path
    )
    mlflow.log_artifact(test_conf_matrix_path)
    os.remove(test_conf_matrix_path)

    test_roc_curve_path = f"{experiment_name}_test_roc_curve.png"
    plot_roc_curve(y_test, y_test_pred_proba, "Curva ROC - Teste", test_roc_curve_path)
    mlflow.log_artifact(test_roc_curve_path)
    os.remove(test_roc_curve_path)
