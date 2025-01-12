import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(model, feature_names):
    importance = model.get_score(importance_type="weight")
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=[x[1] for x in importance_sorted],
        y=[feature_names[int(x[0][1:])] for x in importance_sorted],
    )
    plt.title("Importância das Features")
    plt.xlabel("Importância")
    plt.ylabel("Feature")
    plt.show()
