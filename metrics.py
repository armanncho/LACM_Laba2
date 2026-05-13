import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def calculate_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def calculate_advanced_metrics(y_true, y_pred, y_prob):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return p, r, f1, roc_auc


def plot_loss(history, title='Динамика функции потерь'):
    plt.figure(figsize=(9, 6))
    plt.plot(history['train_loss'], label='Обучающая', color='darkblue', linewidth=2)
    plt.plot(history['val_loss'], label='Валидационная', color='darkorange', linewidth=2)
    plt.title(title)
    plt.xlabel('Количество эпох')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.show()


def plot_decision_boundary(model, X, y, highlight_errors=False):
    plt.figure(figsize=(9, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='white', s=60, alpha=0.85)

    if highlight_errors:
        preds = model.predict(X)
        errors = X[y != preds]
        plt.scatter(errors[:, 0], errors[:, 1], facecolors='none', edgecolors='lime', s=150, linewidths=2,
                    label='Ошибки')

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    bound_x = np.array([x_min, x_max])

    if model.w[1] != 0:
        bound_y = -(model.w[0] * bound_x + model.b) / model.w[1]
        plt.plot(bound_x, bound_y, color='black', linestyle='--', linewidth=2.5, label='Граница классов')
        plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC curve (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.show()