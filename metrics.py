import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def display_performance_report(actual, predicted, probabilities):
    """Выводит красиво оформленную сводку метрик качества."""
    print("\n" + "=" * 30)
    print(" ОТЧЕТ О КАЧЕСТВЕ МОДЕЛИ ")
    print("=" * 30)
    print(f"Доля верных ответов: {metrics.accuracy_score(actual, predicted):.4f}")
    print(f"Точность (Precision): {metrics.precision_score(actual, predicted):.4f}")
    print(f"Полнота (Recall):     {metrics.recall_score(actual, predicted):.4f}")
    print(f"F1-мера:              {metrics.f1_score(actual, predicted):.4f}")
    print(f"Площадь под кривой:   {metrics.roc_auc_score(actual, probabilities):.4f}")
    print("=" * 30 + "\n")


def render_learning_curves(train_history, test_history=None):
    """Строит графики изменения функции ошибки."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_history, label='Обучающая выборка', linewidth=2)
    if test_history:
        ax.plot(test_history, label='Тестовая выборка', linewidth=2, linestyle='--')

    ax.set_title('Динамика функции потерь', fontsize=14)
    ax.set_xlabel('Эпохи обучения', fontsize=12)
    ax.set_ylabel('Значение Loss', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()


def draw_boundary_map(model, X, y):
    """Отрисовка данных и найденной разделяющей гиперплоскости."""
    plt.figure(figsize=(9, 6))
    # Разброс точек
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, alpha=0.7, edgecolors='white')

    # Расчет границы: weights[0]*x + weights[1]*y + bias = 0
    lims = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    decision_line = -(model.weights[0] * lims + model.bias) / model.weights[1]

    plt.plot(lims, decision_line, color='darkgreen', lw=2.5, label='Граница решения')
    plt.title('Геометрия разделения признакового пространства')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


def render_roc_space(actual, probabilities):
    """Строит ROC-кривую в координатах TPR/FPR."""
    fpr, tpr, _ = metrics.roc_curve(actual, probabilities)
    score = metrics.roc_auc_score(actual, probabilities)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f'Модель (AUC={score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', alpha=0.5)
    plt.title('Пространство ROC-анализа')
    plt.xlabel('Доля ложноположительных (FPR)')
    plt.ylabel('Доля истинно положительных (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()