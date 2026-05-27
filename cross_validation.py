import numpy as np
from sklearn.model_selection import KFold
from perceptron import Perceptron


def evaluate_model_robustness(X, y, k_folds=5, step_size=0.1, chunk_size=32, total_epochs=50, show_logs=True):
    """
    Проводит оценку устойчивости модели методом K-Fold кросс-валидации.
    """
    if show_logs:
        print(f"\n[SCAN] Начало {k_folds}-кратной кросс-валидации...")
        print(f"Конфигурация: epochs={total_epochs}, step={step_size}, chunk={chunk_size}")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    performance_metrics = []

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Автоматическая стандартизация текущего фолда
        mu = X_train.mean(axis=0)
        sigma = np.maximum(X_train.std(axis=0), 1e-8)
        X_train, X_val = (X_train - mu) / sigma, (X_val - mu) / sigma

        # Создание и обучение модели
        model = Perceptron(momentum=0.9)
        model.fit(X_train, y_train, total_epochs=total_epochs, step_size=step_size, chunk_size=chunk_size)

        # Оценка точности
        accuracy = np.mean(model.classify(X_val) == y_val) * 100
        if show_logs:
            print(f" > Фолд {i + 1}: Точность составила {accuracy:.2f}%")
        performance_metrics.append(accuracy)

    return np.mean(performance_metrics), np.std(performance_metrics)


def perform_grid_optimization(X, y, k_folds=5):
    """
    Выполняет поиск оптимальных гиперпараметров по сетке (Grid Search).
    """
    print("\n" + "=" * 50)
    print(" АНАЛИЗ ГИПЕРПАРАМЕТРОВ (GRID OPTIMIZATION) ")
    print("=" * 50)

    # Задаем сетку параметров (используя наши новые имена аргументов)
    candidate_lrs = [0.01, 0.05, 0.1]
    candidate_chunks = [16, 32, 64]

    best_score = -1.0
    best_config = (None, None, 0.0, 0.0)  # step, chunk, mean, std

    header = f"{'Step Size':<15} | {'Chunk Size':<12} | {'Avg Accuracy':<15} | {'Std Dev':<10}"
    print(header)
    print("-" * len(header))

    # Фиксируем seed для повторяемости результатов
    np.random.seed(42)

    for lr in candidate_lrs:
        for chunk in candidate_chunks:
            avg_acc, std_acc = evaluate_model_robustness(
                X, y, k_folds=k_folds, step_size=lr, chunk_size=chunk, total_epochs=50, show_logs=False
            )

            # Перевод в дробный формат для отчета
            score_mean, score_std = avg_acc / 100.0, std_acc / 100.0
            print(f"{lr:<15} | {chunk:<12} | {score_mean:<15.4f} | {score_std:<10.4f}")

            if score_mean > best_score:
                best_score = score_mean
                best_config = (lr, chunk, score_mean, score_std)

    print("-" * len(header))
    print(f"Оптимальная конфигурация: Step={best_config[0]}, Chunk={best_config[1]}")
    print(f"Максимальная точность: {best_config[2]:.4f} (±{best_config[3]:.4f})")
    print("=" * 50 + "\n")

    return best_config[0], best_config[1], best_config[2], best_config[3]