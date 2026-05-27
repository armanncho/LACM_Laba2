import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron


def analyze_step_size_impact(X_train, y_train, X_test, y_test):
    """Исследование зависимости скорости сходимости от шага обучения (step_size)."""
    print("\n[INFO] Запуск анализа влияния шага обучения (step_size)...")
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]

    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        # Используем твой новый API: step_size вместо lr
        model = Perceptron()
        history, _ = model.fit(X_train, y_train, total_epochs=58, step_size=lr, chunk_size=32)

        score = np.mean(model.classify(X_test) == y_test) * 100
        print(f" > Шаг {lr:<6} | Итоговая точность: {score:.2f}%")
        plt.plot(history, label=f'step={lr}')

    plt.title('Анализ чувствительности к step_size')
    plt.xlabel('Эпоха')
    plt.ylabel('Функция потерь')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def analyze_momentum_dynamics(X_train, y_train, X_test, y_test):
    """Сравнение стандартного SGD с методом моментов."""
    print("\n[INFO] Сравнение динамики сходимости SGD и Momentum...")
    momentum_vals = [0.0, 0.5, 0.9, 0.99]

    plt.figure(figsize=(10, 6))
    for m in momentum_vals:
        model = Perceptron(momentum=m)
        history, _ = model.fit(X_train, y_train, total_epochs=100, step_size=0.1, chunk_size=32)

        label_text = 'SGD' if m == 0 else f'Momentum (coeff={m})'
        plt.plot(history, label=label_text)
        print(f" > {label_text:<20} | Финальный loss: {history[-1]:.4f}")

    plt.title('Эффективность накопления импульса (Momentum)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def compare_loss_strategies(X_train, y_train, X_test, y_test):
    """Сравнение логистической функции потерь и Hinge loss."""
    print("\n[INFO] Сравнение стратегий потерь (Cross-Entropy vs Hinge)...")
    for strategy in ['cross_entropy', 'hinge']:
        model = Perceptron(cost_func=strategy)
        model.fit(X_train, y_train, total_epochs=50, step_size=0.1, chunk_size=32)
        acc = np.mean(model.classify(X_test) == y_test) * 100
        print(f" > Стратегия {strategy:<15} | Точность: {acc:.2f}%")


def evaluate_weight_decay(X_train, y_train, X_test, y_test):
    """Оценка влияния L2-регуляризации на норму весов."""
    print("\n[INFO] Оценка регуляризации (L2 Weight Decay)...")
    reg_coeffs = [0.0, 0.01, 0.1, 1.0]
    for reg in reg_coeffs:
        model = Perceptron(reg_lambda=reg)
        model.fit(X_train, y_train, total_epochs=50, step_size=0.1, chunk_size=32)
        norm = np.linalg.norm(model.weights)
        print(f" > Lambda {reg:<6} | Норма весов: {norm:.4f}")


def run_all_diagnostics(X_train, y_train, X_test, y_test):
    """Комплексный запуск исследовательских процедур."""
    analyze_step_size_impact(X_train, y_train, X_test, y_test)
    analyze_momentum_dynamics(X_train, y_train, X_test, y_test)
    compare_loss_strategies(X_train, y_train, X_test, y_test)
    evaluate_weight_decay(X_train, y_train, X_test, y_test)