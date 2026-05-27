import numpy as np
import matplotlib.pyplot as plt

# Импортируем обновленные модули
from data_utils import get_prepared_data, generate_custom_data
from sklearn.model_selection import train_test_split
from perceptron import Perceptron
from metrics import display_performance_report, render_roc_space, render_learning_curves, draw_boundary_map
from experiment import run_all_diagnostics
from cross_validation import perform_grid_optimization


def run_research_pipeline():
    """Главная точка входа исследовательской системы."""

    print("\n--- ИССЛЕДОВАТЕЛЬСКАЯ СИСТЕМА ПЕРЦЕПТРОНА ---")
    print("Выберите конфигурацию данных:")
    print(" 1: Стандартный набор данных (make_classification)")
    print(" 2: Синтетические данные (Linear / XOR / Circle)")
    print(" 0: Завершение работы")

    while True:
        choice = input(">> Ваш выбор: ")
        if choice in ['1', '2', '0']: break
        print("Некорректный ввод. Попробуйте снова.")

    if choice == '0': return

    # Инициализация данных
    if choice == '1':
        X_train, X_test, y_train, y_test = get_prepared_data()
        X_full = np.vstack((X_train, X_test))
        y_full = np.hstack((y_train, y_test))
    else:
        print("\nТипы: 1: Linear, 2: XOR, 3: Circle")
        type_map = {'1': 'linear', '2': 'xor', '3': 'circle'}
        choice_type = input("Введите тип (1-3): ")

        raw_x, raw_y = generate_custom_data(dataset_type=type_map.get(choice_type, 'linear'))
        X_train, X_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=0.3, stratify=raw_y)

        # Ручная нормализация
        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
        X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma
        X_full, y_full = raw_x, raw_y

    # Этап 1: Оптимизация параметров
    print("\n[STEP 1] Запуск этапа автоматической оптимизации...")
    best_step, best_chunk, _, _ = perform_grid_optimization(X_full, y_full)

    # Этап 2: Финальное обучение
    model = Perceptron(momentum=0.9)
    print(f"\n[STEP 2] Обучение модели (step={best_step}, chunk={best_chunk})...")
    train_h, test_h = model.fit(
        X_train, y_train, X_test, y_test,
        total_epochs=100, step_size=best_step, chunk_size=best_chunk
    )

    # Этап 3: Анализ качества
    render_learning_curves(train_h, test_h)
    draw_boundary_map(model, X_test, y_test)

    preds = model.classify(X_test)
    display_performance_report(y_test, preds, model.predict_probs(X_test))
    render_roc_space(y_test, model.predict_probs(X_test))

    # Этап 4: Глубокие исследования
    print("\n[STEP 3] Запуск диагностических процедур...")
    run_all_diagnostics(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    run_research_pipeline()