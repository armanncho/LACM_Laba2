from data_utils import prepare_data, standardize_data
from perceptron import Perceptron
from metrics import calculate_accuracy, calculate_advanced_metrics, plot_loss, plot_decision_boundary, plot_roc
from experiment import experiment_learning_rate, experiment_custom_data, experiment_loss_functions, \
    experiment_momentum, experiment_l2_regularization
from cross_validation import grid_search_cv


def main():
    print("--- 1. Инициализация и подготовка данных ---")
    X_tr, X_te, y_tr, y_te = prepare_data()
    X_tr_norm, X_te_norm = standardize_data(X_tr, X_te)

    print("\n--- 2. Старт обучения базовой модели ---")
    base_model = Perceptron()
    loss_history = base_model.fit(X_tr_norm, y_tr, X_te_norm, y_te, epochs=100, lr=0.1, batch_size=32)

    print("\n--- 3. Оценка качества и доп. метрики ---")
    preds_te = base_model.predict(X_te_norm)
    probs_te = base_model.predict_proba(X_te_norm)

    acc_test = calculate_accuracy(y_te, preds_te)
    p, r, f1, roc_auc = calculate_advanced_metrics(y_te, preds_te, probs_te)

    print(f"Accuracy:  {acc_test:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\n--- 4. Отрисовка графиков ---")
    plot_loss(loss_history)
    plot_decision_boundary(base_model, X_te_norm, y_te, highlight_errors=True)
    plot_roc(y_te, probs_te)

    print("\n--- 5. Кросс-валидация ---")
    best_params = grid_search_cv(X_tr_norm, y_tr)
    print(f"Лучшие параметры: {best_params}")

    print("\n--- 6. Дополнительные эксперименты ---")
    print("\n[Моментум]")
    experiment_momentum(X_tr_norm, y_tr, X_te_norm, y_te)

    print("\n[Функции потерь: BCE vs Hinge]")
    experiment_loss_functions(X_tr_norm, y_tr, X_te_norm, y_te)

    print("\n[L2 Регуляризация]")
    experiment_l2_regularization(X_tr_norm, y_tr, X_te_norm, y_te)

    print("\n[Кастомные датасеты]")
    experiment_custom_data()


if __name__ == "__main__":
    main()