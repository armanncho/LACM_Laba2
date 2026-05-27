import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_prepared_data():
    """Формирует и нормализует выборку для базового классификатора."""
    # Генерация синтетических данных с заданными параметрами
    features, labels = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )

    # Разбиение с сохранением долей классов
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.3, stratify=labels, random_state=42
    )

    # Применение Z-score нормализации
    # Используем статистику обучающей выборки для обоих наборов
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)

    # Защита от деления на нулевое отклонение
    sigma = np.maximum(sigma, 1e-8)

    train_x_scaled = (train_x - mu) / sigma
    test_x_scaled = (test_x - mu) / sigma

    return [train_x_scaled, test_x_scaled, train_y, test_y]


def generate_custom_data(dataset_type='linear', n_samples=500, noise_rate=0.05):
    """
    Создает кастомные наборы данных (linear, xor, circle).
    """
    np.random.seed(42)

    if dataset_type == 'linear':
        group_size = n_samples // 2
        # Генерируем два гауссовых облака
        data_0 = np.random.randn(group_size, 2) + np.array([2, 2])
        data_1 = np.random.randn(group_size, 2) + np.array([-2, -2])
        X = np.concatenate([data_0, data_1])
        y = np.concatenate([np.zeros(group_size), np.ones(group_size)])

    elif dataset_type == 'xor':
        # XOR распределение: 4 группы в углах
        chunk = n_samples // 4
        X = np.concatenate([
            np.random.randn(chunk, 2) * 0.7 + np.array([2, 2]),
            np.random.randn(chunk, 2) * 0.7 + np.array([-2, -2]),
            np.random.randn(chunk, 2) * 0.7 + np.array([2, -2]),
            np.random.randn(chunk, 2) * 0.7 + np.array([-2, 2])
        ])
        y = np.concatenate([np.zeros(2 * chunk), np.ones(2 * chunk)])

    elif dataset_type == 'circle':
        # Концентрические окружности
        half = n_samples // 2
        # Внутренний круг
        radii_0 = np.random.rand(half) * 1.5
        theta_0 = np.random.rand(half) * 2 * np.pi
        X0 = np.stack([radii_0 * np.cos(theta_0), radii_0 * np.sin(theta_0)], axis=1)
        # Внешнее кольцо
        radii_1 = 2.5 + np.random.rand(half) * 1.5
        theta_1 = np.random.rand(half) * 2 * np.pi
        X1 = np.stack([radii_1 * np.cos(theta_1), radii_1 * np.sin(theta_1)], axis=1)

        X = np.concatenate([X0, X1])
        y = np.concatenate([np.zeros(half), np.ones(half)])

    # Внесение случайных искажений в метки
    if noise_rate > 0:
        flip_count = int(n_samples * noise_rate)
        flip_indices = np.random.choice(n_samples, flip_count, replace=False)
        y[flip_indices] = 1 - y[flip_indices]

    return X, y