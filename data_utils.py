import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def prepare_data():
    features, labels = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=123,
        n_clusters_per_class=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.30, stratify=labels, random_state=123
    )
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test=None):
    mu_train = np.mean(X_train, axis=0)
    sigma_train = np.std(X_train, axis=0)

    train_normalized = (X_train - mu_train) / sigma_train
    if X_test is not None:
        test_normalized = (X_test - mu_train) / sigma_train
        return train_normalized, test_normalized
    return train_normalized


def generate_custom_data(pattern='linear', n_samples=500, noise=0.05):
    np.random.seed(123)
    if pattern == 'linear':
        X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
        X2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
        X = np.vstack((X1, X2))
        y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    elif pattern == 'xor':
        X = np.random.randn(n_samples, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    elif pattern == 'circle':
        X = np.random.randn(n_samples, 2) * 2
        radius = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        y = (radius > 2.0).astype(int)

    if noise > 0:
        flip_mask = np.random.rand(n_samples) < noise
        y[flip_mask] = 1 - y[flip_mask]

    return X, y