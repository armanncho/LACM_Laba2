import numpy as np
from perceptron import Perceptron
from metrics import calculate_accuracy


def k_fold_cross_validation(X, y, k=5, epochs=100, lr=0.1, batch_size=32):
    fold_len = len(X) // k
    acc_scores = []

    for fold_idx in range(k):
        idx_start = fold_idx * fold_len
        idx_end = (fold_idx + 1) * fold_len

        X_valid = X[idx_start:idx_end]
        y_valid = y[idx_start:idx_end]

        X_train_fold = np.concatenate((X[:idx_start], X[idx_end:]))
        y_train_fold = np.concatenate((y[:idx_start], y[idx_end:]))

        clf = Perceptron()
        clf.fit(X_train_fold, y_train_fold, X_valid, y_valid, epochs=epochs, lr=lr, batch_size=batch_size)

        fold_acc = calculate_accuracy(y_valid, clf.predict(X_valid))
        acc_scores.append(fold_acc)

    return np.mean(acc_scores), np.std(acc_scores)


def grid_search_cv(X, y):
    best_acc = 0
    best_params = {}

    lrs = [0.01, 0.1, 0.5]
    batches = [16, 32, 64]

    for lr in lrs:
        for b in batches:
            mean_acc, std_acc = k_fold_cross_validation(X, y, k=5, lr=lr, batch_size=b)
            print(f"LR={lr}, Batch={b} -> Acc={mean_acc:.4f} (+/- {std_acc:.4f})")
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = {'lr': lr, 'batch_size': b}

    return best_params