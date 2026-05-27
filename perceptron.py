import numpy as np


class Perceptron:
    def __init__(self, reg_lambda=0.0, momentum=0.9, cost_func='cross_entropy', mode='tiny_rand'):
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.cost_func = cost_func
        self.mode = mode

        self.weights = None
        self.bias = 0.0

    def _activation(self, z):
        # Используем логистическую функцию (сигмоиду)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self._activation(z)

    def predict_probs(self, inputs):
        return self.forward(inputs)

    def classify(self, inputs):
        return (self.forward(inputs) >= 0.5).astype(int)

    def _get_loss(self, y_true, y_pred, inputs=None):
        # L2-регуляризация (штраф за веса)
        l2_penalty = self.reg_lambda * np.sum(self.weights ** 2)

        if self.cost_func == 'hinge':
            y_mapped = 2 * y_true - 1
            z = np.dot(inputs, self.weights) + self.bias
            return np.mean(np.maximum(0, 1 - y_mapped * z)) + l2_penalty

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + l2_penalty

    def _init_params(self, n_features):
        init_strategies = {
            'zero': np.zeros(n_features),
            'large_rand': np.random.randn(n_features) * 10.0,
            'tiny_rand': np.random.randn(n_features) * 0.01
        }
        self.weights = init_strategies.get(self.mode, init_strategies['tiny_rand'])
        self.bias = 0.0

    def fit(self, X, y, X_test=None, y_test=None, total_epochs=100, step_size=0.1, chunk_size=32):
        self._init_params(X.shape[1])

        v_weights = np.zeros_like(self.weights)
        v_bias = 0.0

        loss_log, test_log = [], []

        for _ in range(total_epochs):
            indices = np.random.permutation(len(X))
            X_sh, y_sh = X[indices], y[indices]

            for i in range(0, len(X), chunk_size):
                X_batch, y_batch = X_sh[i:i + chunk_size], y_sh[i:i + chunk_size]
                n_samples = len(X_batch)

                if self.cost_func == 'hinge':
                    z = np.dot(X_batch, self.weights) + self.bias
                    mapped_y = 2 * y_batch - 1
                    is_active = (1 - mapped_y * z > 0).astype(float)
                    grad_w = (1 / n_samples) * np.dot(X_batch.T,
                                                      -is_active * mapped_y) + 2 * self.reg_lambda * self.weights
                    grad_b = (1 / n_samples) * np.sum(-is_active * mapped_y)
                else:
                    diff = self.forward(X_batch) - y_batch
                    grad_w = (1 / n_samples) * np.dot(X_batch.T, diff) + 2 * self.reg_lambda * self.weights
                    grad_b = (1 / n_samples) * np.sum(diff)

                # Шаг градиентного спуска с моментом
                v_weights = self.momentum * v_weights + step_size * grad_w
                v_bias = self.momentum * v_bias + step_size * grad_b

                self.weights -= v_weights
                self.bias -= v_bias

            loss_log.append(self._get_loss(y, self.forward(X), X))
            if X_test is not None:
                test_log.append(self._get_loss(y_test, self.forward(X_test), X_test))

        return loss_log, test_log