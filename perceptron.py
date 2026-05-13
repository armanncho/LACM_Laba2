import numpy as np

class Perceptron:
    def __init__(self): # вес модели
        self.w = None
        self.b = 0.0

    def sigmoid(self, z):
        z_bounded = np.clip(z, -250, 250) # защита
        return 1.0 / (1.0 + np.exp(-z_bounded))

    def forward(self, X): # измеряет "расстояние" между предсказанной вероятностью и реальной меткой
        activation = np.dot(X, self.w) + self.b
        return self.sigmoid(activation)

    def compute_loss(self, y_true, X, loss_type='bce', l2_reg=0.0):
        if loss_type == 'bce':
            preds = self.forward(X)
            eps = 1e-12
            preds_safe = np.clip(preds, eps, 1.0 - eps)
            loss = -(y_true * np.log(preds_safe) + (1.0 - y_true) * np.log(1.0 - preds_safe))
            return np.mean(loss) + 0.5 * l2_reg * np.sum(self.w ** 2)
        elif loss_type == 'hinge': # 0 и 1 временно переводятся в -1 и 1.
            linear_out = np.dot(X, self.w) + self.b
            y_calc = np.where(y_true == 0, -1, 1)
            loss = np.maximum(0, 1 - y_calc * linear_out)
            return np.mean(loss) + 0.5 * l2_reg * np.sum(self.w ** 2)

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, loss_type='bce', l2_reg=0.0, momentum=0.0):
        num_records, num_feats = X_train.shape
        self.w = np.random.normal(loc=0.0, scale=0.05, size=num_feats)
        self.b = 0.0

        v_w = np.zeros_like(self.w)
        v_b = 0.0

        metrics = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            idx = np.random.permutation(num_records)
            X_mixed = X_train[idx]
            y_mixed = y_train[idx]

            for batch_start in range(0, num_records, batch_size):
                batch_end = min(batch_start + batch_size, num_records)
                X_mini = X_mixed[batch_start:batch_end]
                y_mini = y_mixed[batch_start:batch_end]

                if loss_type == 'bce':
                    preds_mini = self.forward(X_mini)
                    diff = preds_mini - y_mini
                    grad_w = np.dot(X_mini.T, diff) / len(X_mini) + l2_reg * self.w
                    grad_b = np.mean(diff)
                elif loss_type == 'hinge':
                    y_mini_calc = np.where(y_mini == 0, -1, 1)
                    linear_out = np.dot(X_mini, self.w) + self.b
                    mask = (y_mini_calc * linear_out) < 1
                    grad_w = -np.dot(X_mini[mask].T, y_mini_calc[mask]) / len(X_mini) + l2_reg * self.w
                    grad_b = -np.sum(y_mini_calc[mask]) / len(X_mini)

                v_w = momentum * v_w + lr * grad_w
                v_b = momentum * v_b + lr * grad_b

                self.w -= v_w
                self.b -= v_b

            metrics['train_loss'].append(self.compute_loss(y_train, X_train, loss_type, l2_reg))
            metrics['val_loss'].append(self.compute_loss(y_val, X_val, loss_type, l2_reg))

        return metrics

    def predict(self, X):
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.forward(X)