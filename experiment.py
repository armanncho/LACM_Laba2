from perceptron import Perceptron
from metrics import calculate_accuracy, plot_loss, plot_decision_boundary
from data_utils import generate_custom_data, standardize_data

def experiment_learning_rate(X_tr, y_tr, X_te, y_te, learning_rates=(0.001, 0.01, 0.5, 1.0)):
    for rate in learning_rates:
        net = Perceptron()
        stats = net.fit(X_tr, y_tr, X_te, y_te, epochs=100, lr=rate, batch_size=32)
        acc = calculate_accuracy(y_te, net.predict(X_te))
        print(f"LR = {rate} -> Accuracy: {acc:.4f}")

def experiment_custom_data():
    for pattern in ['linear', 'xor', 'circle']:
        X, y = generate_custom_data(pattern=pattern)
        X_norm = standardize_data(X)
        net = Perceptron()
        net.fit(X_norm, y, X_norm, y, epochs=150, lr=0.1, batch_size=16)
        acc = calculate_accuracy(y, net.predict(X_norm))
        print(f"Паттерн: {pattern.upper()} -> Точность: {acc:.4f}")
        plot_decision_boundary(net, X_norm, y)

def experiment_loss_functions(X_tr, y_tr, X_te, y_te):
    for l_type in ['bce', 'hinge']:
        net = Perceptron()
        stats = net.fit(X_tr, y_tr, X_te, y_te, epochs=100, lr=0.05, batch_size=32, loss_type=l_type)
        acc = calculate_accuracy(y_te, net.predict(X_te))
        print(f"Loss = {l_type.upper()} -> Accuracy: {acc:.4f}")
        plot_loss(stats, title=f'Loss: {l_type.upper()}')

def experiment_momentum(X_tr, y_tr, X_te, y_te):
    for beta in [0.0, 0.5, 0.9, 0.99]:
        net = Perceptron()
        stats = net.fit(X_tr, y_tr, X_te, y_te, epochs=50, lr=0.01, batch_size=16, momentum=beta)
        acc = calculate_accuracy(y_te, net.predict(X_te))
        print(f"Momentum = {beta} -> Accuracy: {acc:.4f}")
        plot_loss(stats, title=f'Momentum beta={beta}')

def experiment_l2_regularization(X_tr, y_tr, X_te, y_te):
    for l2 in [0.0, 0.01, 0.1, 1.0]:
        net = Perceptron()
        net.fit(X_tr, y_tr, X_te, y_te, epochs=100, lr=0.1, batch_size=32, l2_reg=l2)
        acc = calculate_accuracy(y_te, net.predict(X_te))
        print(f"L2 = {l2} -> Accuracy: {acc:.4f}, Norm w: {sum(net.w**2):.4f}")