import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression():
    def __init__(self):
        self.losses = []
        self.train_accuracies = []

    def fit(self, x, y, epochs):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        self.learqing_rate = 0.1

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def compute_loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred + 1e-9) + (1-y) * np.log(1 - y_pred + 1e-9))

    def compute_gradients(self, x, y, y_pred):
        # derivative of binary cross entropy
        gradient_b = np.mean(y_pred - y)
        gradients_w = np.matmul(x.transpose(), y_pred - y)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - self.learqing_rate * error_w
        self.bias = self.bias - self.learqing_rate * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
