"""
Standard backpropagation baseline network for comparison with MoE.
Simple 2-layer MLP with sigmoid activations, used in Table 1 experiments.
"""

import numpy as np


class BackpropNetwork:
    """
    Standard feedforward network with one hidden layer.
    Trained with batch gradient descent for fair comparison with MoE.
    Architecture: input -> hidden (sigmoid) -> output (sigmoid)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, seed=None):
        self._rng = np.random.RandomState(None if seed is None else int(seed))
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        # Xavier-style initialization
        scale1 = 1.0 / np.sqrt(self.input_dim)
        scale2 = 1.0 / np.sqrt(self.hidden_dim)
        self.W1 = (self._rng.randn(self.input_dim, self.hidden_dim) * scale1).astype(np.float32)
        self.b1 = (self._rng.randn(self.hidden_dim) * scale1).astype(np.float32)
        self.W2 = (self._rng.randn(self.hidden_dim, self.output_dim) * scale2).astype(np.float32)
        self.b2 = (self._rng.randn(self.output_dim) * scale2).astype(np.float32)
        self.x = None  # cached for backward pass
        self.h = None  # hidden activations
        self.output = None  # output activations

    def forward(self, x):
        """Forward pass through 2-layer network with sigmoid activations"""
        x = x.astype(np.float32, copy=False)
        self.x = x
        z1 = x @ self.W1 + self.b1
        self.h = self._sigmoid(z1)
        z2 = self.h @ self.W2 + self.b2
        self.output = self._sigmoid(z2)
        return self.output

    def backward(self, target):
        """
        Backpropagation using cross-entropy gradient with sigmoid outputs.
        Returns gradients for all parameters.
        """
        y = target.astype(np.float32, copy=False)
        y_hat = self.output
        h = self.h
        x = self.x
        # Output layer gradients (cross-entropy + sigmoid derivative)
        delta2 = (y_hat - y) * y_hat * (1.0 - y_hat)
        grad_W2 = h.T @ delta2
        grad_b2 = np.sum(delta2, axis=0)
        # Hidden layer gradients (backprop through sigmoid)
        delta1 = (delta2 @ self.W2.T) * h * (1.0 - h)
        grad_W1 = x.T @ delta1
        grad_b1 = np.sum(delta1, axis=0)
        return grad_W1.astype(np.float32), grad_b1.astype(np.float32), \
               grad_W2.astype(np.float32), grad_b2.astype(np.float32)

    def update(self, grad_W1, grad_b1, grad_W2, grad_b2, learning_rate):
        lr = np.float32(learning_rate)
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2

    def train_step(self, x, target, learning_rate):
        self.forward(x)
        gW1, gb1, gW2, gb2 = self.backward(target)
        self.update(gW1, gb1, gW2, gb2, learning_rate)

    def predict(self, x):
        return self.forward(x)

    def predict_class(self, x):
        out = self.forward(x)
        return np.argmax(out, axis=1)

    def compute_accuracy(self, x, target):
        preds = self.predict_class(x)
        labels = np.argmax(target, axis=1)
        return np.mean(preds == labels)

    def compute_squared_error(self, x, target):
        y = target.astype(np.float32, copy=False)
        y_hat = self.forward(x)
        return np.mean((y_hat - y) ** 2)

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid to avoid overflow"""
        out = np.empty_like(x, dtype=np.float32)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg])
        out[neg] = ex / (1.0 + ex)
        return out