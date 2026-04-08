#!/usr/bin/env python3
"""
Pure Python Logistic Regression (no numpy, no scikit-learn)
"""

import math
import random

class LogisticRegression:
    def __init__(self, input_size, lr=0.01):
        self.lr = lr
        # Small random initialization
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(input_size)]
        self.bias = 0.0

    def _sigmoid(self, x):
        # Clip to prevent overflow
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def predict_proba(self, x):
        """Return probability of class 1."""
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self._sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def fit(self, X, y, epochs=20, batch_size=32, verbose=True):
        n_samples = len(X)
        for epoch in range(epochs):
            total_loss = 0.0
            # Shuffle
            indices = list(range(n_samples))
            random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start+batch_size]
                grad_w = [0.0] * len(self.weights)
                grad_b = 0.0
                for idx in batch_idx:
                    xi = X[idx]
                    yi = y[idx]
                    prob = self.predict_proba(xi)
                    error = prob - yi
                    for j in range(len(self.weights)):
                        grad_w[j] += error * xi[j]
                    grad_b += error
                    # Cross-entropy loss
                    total_loss += -(yi * math.log(prob + 1e-8) + (1-yi) * math.log(1-prob + 1e-8))
                # Update
                for j in range(len(self.weights)):
                    self.weights[j] -= self.lr * grad_w[j] / len(batch_idx)
                self.bias -= self.lr * grad_b / len(batch_idx)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_samples:.6f}")
