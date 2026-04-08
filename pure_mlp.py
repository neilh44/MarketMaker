#!/usr/bin/env python3
"""
Pure Python MLP Classifier (no numpy, no scikit-learn)
Corrected backpropagation with proper delta handling.
"""

import math
import random

class PurePythonMLP:
    def __init__(self, input_size=540, hidden_sizes=(64, 32), output_size=1):
        self.layers = []
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        for i in range(len(sizes) - 1):
            limit = math.sqrt(6.0 / (sizes[i] + sizes[i+1]))
            weights = [[random.uniform(-limit, limit) for _ in range(sizes[i])]
                       for _ in range(sizes[i+1])]
            biases = [0.0] * sizes[i+1]
            self.layers.append((weights, biases))

    def _sigmoid(self, x):
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def _sigmoid_derivative(self, output):
        return output * (1.0 - output)

    def forward(self, x):
        activations = [x]
        for weights, biases in self.layers:
            z = []
            for w_row, b in zip(weights, biases):
                val = sum(w_i * a_i for w_i, a_i in zip(w_row, activations[-1])) + b
                z.append(val)
            a = [self._sigmoid(v) for v in z]
            activations.append(a)
        return activations

    def predict(self, x):
        acts = self.forward(x)
        return acts[-1][0]

    def fit(self, X, y, epochs=10, lr=0.01, batch_size=32):
        n_samples = len(X)
        for epoch in range(epochs):
            total_loss = 0.0
            indices = list(range(n_samples))
            random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start+batch_size]

                # Initialize gradient accumulators
                grad_w_accum = []
                grad_b_accum = []
                for weights, biases in self.layers:
                    grad_w_accum.append([[0.0] * len(weights[0]) for _ in range(len(weights))])
                    grad_b_accum.append([0.0] * len(biases))

                for idx in batch_idx:
                    # Forward pass
                    acts = self.forward(X[idx])
                    output = acts[-1][0]
                    target = y[idx]
                    loss = (output - target) ** 2
                    total_loss += loss

                    # --- Backward pass (corrected) ---
                    # deltas[i] will be list of deltas for neurons in layer i
                    deltas = [None] * len(self.layers)

                    # Output layer delta (only one neuron)
                    delta_output = (output - target) * self._sigmoid_derivative(output)
                    deltas[-1] = [delta_output]

                    # Backpropagate to hidden layers
                    for l in range(len(self.layers)-2, -1, -1):
                        next_weights, _ = self.layers[l+1]
                        next_deltas = deltas[l+1]
                        curr_acts = acts[l+1]  # activations of current layer
                        new_deltas = []
                        for j in range(len(curr_acts)):
                            # sum over neurons in next layer
                            error = sum(next_weights[i][j] * next_deltas[i] for i in range(len(next_weights)))
                            new_deltas.append(error * self._sigmoid_derivative(curr_acts[j]))
                        deltas[l] = new_deltas

                    # Accumulate gradients
                    for l in range(len(self.layers)):
                        weights, biases = self.layers[l]
                        prev_acts = acts[l]   # input to layer l
                        curr_deltas = deltas[l]
                        for i in range(len(weights)):
                            for j in range(len(weights[i])):
                                grad_w_accum[l][i][j] += curr_deltas[i] * prev_acts[j]
                            grad_b_accum[l][i] += curr_deltas[i]

                # Update weights and biases
                batch_len = len(batch_idx)
                for l in range(len(self.layers)):
                    weights, biases = self.layers[l]
                    for i in range(len(weights)):
                        for j in range(len(weights[i])):
                            weights[i][j] -= lr * grad_w_accum[l][i][j] / batch_len
                        biases[i] -= lr * grad_b_accum[l][i] / batch_len

            avg_loss = total_loss / n_samples
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# -------------------------------------------------------------------
# Generate synthetic data and test
# -------------------------------------------------------------------
def generate_synthetic_data(n_samples=1000, input_dim=540):
    X = [[random.gauss(0, 1) for _ in range(input_dim)] for _ in range(n_samples)]
    y = [1 if sum(x) > 0 else 0 for x in X]
    return X, y

print("Generating synthetic data...")
X, y = generate_synthetic_data(n_samples=500)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training pure Python MLP...")
mlp = PurePythonMLP(input_size=540, hidden_sizes=(32, 16), output_size=1)
mlp.fit(X_train, y_train, epochs=5, lr=0.01)

correct = 0
for x, true_y in zip(X_test, y_test):
    pred = mlp.predict(x)
    if (pred > 0.5) == true_y:
        correct += 1
print(f"Test Accuracy: {correct/len(X_test):.4f}")
