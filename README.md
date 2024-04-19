# Neural-Network

This repository is dedicated to providing educational resources and code examples on how to build neural networks from scratch in Python. It aims to help users understand the foundational concepts of neural network architectures, including how they learn and make predictions.

## Prerequisites

- Python 3.6 or higher
- Basic knowledge of linear algebra and calculus

## Installation

No specific library installations are required for the basic examples as they use pure Python. For more advanced examples, you may need NumPy:

```bash
pip install numpy
```

## Example - A Simple Neural Network

Here's an example of a very simple neural network without using any deep learning frameworks, intended to illustrate the basic concepts.

### `simple_nn.py`

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return cost

# Example usage
np.random.seed(1)
input_size = 3  # Number of features
hidden_size = 4  # Number of hidden units
output_size = 1  # Number of output units

X = np.random.randn(input_size, 1)
Y = np.array([[1]])

parameters = initialize_parameters(input_size, hidden_size, output_size)
A2, cache = forward_propagation(X, parameters)
cost = compute_cost(A2, Y)
print(f"Cost: {cost}")
```

## Contributing

Contributions to this project are welcome! Please fork the repository, add your contributions in a new branch, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
