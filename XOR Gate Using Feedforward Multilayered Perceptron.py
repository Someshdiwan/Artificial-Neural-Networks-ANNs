
The XOR (exclusive OR) gate outputs `1` if the two inputs are different, and `0` if they are the same. XOR is not linearly separable, so a single-layer perceptron cannot implement it. However, a multilayer perceptron (MLP) with a hidden layer can.

---

Truth Table for XOR Gate
| Input A | Input B | XOR Output |
|---------|---------|------------|
|    0    |    0    |      0     |
|    0    |    1    |      1     |
|    1    |    0    |      1     |
|    1    |    1    |      0     |

---
Architecture of the MLP

1. Input Layer:
   - Two input neurons (representing Input A and Input B).

2. Hidden Layer:
   - Two neurons with activation functions.

3. Output Layer:
   - One neuron with an activation function (step/sigmoid function).

4. Activation Function:
   - Sigmoid activation is used to introduce non-linearity.
---

Code Implementation

import numpy as np

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR gate implementation using MLP
class XORGateMLP:
    def __init__(self):
        # Initialize weights and biases for a 2-2-1 network
        self.input_weights = np.random.rand(2, 2)  # Weights for the input to hidden layer
        self.hidden_bias = np.random.rand(2)      # Bias for the hidden layer
        self.output_weights = np.random.rand(2, 1)  # Weights for hidden to output layer
        self.output_bias = np.random.rand(1)        # Bias for the output layer

    def forward_propagation(self, inputs):
        # Hidden layer computations
        self.hidden_input = np.dot(inputs, self.input_weights) + self.hidden_bias
        self.hidden_output = sigmoid(self.hidden_input)

        # Output layer computations
        self.output_input = np.dot(self.hidden_output, self.output_weights) + self.output_bias
        self.output = sigmoid(self.output_input)
        return self.output

    def backward_propagation(self, inputs, target_output, learning_rate):
        # Calculate output error and its gradient
        output_error = target_output - self.output
        output_gradient = output_error * sigmoid_derivative(self.output)

        # Propagate error to the hidden layer
        hidden_error = output_gradient.dot(self.output_weights.T)
        hidden_gradient = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.output_weights += self.hidden_output.T.dot(output_gradient) * learning_rate
        self.output_bias += np.sum(output_gradient, axis=0) * learning_rate
        self.input_weights += inputs.T.dot(hidden_gradient) * learning_rate
        self.hidden_bias += np.sum(hidden_gradient, axis=0) * learning_rate

    def train(self, inputs, target_output, learning_rate=0.5, epochs=10000):
        for _ in range(epochs):
            self.forward_propagation(inputs)
            self.backward_propagation(inputs, target_output, learning_rate)

    def predict(self, inputs):
        return self.forward_propagation(inputs)

# XOR input and output
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
target_output = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Initialize the XOR gate MLP
xor_gate = XORGateMLP()

# Train the MLP
xor_gate.train(inputs, target_output, learning_rate=0.1, epochs=10000)

# Test the MLP
print("Testing XOR Gate MLP:")
for input_data, target in zip(inputs, target_output):
    prediction = xor_gate.predict(input_data)
    print(f"Input: {input_data}, Predicted Output: {np.round(prediction[0], 2)}, Target: {target[0]}")


---


Explanation of the Code

1. Weight Initialization:
   - Random weights and biases are used to initialize the MLP.

2. Forward Propagation:
   - Computes outputs of the hidden and output layers using the sigmoid activation function.

3. Backward Propagation:
   - Updates weights and biases based on the error using gradient descent.

4. Training:
   - The model is trained over multiple epochs to minimize the error.

5. Prediction:
   - After training, the MLP is tested using XOR inputs, and the outputs are compared with the expected results.

---

Output

After training for a sufficient number of epochs, the network should approximate the XOR function:

```
Testing XOR Gate MLP:
Input: [0 0], Predicted Output: 0.0, Target: 0
Input: [0 1], Predicted Output: 1.0, Target: 1
Input: [1 0], Predicted Output: 1.0, Target: 1
Input: [1 1], Predicted Output: 0.0, Target: 0
```

---
    