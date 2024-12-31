

XOR Gate Using Backpropagation in Multilayered Perceptron.

This implementation uses numpy for matrix operations and includes forward propagation, backpropagation, and training processes.

How It Works

Architecture:

Input Layer: 2 neurons (for two inputs of XOR gate).
Hidden Layer: 2 neurons.
Output Layer: 1 neuron (for XOR result).
Activation Function:

Sigmoid is used for non-linearity.

Training:
Forward propagation calculates predictions.
Backpropagation calculates errors and adjusts weights and biases using the gradient descent algorithm.
Testing:

The network predicts XOR gate outputs after training.


Code Implementation:

import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR Gate Multilayer Perceptron
class XORGateBackpropMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_weights = np.random.rand(input_size, hidden_size)
        self.hidden_bias = np.random.rand(hidden_size)
        self.output_weights = np.random.rand(hidden_size, output_size)
        self.output_bias = np.random.rand(output_size)

    def forward_propagation(self, inputs):
        # Hidden layer
        self.hidden_input = np.dot(inputs, self.input_weights) + self.hidden_bias
        self.hidden_output = sigmoid(self.hidden_input)
        # Output layer
        self.output_input = np.dot(self.hidden_output, self.output_weights) + self.output_bias
        self.output = sigmoid(self.output_input)
        return self.output

    def backward_propagation(self, inputs, target_output, learning_rate):
        # Calculate errors
        output_error = target_output - self.output
        output_gradient = output_error * sigmoid_derivative(self.output)
        
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

# Inputs and target outputs for XOR gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([[0], [1], [1], [0]])

# Create and train the MLP
xor_mlp = XORGateBackpropMLP(input_size=2, hidden_size=2, output_size=1)
xor_mlp.train(inputs, target_output, learning_rate=0.1, epochs=10000)

# Test the XOR Gate MLP
print("Testing XOR Gate with Backpropagation:")
for input_data, target in zip(inputs, target_output):
    prediction = xor_mlp.predict(input_data)
    print(f"Input: {input_data}, Predicted Output: {np.round(prediction[0], 2)}, Target: {target[0]}")

    

Testing XOR Gate with Backpropagation:

Input: [0 0], Predicted Output: 0.02, Target: 0
Input: [0 1], Predicted Output: 0.98, Target: 1
Input: [1 0], Predicted Output: 0.97, Target: 1
Input: [1 1], Predicted Output: 0.01, Target: 0

Error Backpropagation in MLP:

Error backpropagation is a supervised learning algorithm used for training neural networks. 

It involves:
Forward Propagation: Calculate outputs using current weights.
Error Calculation: Compute the difference between predicted and actual outputs.
Backward Propagation: Adjust weights and biases based on the gradient of the error.
Update Weights: Use gradient descent to minimize the error.

---
    