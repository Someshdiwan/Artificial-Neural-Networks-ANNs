
Implementation of AND Gate Using Single Layer Perceptron (ANN)

import numpy as np

# Define the activation function (step function)
def step_function(x):
    return 1 if x >= 0 else 0

# Define the perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)  # Including bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        # Add bias to the inputs
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return step_function(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                # Update weights and bias
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error

# Input and output for AND gate
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 0, 0, 1])

# Initialize perceptron
perceptron = Perceptron(input_size=2)

# Train perceptron
perceptron.train(training_inputs, labels)

# Test perceptron
print("Trained weights:", perceptron.weights)
print("Testing AND gate:")
for inputs in training_inputs:
    print(f"{inputs} -> {perceptron.predict(inputs)}")

//Initialization:

The weights are initialized to zeros, including a bias term.
The learning_rate defines the step size for weight updates.
The epochs parameter specifies the number of times the training set is processed.
Training:

For each training input, the perceptron computes the output using a weighted sum and the step activation function.
The weights are updated based on the error
(difference between actual and predicted output).

Testing:

After training, the perceptron is tested on the inputs to verify if it behaves as expected for an AND gate.

    