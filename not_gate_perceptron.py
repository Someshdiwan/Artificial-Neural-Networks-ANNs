
Implementation of NOT Gate Using Single Layer Perceptron (ANN).

The NOT Gate has a single input and a single output, making it simpler than gates like AND or OR.

import numpy as np

# Define the step function (activation function)
def step_function(x):
    return 1 if x >= 0 else 0

# Define the Perceptron class for NOT Gate
class PerceptronNOT:
    def __init__(self):
        # Initialize weight and bias for a NOT gate
        self.weight = -1  # A negative weight to invert the input
        self.bias = 0.5   # A positive bias to shift the decision boundary

    def predict(self, input_value):
        # Compute weighted sum and apply activation function
        summation = self.weight * input_value + self.bias
        return step_function(summation)

# Input and output for NOT gate
inputs = np.array([0, 1])  # Input values: 0 and 1
target_output = np.array([1, 0])  # Expected output: NOT 0 = 1, NOT 1 = 0

# Initialize the Perceptron for NOT gate
not_gate = PerceptronNOT()

# Test the Perceptron
print("Testing Perceptron NOT gate:")
for input_value in inputs:
    output = not_gate.predict(input_value)
    print(f"Input: {input_value}, Output: {output}")


    
    