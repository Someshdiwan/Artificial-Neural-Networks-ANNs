

The implementation of an OR Gate using a single-layer perceptron is quite similar to the implementation of an AND Gate. 
The key difference lies in the weights and bias values that the perceptron learns during training, as they reflect the logic of the OR Gate.

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

# Input and output for OR gate
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 1, 1, 1])

# Initialize perceptron
perceptron = Perceptron(input_size=2)

# Train perceptron
perceptron.train(training_inputs, labels)

# Test perceptron
print("Trained weights:", perceptron.weights)
print("Testing OR gate:")
for inputs in training_inputs:
    print(f"{inputs} -> {perceptron.predict(inputs)}")

// Note
   - The OR Gate has two inputs and one output.
   - It outputs `1` if **at least one input is `1`**, otherwise it outputs `0`.
   - Truth table:

     | Input 1 | Input 2 | Output |
     |---------|---------|--------|
     |    0    |    0    |   0    |
     |    0    |    1    |   1    |
     |    1    |    0    |   1    |
     |    1    |    1    |   1    |

     
3.Explanation of the Code

Initialization:
- The perceptron is initialized with zero weights, including a bias term.
- Learning rate (`0.1`) and the number of epochs (`100`) control the weight updates during training.

Training:
- For each training input, the perceptron calculates the weighted sum and applies the step activation function.
- If the output doesn’t match the label, the weights are updated based on the error (`label - prediction`).

Testing:
- After training, the perceptron is tested on the input values of the OR Gate to verify its functionality.

4.Expected Output**
After training the perceptron, it should learn the weights and bias for the OR Gate. When tested, the output will match the OR Gate's truth table:

5 Key Insights
- The perceptron successfully learns the weights and bias required to implement the OR Gate logic.
- The step activation function ensures binary outputs (0 or 1).
- This is a simple example of how a perceptron can act as a linear classifier for linearly separable data (e.g., OR Gate).

