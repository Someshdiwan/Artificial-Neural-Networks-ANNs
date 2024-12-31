
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


Truth Table for NOT Gate
| Input | Output (NOT Input) |
|-------|---------------------|
|   0   |         1          |
|   1   |         0          |

How the Perceptron Works for NOT Gate

1. Logic Behind the NOT Gate:
   - A NOT gate produces the opposite of the input.
   - When the input is `0`, the output should be `1`.
   - When the input is `1`, the output should be `0`.

2. Perceptron Model for NOT Gate:**
   - Weight (`w`)**: A negative value (`-1`) is used to invert the input.
   - Bias (`b`)**: A positive value (`0.5`) ensures the output flips at the right threshold.

3. Mathematical Representation:
   The perceptron calculates the weighted sum:
   \[
   	ext{Sum} = (w \cdot 	ext{Input}) + b
   \]
   This sum is passed through a step activation function:
   \[
   	ext{Output} = 
   egin{cases} 
   1 & 	ext{if Sum} \geq 0 \
   0 & 	ext{if Sum} < 0 
   \end{cases}
   \]

   - For Input = 0:
     \[
     	ext{Sum} = (-1 \cdot 0) + 0.5 = 0.5 \quad \Rightarrow 	ext{Output} = 1
     \]
   - For Input = 1:
     \[
     	ext{Sum} = (-1 \cdot 1) + 0.5 = -0.5 \quad \Rightarrow 	ext{Output} = 0
     \]

1. Weight and Bias:
   - The weight `-1` inverts the input.
   - The bias `0.5` shifts the threshold so that the perceptron behaves like a NOT Gate.

2. Activation Function:
   - The step function outputs `1` when the sum is `>= 0` and `0` otherwise.

3. Testing:
   - The code iterates through the inputs `[0, 1]` and predicts the corresponding outputs.

    