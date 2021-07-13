#Alles, was im Buech staat, wird da vo mier umgsetzt. Die Datei = NNFS Sandbox

import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5], [5.0, 4.3, 2.3, 5.2], [-1.5, 3.2, 3.5, 9.0]]
gwicht = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.9, 0.3, -0.5], [0.26, 0.27, 0.28, 0.29]]
biases = [2.0, 3.0, 0.5]

gwicht2 = [[0.1, 0.4, 0.3],[0.7, 0.2, 0.1],[0.8, 0.7, 0.6]]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(gwicht).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(gwicht2).T) + biases2

print(layer2_outputs)