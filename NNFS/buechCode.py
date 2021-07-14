#Alles, was im Buech staat, wird da vo mier umgsetzt (inkl nnfs Packet). Die Datei = NNFS Sandbox

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#Dense Layer:
class Layer_Dense:
    #Initalisierung des Layers
    def __init__(self, n_inputs, n_neuronen):
        #Gwicht und Biases Initalisierung
        self.gwicht = 0.01 * np.random.randn(n_inputs, n_neuronen)
        self.biases = np.zeros((1, n_neuronen))
    #Forward-Pass
    def forward(self, inputs):
        #Berechnung des Outputs
        self.output = np.dot(inputs, self.gwicht) + self.biases
#Erstellen eines Datensets
X, y = spiral_data(samples=100, classes=3)
#Kreation eines Layers mit 2 Inputs und 3 Outputss
dense1 = Layer_Dense(2, 3)
#Forward-Pass der Daten durch Layer
dense1.forward(X)
#Print des Ouputs
print(dense1.output[:5])