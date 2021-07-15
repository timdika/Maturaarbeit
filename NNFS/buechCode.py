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

class Aktivierung_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Aktivierung_Softmax:
    
    def forward(self, inputs):
        exp_werte = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        wahrscheinlichkeiten = exp_werte / np.sum(exp_werte, axis=1, keepdims=True)
        self.output = wahrscheinlichkeiten
#Erstellen eines Datensets
X, y = spiral_data(samples=100, classes=3)
#Kreation eines Layers mit 2 Inputs und 3 Outputss
dense1 = Layer_Dense(2, 3)

aktivierung1 = Aktivierung_ReLU()


dense2 = Layer_Dense(3, 3)
aktivierung2 = Aktivierung_Softmax()
#Forward-Pass der Daten durch Layer
dense1.forward(X)

aktivierung1.forward(dense1.output)

dense2.forward(aktivierung1.output)

aktivierung2.forward(dense2.output)
#Print des Ouputs
#print(dense1.output[:5])
#print(aktivierung1.output[:5])
print(aktivierung2.output[:5])