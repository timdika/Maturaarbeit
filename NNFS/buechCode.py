#Alles, was im Buech staat, wird da vo mier umgsetzt (inkl nnfs Packet). Die Datei = NNFS Sandbox

import numpy as np
import math
import random
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

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

        self.inputs = inputs #Zwischenspeicherung für Zruggprop?
    def backward(self, dvalues): #Zruggpropagation (Kap 9 bis Siite 214)
        self.dgwicht = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.gwicht.T)

class Aktivierung_ReLU: #???

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues): #Zruggpropagation - ???
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0

class Aktivierung_Softmax:
    
    def forward(self, inputs): #???
        exp_werte = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        wahrscheinlichkeiten = exp_werte / np.sum(exp_werte, axis=1, keepdims=True)
        self.output = wahrscheinlichkeiten

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) #Wir machen einen uninitalisierten Array

        #Enumerate outputs and gradients:
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)): #???
            single_output = single_output.reshape(-1, 1) #Flatten out array
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Verlust:

    def kalulieren(self, output, y): #??? y??? Wieso funktioniert forward ohni def (wA: Z.66+)
        sample_verluste = self.forward(output, y)
        data_verlust = np.mean(sample_verluste)
        return data_verlust

class Verlust_CatCrossEnt(Verlust):

    def forward(self, y_pred, y_true): #??? Was genau y_pred & y_true???
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)#??? Was isch das???

        if len(y_true.shape) == 1: #??? .shape und wieso == 1???
            korrekte_sicherheiten = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            korrekte_sicherheiten = np.sum(y_pred_clipped*y_true, axis=1)

        neg_log_wahrscheinlichkeiten = -np.log(korrekte_sicherheiten)
        return neg_log_wahrscheinlichkeiten

    def backward(self, dvalues, y_true):
        samples = len(dvalues) #Anzahl samples
        labels = len(dvalues[0]) #Anzahl Labels in jedem Sample

        if len(y_true.shape) == 1: #"If labels are sparse, turn them into 1hot vectors" #???
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues #Gradient ausrechenen #??? Was Gradient und wieso so???
        self.dinputs = self.dinputs / samples #Gradient normalisieren #???

#Softmax-Klassifizierer - kombiniert SoftmaxAktivierung und CrossEntLoss für schnelleres backward
class Aktivierung_Softmax_Verlust_CatCrossEnt():

    def __init__(self): #Kreierung von Aktivierung und Loss Funktionen Objekte
        self.aktivierung = Aktivierung_Softmax()
        self.verlust = Verlust_CatCrossEnt()

    def forward(self, inputs, y_true): #??? y_true???
        self.aktivierung.forward(inputs) #Output Layer Aktivierungsfunktion
        self.output = self.aktivierung.output #Set the ouput
        return self.verlust.kalulieren(self.output, y_true) #Lossvalue berechnen und geben

    def backward(self, dvalues, y_true):
        samples = len(dvalues) #Anzahl samples

        if len(y_true.shape) == 2: #"If labels are 1hot encoded, turn them into discrete values"
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy() #Kopieren für sichere Modifizierung
        self.dinputs[range(samples), y_true] -= 1 #Gradient berechnen
        self.dinputs = self.dinputs / samples #Gradienten normalisieren

class Optimizer_SGD:
    #Initalisierung Optimizer. Lernrate = 1 - Basis für diesen Optimizer
    def __init__(self, lern_rate=1.0, decay=0):
        self.lern_rate = lern_rate
        self.momentane_lern_rate = lern_rate
        self.decay = decay
        self.iterationen = 0
    #Einmal aufrufen BEVOR irgendein Parameter updatet
    def pre_update_params(self):
        if self.decay:
            self.momentane_lern_rate = self.lern_rate * \
                (1. / (1. + self.decay * self.iterationen))
    #Parameter updaten
    def update_params(self, layer):
        layer.gwicht += -self.lern_rate * layer.dgwicht
        layer.biases += -self.lern_rate * layer.dbiases
    #Einmal aufrufen NACHDEM Parameter updatet
    def post_update_params(self):
        self.iterationen += 1


#Erstellen eines Datensets
X, y = spiral_data(samples=100, classes=3)
#Kreation eines Layers mit 2 Inputs und 3 Outputss
dense1 = Layer_Dense(2, 64)

aktivierung1 = Aktivierung_ReLU()


dense2 = Layer_Dense(64, 3)

loss_aktivierung = Aktivierung_Softmax_Verlust_CatCrossEnt()

optimizer = Optimizer_SGD(decay=1e-3)


for epoche in range(10001):
    #Forward-Pass der Daten durch Layer
    dense1.forward(X)

    aktivierung1.forward(dense1.output)

    dense2.forward(aktivierung1.output)


    loss = loss_aktivierung.forward(dense2.output, y)

    #print(loss_aktivierung.output[:5])#Die erschte 5 Samples

    #print("loss: ", loss)



    vorhersagen = np.argmax(loss_aktivierung.output, axis=1) #???
    if len(y.shape) == 2: #??? .shape und ==2???
        y = np.argmax(y, axis=1)
    genauigkeit = np.mean(vorhersagen==y) #??? Wieso mean vo == ???
    
    if not epoche % 100:
        print(f'Epoche: {epoche}, ' +
        f'Genau: {genauigkeit:.3f}, ' +
        f'Loss: {loss:.3f}, ' +
        f'LR: {optimizer.momentane_lern_rate}')

    #Zruggpass:
    loss_aktivierung.backward(loss_aktivierung.output, y)
    dense2.backward(loss_aktivierung.dinputs)
    aktivierung1.backward(dense2.dinputs)
    dense1.backward(aktivierung1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    #Gradienten:
    #print(dense1.dgwicht)
    #print(dense1.dbiases)
    #print(dense2.dgwicht)
    #print(dense2.dbiases)
    