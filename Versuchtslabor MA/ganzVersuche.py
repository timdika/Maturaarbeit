#Ganze Versuchs-Sandbox

import os, sys
import numpy as np
import PIL
from PIL import Image
import random


#LAYER:
class Layer_Dense:
    def __init__(self, n_inputs, n_neuronen):
        self.gwicht = 0.01 * np.random.randn(n_inputs, n_neuronen)
        self.biases = np.zeros((1, n_neuronen))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.gwicht) + self.biases

        self.inputs = inputs

    def backwards(self, dvalues):
        self.dgwicht = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.gwicht.T)

class Aktivierung_ReLu:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backwards(self, dvalues):
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


class HerrAdam: #Gute Start-Lernrate = 0.001, decaying runter zu 0.00001
    #Initalisierung Optimizer. Lernrate = 1 - Basis für diesen Optimizer
    def __init__(self, lern_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.lern_rate = lern_rate
        self.momentane_lern_rate = lern_rate
        self.decay = decay
        self.iterationen = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    #Einmal aufrufen BEVOR irgendein Parameter updatet
    def pre_update_params(self):
        if self.decay:
            self.momentane_lern_rate = self.lern_rate * \
                (1. / (1. + self.decay * self.iterationen))
    #Parameter updaten
    def update_params(self, layer):
        #If we use Momentum
        if not hasattr(layer, 'gwicht_cache'):
            layer.gwicht_momenta = np.zeros_like(layer.gwicht)
            layer.gwicht_cache = np.zeros_like(layer.gwicht)
            layer.bias_momenta = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.gwicht_momenta = self.beta_1 * layer.gwicht_momenta + (1-self.beta_1) * layer.dgwicht
        layer.bias_momenta = self.beta_1 * layer.bias_momenta + (1-self.beta_1) * layer.dbiases

        gwicht_momenta_korrigiert = layer.gwicht_momenta / (1-self.beta_1 ** (self.iterationen + 1))
        bias_momenta_korrigiert = layer.bias_momenta / (1-self.beta_1 ** (self.iterationen + 1))
        
        layer.gwicht_cache = self.beta_2 * layer.gwicht_cache + (1-self.beta_2) * layer.dgwicht ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases ** 2

        gwicht_cache_korrigiert = layer.gwicht_cache / (1-self.beta_2 ** (self.iterationen + 1))
        bias_cache_korrigiert = layer.bias_cache / (1-self.beta_2 ** (self.iterationen + 1))


        layer.gwicht += -self.momentane_lern_rate * gwicht_momenta_korrigiert / (np.sqrt(gwicht_cache_korrigiert) + self.epsilon)
        layer.biases += -self.momentane_lern_rate * bias_momenta_korrigiert / (np.sqrt(bias_cache_korrigiert) + self.epsilon)
    def post_update_params(self):
        self.iterationen += 1


