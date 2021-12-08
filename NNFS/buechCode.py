#Alles, was im Buech staat, wird da vo mier umgsetzt (inkl nnfs Packet). Die Datei = NNFS Sandbox


import numpy as np
import nnfs
from nnfs.datasets import spiral, spiral_data



nnfs.init()

#Dense Layer:
class Layer_Dense:
    #Initalisierung des Layers
    def __init__(self, n_inputs, n_neuronen, gwicht_regularizierer_l1=0, gwicht_regularizierer_l2=0, 
    bias_regularizierer_l1=0, bias_regularizierer_l2=0):
        #Gwicht und Biases Initalisierung
        self.gwicht = 0.01 * np.random.randn(n_inputs, n_neuronen)
        self.biases = np.zeros((1, n_neuronen))

        self.gwicht_regularizierer_l1 = gwicht_regularizierer_l1
        self.gwicht_regularizierer_l2 = gwicht_regularizierer_l2
        self.bias_regularizierer_l1 = bias_regularizierer_l1
        self.bias_regularizierer_l2 = bias_regularizierer_l2
    #Forward-Pass
    def forward(self, inputs, training):
        #Berechnung des Outputs
        self.inputs = inputs
        self.output = np.dot(inputs, self.gwicht) + self.biases

    def backward(self, dvalues): #Zruggpropagation (Kap 9 bis Siite 214)
        self.dgwicht = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.gwicht_regularizierer_l1 > 0:
            dL1 = np.ones_like(self.gwicht)
            dL1[self.gwicht < 0] = -1
            self.dgwicht += self.gwicht_regularizierer_l1 * dL1
        # L2 on weights
        if self.gwicht_regularizierer_l2 > 0:
            self.dgwicht += 2 * self.gwicht_regularizierer_l2 * \
            self.gwicht
        # L1 on biases
        if self.bias_regularizierer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizierer_l1 * dL1
        # L2 on biases
        if self.bias_regularizierer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizierer_l2 * \
            self.biases

        self.dinputs = np.dot(dvalues, self.gwicht.T)


class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Aktivierung_ReLU:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0

    def vorhersagen(self, outputs):
        return outputs

class Aktivierung_Softmax:
    
    def forward(self, inputs, training):
        self.inputs = inputs
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
    
    def vorhersagen(self, outputs):
        return np.argmax(outputs, axis=1)

class Aktivierung_Linear:

    def forward(self, inputs, training): #TRAINING
        #Just remember values
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        #derivative is 1, 1 * dvalues = dvalues - the chain rule (Kettenregel)
        self.dinputs = dvalues.copy()

    def vorhersagen(self, outputs):
        return outputs

class Aktivierung_Sigmoid:

    def forward(self, inputs, training):
        
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def vorhersagen(self, outputs):
        return (outputs > 0.5) * 1
class Verlust:

    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0
        # L1 regularization - weights
        # calculate only when factor greater than 0
        for layer in self.trainable_layers:

            if layer.gwicht_regularizierer_l1 > 0:
                regularization_loss += layer.gwicht_regularizierer_l1 * \
                np.sum(np.abs(layer.gwicht))
            # L2 regularization - weights
            if layer.gwicht_regularizierer_l2 > 0:
                regularization_loss += layer.gwicht_regularizierer_l2 * \
                np.sum(layer.gwicht *
                layer.gwicht)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizierer_l1 > 0:
                regularization_loss += layer.bias_regularizierer_l1 * \
                np.sum(np.abs(layer.biases))
            # L2 regularization - biases
            if layer.bias_regularizierer_l2 > 0:
                regularization_loss += layer.bias_regularizierer_l2 * \
                np.sum(layer.biases *
                layer.biases)
        
        return regularization_loss
    
    #Set/remember trainable layers:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def kalkulieren(self, output, y, *, include_reg=False):  #Wieso funktioniert forward ohni def (wA: Z.66+)
        #Calculate sample losses:
        sample_verluste = self.forward(output, y)
        #Calculate mean loss:
        data_verlust = np.mean(sample_verluste)

        if not include_reg:
            return data_verlust
        #Return the data and reg loss:
        return data_verlust, self.regularization_loss()

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

    #def __init__(self): #Kreierung von Aktivierung und Loss Funktionen Objekte
     #   self.aktivierung = Aktivierung_Softmax()
      #  self.verlust = Verlust_CatCrossEnt()

    #def forward(self, inputs, y_true): #??? y_true???
     #   self.aktivierung.forward(inputs) #Output Layer Aktivierungsfunktion
      #  self.output = self.aktivierung.output #Set the ouput
       # return self.verlust.kalkulieren(self.output, y_true) #Lossvalue berechnen und geben

    def backward(self, dvalues, y_true):
        samples = len(dvalues) #Anzahl samples

        if len(y_true.shape) == 2: #"If labels are 1hot encoded, turn them into discrete values"
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy() #Kopieren für sichere Modifizierung
        self.dinputs[range(samples), y_true] -= 1 #Gradient berechnen
        self.dinputs = self.dinputs / samples #Gradienten normalisieren

class Verlust_BinCrossent(Verlust):

    def forward(self, y_pred, y_true):

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_verluste = -(y_true * np.log(y_pred_clipped) + (1-y_true)* np.log(1-y_pred_clipped))
        sample_verluste = np.mean(sample_verluste, axis=1)

        return sample_verluste

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_dvalues - (1-y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Verlust):

    def forward(self, y_pred, y_true):

        sample_losses = np.mean((y_true - y_pred)**2, axis=1)

        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs

        self.dinputs = self.dinputs / samples

class Verlust_MeanAbsoluteError(Verlust):

    def forward(self, y_pred, y_true):
        sample_verluste = np.mean(np.abs(y_true - y_pred), axis=1)
        return sample_verluste

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Optimizer_SGD: #Gute Start-Lernrate = 1.0, decay runter zu 0.1
    #Initalisierung Optimizer. Lernrate = 1 - Basis für diesen Optimizer
    def __init__(self, lern_rate=1.0, decay=0., momentum = 0.):
        self.lern_rate = lern_rate
        self.momentane_lern_rate = lern_rate
        self.decay = decay
        self.iterationen = 0
        self.momentum = momentum
    #Einmal aufrufen BEVOR irgendein Parameter updatet
    def pre_update_params(self):
        if self.decay:
            self.momentane_lern_rate = self.lern_rate * \
                (1. / (1. + self.decay * self.iterationen))
    #Parameter updaten
    def update_params(self, layer):
        #If we use Momentum
        if self.momentum:
            #If layer does not contain momentum arrays, create them filled with 0s
            if not hasattr(layer, 'gwicht_momenta'):
                layer.gwicht_momenta = np.zeros_like(layer.gwicht)
                #If there is no momentum array for weights the array doesnt exst for biases yet either
                layer.bias_momenta = np.zeros_like(layer.biases)
            #Build weight updates with momentum - take previous updates multiplied by retain factor
            #and update with current gradients
            gwicht_updates = \
                self.momentum * layer.gwicht_momenta - \
                self.momentane_lern_rate * layer.dgwicht
            layer.gwicht_momenta = gwicht_updates
            #Build Bias updates
            bias_updates = \
                self.momentum * layer.bias_momenta - \
                self.momentane_lern_rate * layer.dbiases
            layer.bias_momenta = bias_updates
        #Vanilla SGD updates (as before mometum update)
        else: 

            gwicht_updates = -self.momentane_lern_rate * \
                layer.dgwicht
            bias_updates = -self.momentane_lern_rate * \
                layer.dbiases

        #Update weights and biases using either vanilla or momentum updates:
        layer.gwicht += gwicht_updates
        layer.biases += bias_updates
    #Einmal aufrufen NACHDEM Parameter updatet
    def post_update_params(self):
        self.iterationen += 1

#AdaGrad: 
class Optimizer_AdaGrad:
    #Initalisierung Optimizer. Lernrate = 1 - Basis für diesen Optimizer
    def __init__(self, lern_rate=1.0, decay=0., epsilon=1e-7):
        self.lern_rate = lern_rate
        self.momentane_lern_rate = lern_rate
        self.decay = decay
        self.iterationen = 0
        self.epsilon = epsilon
    #Einmal aufrufen BEVOR irgendein Parameter updatet
    def pre_update_params(self):
        if self.decay:
            self.momentane_lern_rate = self.lern_rate * \
                (1. / (1. + self.decay * self.iterationen))
    #Parameter updaten
    def update_params(self, layer):
        #If we use Momentum
        if not hasattr(layer, 'gwicht_cache'):
            layer.gwicht_cache = np.zeros_like(layer.gwicht)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.gwicht_cache += layer.dgwicht ** 2
        layer.bias_cache += layer.dbiases ** 2


        layer.gwicht += -self.momentane_lern_rate * \
                        layer.dgwicht / \
                        (np.sqrt(layer.gwicht_cache) + self.epsilon)
        layer.biases += -self.momentane_lern_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)
    def post_update_params(self):
        self.iterationen += 1



#RMSProp:
class Optimizer_RMSProp:
    #Initalisierung Optimizer. Lernrate = 1 - Basis für diesen Optimizer
    def __init__(self, lern_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.lern_rate = lern_rate
        self.momentane_lern_rate = lern_rate
        self.decay = decay
        self.iterationen = 0
        self.epsilon = epsilon
        self.rho = rho
    #Einmal aufrufen BEVOR irgendein Parameter updatet
    def pre_update_params(self):
        if self.decay:
            self.momentane_lern_rate = self.lern_rate * \
                (1. / (1. + self.decay * self.iterationen))
    #Parameter updaten
    def update_params(self, layer):
        #If we use Momentum
        if not hasattr(layer, 'gwicht_cache'):
            layer.gwicht_cache = np.zeros_like(layer.gwicht)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.gwicht_cache = self.rho * layer.gwicht_cache + (1-self.rho) * layer.dgwicht ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases ** 2


        layer.gwicht += -self.momentane_lern_rate * \
                        layer.dgwicht / \
                        (np.sqrt(layer.gwicht_cache) + self.epsilon)
        layer.biases += -self.momentane_lern_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)
    def post_update_params(self):
        self.iterationen += 1

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

class Layer_Input:
    def forward(self, inputs, training): #TRAINING

        self.output = inputs

class Genauigkeit:

    def kalkulieren(self, vorhersagen, y):

        vergleiche = self.vergleichen(vorhersagen, y)

        genauigkeit = np.mean(vergleiche)

        return genauigkeit

class Genauigkeit_Categorial(Genauigkeit):

    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def vergleichen(self, vorhersagen, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
            return vorhersagen == y
class Genauigkeit_Regression(Genauigkeit):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def vergleichen(self, vorhersagen, y):
        return np.absolute(vorhersagen - y) < self.precision
class Model:

    def __init__(self):
        #Create a list of network objects
        self.layers = []
        self.softmax_classifier_output = None
    #Add objects to the model:
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, genauigkeit): # Set loss and optimizer
        self.loss = loss
        self.optimizer = optimizer
        self.genauigkeit = genauigkeit
    
    def finalize(self):
        #Create and set the inputs layer:
        self.input_layer = Layer_Input()

        #Count all the objects:
        layer_count = len(self.layers)

        self.trainable_layers = [] #Initialize a list containing trainable layers:

        #Iterate the objects:
        for i in range(layer_count):

            #If its the first layer, the prev. layer object is the input layer:
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            #All layers except for the first and the last:
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            #The last layer - the next object is the loss:
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            #If layer contains an attribute called "gwicht", its a trainable layer - add to list
            if hasattr(self.layers[i], 'gwicht'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)


        if isinstance(self.layers[-1], Aktivierung_Softmax) and isinstance(self.loss, Verlust_CatCrossEnt):
            self.softmax_classifier_output = Aktivierung_Softmax_Verlust_CatCrossEnt()

    def train(self, X, y, *, epochen=1, print_every=1, validation_data=None): #Model trainieren
        
        self.genauigkeit.init(y) #Vlt von Regression auf Seite 488!!!
        #Main training loop:
        for epoch in range(1, epochen+1):

            output = self.forward(X, training=True)

            data_loss, regularization_loss = self.loss.kalkulieren(output, y, include_reg=True)
            loss = data_loss + regularization_loss

            vorhersagen = self.output_layer_activation.vorhersagen(output)
            genauigkeit = self.genauigkeit.kalkulieren(vorhersagen, y)
        
            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epochen % print_every:
                print(f'epoche: {epoche},' +
                      f'genau: {genauigkeit:.3f},' +
                    f'loss: {loss:.3f}, (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.momentane_lern_rate}')
    
        if validation_data is not None:
            X_val, y_val = validation_data
            output = self.forward(X_val, training=False)
            loss = self.loss.kalkulieren(output, y_val)
            vorhersagen = self.output_layer_activation.vorhersagen(output)
            genauigkeit = self.genauigkeit.kalkulieren(vorhersagen, y_val)

            print(f'validation, ' + 
                f'genau: {genauigkeit:.3f}, '+
                f'loss: {loss:.3f}')

    def forward(self, X, training):

        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            
            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)



X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

#Model initalisieren:
model = Model()

#Add layers:
model.add(Layer_Dense(2, 512, gwicht_regularizierer_l2=5e-4, bias_regularizierer_l2=5e-4))
model.add(Aktivierung_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Aktivierung_Softmax())

#IM BUECH ISCH loss=Loss_MeanSquaredError()
model.set(
    loss=Verlust_CatCrossEnt(), optimizer=HerrAdam(lern_rate=0.05, decay=5e-5), 
    genauigkeit=Genauigkeit_Categorial())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochen=10000, print_every=100)
