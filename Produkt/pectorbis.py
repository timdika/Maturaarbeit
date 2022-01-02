import os
import numpy as np
import matplotlib.pyplot as plt
import cv2




def Datensatz_laden(datensatz, path):

    labels = os.listdir(os.path.join(path, datensatz))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, datensatz, label)):
            image = cv2.imread(os.path.join(path, datensatz, label, file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data_med(path):

    X, y = Datensatz_laden('train', path)
    X_test, y_test = Datensatz_laden('test', path)

    return X, y, X_test, y_test


X, y, X_test, y_test = create_data_med('C:/Users/timof/Desktop/Dokumente_und_Datensatz_MA/ZweiFallDatensatz')

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

keys = np.array(range(X.shape[0]))
print(keys)
np.random.shuffle(keys)
print(keys)

X = X[keys]
y = y[keys]

BATCH_SIZE = 2

schritte = X.shape[0] // BATCH_SIZE

if schritte * BATCH_SIZE < X.shape[0]:
    schritte += 1