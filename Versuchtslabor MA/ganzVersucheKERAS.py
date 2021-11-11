import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt

bild_groesse = (1024, 1024)
batch_groesse = 32

trainings_ds = tf.keras.preprocessing.image_dataset_from_directory()
val_ds = tf.keras.preprocessing.image_dataset_from_directory()

