#Versuch Datenscan
# LINK: https://www.kite.com/python/answers/how-to-convert-an-image-to-an-array-in-python
import os, sys
import numpy as np
from PIL import Image

im = Image.open("C:/Users/timof/Desktop/MA/Versuchtslabor MA/testSchlechtRoent.jpg").convert('L')
im.save('pil-grayscale.png')
#im_array = np.array(im)

#print(im_array.shape)
#print("")
#print(im_array) #Was macht das?
#print(np.asarray(im)) #Ganze Liste von einzelnen Pixel - Jeder Pixel als drei Werte
#print(im.mode)
