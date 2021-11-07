#Versuch Datenscan
# LINK: https://www.kite.com/python/answers/how-to-convert-an-image-to-an-array-in-python
import os, sys
import numpy as np
from PIL import Image

im = Image.open("testSchlechtRoent.jpg")
im_array = np.array(im)

print(im_array.shape)
print("")
print(im_array) #Was macht das?

