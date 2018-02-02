import matplotlib.pyplot as plt
import matplotlib.image as im
from PIL import Image
import numpy as np
img = Image.open('1.png')
img.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place
#imgplot = plt.imshow(img)
imgplot = plt.imshow(img, interpolation="bicubic")
plt.show()
