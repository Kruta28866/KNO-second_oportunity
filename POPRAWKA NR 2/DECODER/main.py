import tensorflow as tf
from sympy import false
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist # load dataset

keras.resize(fashion_mnist)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)

print(train_labels[:10])

class_names = ['top', 'trouser', 'pullover','dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

plt.figure()
plt.imshow(train_images[3])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential
