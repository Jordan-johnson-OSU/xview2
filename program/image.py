import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pathlib
from pathlib import Path

p = Path("D:/Fall2019/AI/train/train/images")

print(p)
image_count = len(list(p.glob('*.png')))
print(image_count)

CLASS_NAMES_pre = np.array([item.name for item in p.glob('*_pre_*') if item.name != "LICENSE.txt"])
CLASS_NAMES_post = np.array([item.name for item in p.glob('*_post_*') if item.name != "LICENSE.txt"])

print(CLASS_NAMES_pre, ": ", CLASS_NAMES_pre.size,  CLASS_NAMES_post, ": ", CLASS_NAMES_post.size)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(1./255)

print(image_generator)

model = Sequential()

print(CLASS_NAMES_post[0].shape)

model.add(Convolution2D(32,3,3, input_shape= (64,64,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(units = 128, activation = 'softmax'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_dataGen = ImageDataGenerator(rescale=1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


