from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tqdm import tqdm
from PIL import Image
import json
import time
import h5py


def load_json_and_img(data_dir):
    """
    Assumptions about the data are that it looks like ../data/train/{damage}/images and ../data/train/{damage}/labels

    :param data_dir:
    :return: images, labels
    """

    data_dir = data_dir + "/train/images"

    # If you wanted to Hard coded Path for training data...
    # data_dir = "C:/Dev/Workspaces/Python/AI Learning/program/data/train/images"

    image_paths = []
    """
    for pic in os.listdir(data_dir):
        image_paths.append((str(data_dir) + "/" + pic))
    """

    """
    df = pd.read_csv(data_dir + "/xBD_csv/train.csv")
    class_weights = compute_class_weight('balanced', np.unique(df['labels'].to_list()), df['labels'].to_list());
    weights = dict(enumerate(class_weights))

    samples = df['uuid'].count()
    steps = np.ceil(samples/64)
    """

    image_paths.extend([(data_dir + "/" + pic) for pic in os.listdir(data_dir)])
    # img_paths = np.asarray(image_paths)
    images = []
    labels = []

    i = 0
    for img_path in tqdm(image_paths):

        img_obj = Image.open(img_path)
        # resize the image
        IMAGE_SHAPE = (512, 512)
        img_obj = img_obj.resize(IMAGE_SHAPE)

        img_array = np.array(img_obj)
        images.append(img_array)

        # images.append(img_array)

        # labels.append(damage_type)
    images = np.asarray(images)
    return images


def load_test_images(data_dir):
    data_dir = data_dir + "/test/images"

    # If you wanted to Hard coded Path for training data...
    # data_dir = "C:/Dev/Workspaces/Python/AI Learning/program/data/train/images"

    image_paths = []

    image_paths.extend([(data_dir + "/" + pic) for pic in os.listdir(data_dir)])
    # img_paths = np.asarray(image_paths)
    images = []

    i = 0
    for img_path in tqdm(image_paths):
        if i == 66:
            break
        img_obj = Image.open(img_path)
        img_array = np.array(img_obj)
        images.append(img_array)
        # images.append(img_array)

        i = i + 1
    images = np.asarray(images)
    return images


test_images = load_test_images('D:/Fall2019/AI/train/test')
test_resize_images = tf.compat.v2.image.resize_with_pad(
    test_images,
    512,
    512,
    # method=ResizeMethod.BILINEAR
)

train_images, train_labels = load_json_and_img('D:/Fall2019/AI/train')

#loading the model with name in " " and in same directory.
model = tf.keras.models.load_model("Model")

model.summary()

print(train_labels)
label1 = LabelEncoder()
train_labels = label1.fit_transform(train_labels)
print(train_labels)

predict = model.predict(train_labels)
print(predict)
