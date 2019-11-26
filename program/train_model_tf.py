"""
CS4793

@authors: Melanie Bischoff, Malay Bhakta, Jordan Johnson
"""

import sys
import json
import os
import math
import random
import argparse
import logging
import platform
import cv2
import datetime
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Add, Input, Concatenate
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO)

# Set a random seed so runs are repeatable
np.random.seed(98234)  # need to set numpy seed since np.random.shuffle is used
tf.random.set_seed(98234)  # and tensorflow graph seed

damage_intensity_encoding = defaultdict(lambda: 0)
damage_intensity_encoding['destroyed'] = 3
damage_intensity_encoding['major-damage'] = 2
damage_intensity_encoding['minor-damage'] = 1
damage_intensity_encoding['no-damage'] = 0

"""
Goal: pretend like we know something about AI programming....
   
Description: 
after 
    
"""
def run_analysis():
    """
    Classification ranking?  Print out stuff?
    :return:
    """

def test_model(model):
    """

    :param model:
    :return:
    """
    return model

def train_model(model, train_data, train_labels):
    """
    look at the the xview2/baseline/model/damage_classification.py and xview2/baseline/model/model.py

    :param model:
    :return: model
    """
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=.15)
    model.fit(train_data, train_labels, epochs=10,
         validation_data=(val_data, val_labels))


    return model

def build_model():
    """
    look at the the xview2/baseline/model/damage_classification.py and xview2/baseline/model/model.py
    CNN?
    How many layers?  What weights?  What method? Optimizer, etc.

    :param data_dir:
    :return: model
    """

    #TODO: is this the right thing to start on?
    model = models.Sequential()

    #1024 x 1024 is how big the images are
    inputs = Input(shape=(1024, 1024))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1024, 1024)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    #print the output
    model.summary()

    #are more layers better?

    #Add Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    #model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    #model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='softmax'))
    model.add(layers.Dense(4, activation='relu'))

    #print the output
    model.summary()

    #TODO: add weights and generators and optimizers

    # adam optimizer
    adam = keras.optimizers.Adam(lr=0.0001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 decay=0.0,
                                 amsgrad=False)

    #Compile
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #TODO: return model created;
    return model

def colorize_mask_(mask, color_map=None):
    """
    Attaches a color palette to a PIL image. So long as the image is saved as a PNG, it will render visibly using the
    provided color map.
    :param mask: PIL image whose values are only 0 to 4 inclusive
    :param color_map: np.ndarray or list of 3-tuples with 5 rows
    :return:
    """
    color_map = color_map or np.array([(0, 0, 0),  # 0=background
                                       (255, 0, 0),  # no damage (or just 'building' for localization)
                                       (0, 255, 0),  # minor damage
                                       (0, 0, 255),  # major damage
                                       (128, 128, 0),  # destroyed
                                       ])
    assert color_map.shape == (5, 3)
    mask.putpalette(color_map.astype(np.uint8))
    return None

def load_json_and_img(data_dir):
    """
    Assumptions about the data are that it looks like ../data/train/{damage}/images and ../data/train/{damage}/labels

    :param data_dir:
    :return: images, labels
    """

    data_dir = data_dir + "/train/images"

    #If you wanted to Hard coded Path for training data...
    #data_dir = "C:/Dev/Workspaces/Python/AI Learning/program/data/train/images"

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
    #img_paths = np.asarray(image_paths)
    images = []
    labels = []

    for img_path in tqdm(image_paths):

        img_obj = Image.open(img_path)
        img_array = np.array(img_obj)
        images.append(img_array)

        #Get corresponding label for the current image
        label_path = img_path.replace('png', 'json').replace('images', 'labels')
        label_file = open(label_path)
        label_data = json.load(label_file)
        damage_type = "no-damage"

        for feat in label_data['features']['xy']:

            # only images post-disaster will have damage type
            try:
                damage_type = feat['properties']['subtype']
                if damage_type != "no-damage":
                    break

            except:  # pre-disaster damage is default no-damage
                damage_type = "no-damage"
                continue

        labels.append(damage_type)

    return images, labels

def main():
    """
    Main Method to parse the inputs, and methods to feed in json data, load images into numpy array, and build/train/test model.
    :return:
    """

    parser = argparse.ArgumentParser(description='Run XView2 Challenge Disaster Damage Classification Training & Evaluation')
    parser.add_argument('--data',
                        #default='..\\..\\data',
                        default='C:/Dev/Workspaces/Python/CS4793/xview2/data',
                        #Malay's dir
                        #default='C:/Dev/Workspaces/Python/CS4793/xview2/data',
                        metavar="/home/scratch1/cs4793/data",
                        help="Full path to the parent data directory")
    parser.add_argument('--val_split_pct',
                        required=False,
                        default=0.1,
                        metavar='Percentage to use for validation',
                        help="Percentage to use for validation")
    args = parser.parse_args()

    """
    Maybe these should be the steps:
        1. Identify & Mask the buildings for the pre disasters.
        2. Use those 
    """


    logging.info("Started Load JSON and Image into numpy")
    #image_array, label_array = load_json_and_img(args.data, float(args.val_split_pct))
    image_array, label_array = load_json_and_img(args.data)
    logging.info("Finished Load JSON and Image into numpy")

    logging.info("Started build model")
    model = build_model()
    logging.info("Finished build model")

    logging.info("Started train model")
    model = train_model(model, image_array, label_array)
    logging.info("Finished train model")

    logging.info("Started test model")
    model = test_model(model)
    logging.info("Finished test model")

    logging.info("Started analysis")
    run_analysis(model)
    logging.info("Finished analysis")


if __name__ == "__main__":
    main()
